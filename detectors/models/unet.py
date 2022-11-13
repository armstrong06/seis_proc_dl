import torch
import h5py 
import numpy as np
import os 
from torch.utils.data.sampler import SubsetRandomSampler
import sys
sys.path.insert(0, "/home/armstrong/Research/git_repos/seis-proc-dl")
from model.base_model import BaseModel
from utils.model_helpers import NumpyDataset
sys.path.insert(0, "/home/armstrong/Research/git_repos/seis-proc-dl/detectors")
from executor.unet_trainer import UNetTrainer
from evaluation.unet_evaluator import UNetEvaluator
from evaluation.mulitple_model_evaluation import MultiModelEval
import pandas as pd
import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def set_deterministic_random_seed(seed):
    """Followed instructions from
    https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--
    VmlldzoxMDA2MDQy"""
    print("Setting deterministic seed...")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

class UNet(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
        self.device = torch.device(self.config.train.torch_device)
        self.model = self.build(self.config.model.num_channels, self.config.model.num_classes)
        self.minimum_presigmoid_value = self.config.model.minimum_presigmoid_value

        self.phase_type = self.config.model.phase_type
        self.learning_rate = self.config.train.learning_rate
        self.batch_size = self.config.train.batch_size
        self.detection_threshold = self.config.train.detection_threshold
        self.train_file = self.config.train.train_hdf5_file
        self.validation_file = self.config.train.validation_hdf5_file
        self.model_out_dir = self.config.train.model_out_directory
        
        self.model_path = self.model_out_dir #self.make_model_path(self.model_out_dir)
        self.evaluator = None
        self.center_window = self.config.data.maxlag

        self.train_epochs = self.config.train.epochs
        self.evaluation_epoch = -1  # Epoch that is being evaluate - -1 if model not trained/loaded

        self.results_out_dir = None #f"{self.model_out_dir}/results"

        # If the seed is set in the configs file, default set it to 3940
        # To not set a random seed throughout, use a seed < 0
        try:
            seed = self.config.model.seed
        except:
            seed = 3940

        if seed >= 0:
            # np.random.seed(seed)
            # torch.manual_seed(seed)
            print("Setting random seed to", seed)
            set_deterministic_random_seed(seed)
        else:
            print("No random seed set")
            seed = None

        self.seed = seed


    def set_results_out_dir(self, test_type, have_df=False):
        outdir = f"{self.model_out_dir}/{test_type}_results"
        if have_df:
            outdir = f"{outdir}_sep"

        if (os.path.exists(outdir)):
            print(f"output directory {outdir} already exists.")
            outdir = f"{outdir}1"

        self.results_out_dir = outdir

    @staticmethod
    def read_data(data_file):
        # TODO: This didn't work if data_file path is relative to the main script location
        with h5py.File(data_file, "r") as f:
            X = f['X'][:]
            Y = f['Y'][:]
            T = f['Pick_index'][:] 

        return X, Y, T

    def load_model_state(self, model_in):
        if self.evaluation_epoch > 0:
            msg = "Can't load model state after training is complete..."
            raise ValueError(msg)

        if (not os.path.exists(model_in)):
            msg = f"Model {model_in} does not exist"
            raise ValueError(msg)

        print(f"Loading model state with {model_in}")
        check_point = torch.load(model_in)
        self.model.load_state_dict(check_point['model_state_dict'])
        print(check_point["epoch"], check_point["loss"])

        self.evaluation_epoch = check_point["epoch"]

    def load_data(self, data_file, shuffle=True):
        # TODO: This didn't work if data_file path is relative to the main script location
        X, Y, _ = self.read_data(data_file)  
            
        dataset = NumpyDataset(X, Y)
        n_samples = len(dataset)
        indices = list(range(n_samples))

        sampler=None
        if shuffle:
            # Randomize rows since I packed the data then the noise
            shuffled_indices = np.random.choice(indices, size=n_samples, replace=False)
            sampler = SubsetRandomSampler(shuffled_indices)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
        )

        return loader

    def build(self, num_channels, num_classes):
        return UNetModel(num_channels=num_channels, num_classes=num_classes).to(self.device)

    def train(self):
        train_loader = self.load_data(self.train_file)
        validation_loader = None
        if self.validation_file is not None:
            validation_loader = self.load_data(self.validation_file)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss = torch.nn.BCEWithLogitsLoss()

        trainer = UNetTrainer(self.model, optimizer, loss, self.model_path, self.device, 
                        phase_type=self.phase_type, detection_thresh=self.detection_threshold, 
                        minimum_presigmoid_value=self.minimum_presigmoid_value, random_seed=self.seed)
        trainer.train(train_loader, self.train_epochs, val_loader=validation_loader)
        self.evaluation_epoch = self.train_epochs

    def evaluate(self, test_file, test_type, pick_method="single"):
        "Evaluate dataset on the final model"
        if self.evaluation_epoch < 0:
            print("No model state loaded - load or train a model first")
            return

        self.set_results_out_dir(test_type)

        #test_loader = self.load_data(test_file, shuffle=False)
        X, Y, T = self.read_data(test_file)
        evaluator = UNetEvaluator(self.batch_size, self.device, self.center_window, 
                                minimum_presigmoid_value=self.minimum_presigmoid_value, 
                                random_seed=self.seed)
        evaluator.set_model(self.model)
        post_probs, pick_info = evaluator.apply_model(X, pick_method)
        results = evaluator.tabulate_metrics(T, pick_info[1], pick_info[0], str(self.evaluation_epoch))
        evaluator.save_posterior_probs(post_probs, pick_info[1], pick_info[0], self.results_out_dir, self.evaluation_epoch)
        resids = evaluator.calculate_residuals(T, pick_info[0], pick_info[1], self.evaluation_epoch)
        evaluator.save_result(results, f"{self.results_out_dir}/{self.evaluation_epoch}_summary.csv")
        evaluator.save_result(resids, f"{self.results_out_dir}/{self.evaluation_epoch}_residuals.csv")

    # TODO: Make this a class method?
    # TODO: Add way to evalute the selected pre-trained model first?
    def evaluate_specified_models_old(self, test_file, epochs, test_type, batch_size=None, 
                                    tols=np.linspace(0.05, 0.95, 21), pick_method="single", mew=False, df=None):
        if self.evaluation_epoch >= 0:
            print("Can't do multi-model evaluation with model state loaded")
        
        have_df = False
        if df is not None:
            have_df = True
            df = pd.read_csv(df)

        self.set_results_out_dir(test_type, have_df=have_df)

        if batch_size is None:
            batch_size = self.batch_size
        single_evaluator = UNetEvaluator(batch_size, self.device, self.center_window, 
                                minimum_presigmoid_value=self.minimum_presigmoid_value)
        multi_evaluator = MultiModelEval(self.model, self.model_path, epochs, single_evaluator, self.results_out_dir)
        
        if mew:
            multi_evaluator.evaluate_over_models_mew(test_file, tols, df=df)
        else:
            multi_evaluator.evaluate_over_models(test_file, tols, pick_method, df=df)
    
    def evaluate_specified_models_new(self, test_file, epochs, test_type, batch_size=None, 
                                    tols=np.linspace(0.05, 0.95, 21), pick_method="single", mew=False, df=None):
        if self.evaluation_epoch >= 0:
            print("Can't do multi-model evaluation with model state loaded")
        
        have_df = False
        if df is not None:
            have_df = True
            df = pd.read_csv(df)

        self.set_results_out_dir(test_type, have_df=have_df)

        if batch_size is None:
            batch_size = self.batch_size

        evaluator = UNetEvaluator(batch_size, self.device, self.center_window, 
                                  minimum_presigmoid_value=self.minimum_presigmoid_value, 
                                  random_seed=self.seed)
        
        evaluator.set_model(self.model)
        
        evaluator.evaluate_over_models(test_file, epochs, self.model_path, self.results_out_dir, tols, pick_method, df=df)
               

    def make_model_path(self, path):
        return f'{path}/{self.phase_type}_models_{self.batch_size}_{self.learning_rate}'

class UNetModel(torch.nn.Module):

    def __init__(self, num_channels=3, num_classes=1, k=3):
        super(UNetModel, self).__init__()
        from torch.nn import MaxPool1d, Conv1d, ConvTranspose1d
        self.relu = torch.nn.ReLU()
        #k = 3 #7
        p = k//2
        self.maxpool = MaxPool1d(kernel_size=2, stride=2)
        self.conv11 = Conv1d(num_channels, 64, kernel_size=k, padding=p)
        self.conv12 = Conv1d(64, 64, kernel_size=k, padding=p)
        self.bn1 = torch.nn.BatchNorm1d(64, eps=1e-05, momentum=0.1)

        self.conv21 = Conv1d(64, 128, kernel_size=k, padding=p)
        self.conv22 = Conv1d(128, 128, kernel_size=k, padding=p)
        self.bn2 = torch.nn.BatchNorm1d(128, eps=1e-05, momentum=0.1)

        self.conv31 = Conv1d(128, 256, kernel_size=k, padding=p)
        self.conv32 = Conv1d(256, 256, kernel_size=k, padding=p)
        self.bn3 = torch.nn.BatchNorm1d(256, eps=1e-05, momentum=0.1)

        self.conv41 = Conv1d(256, 512, kernel_size=k, padding=p)
        self.conv42 = Conv1d(512, 512, kernel_size=k, padding=p)
        self.bn4 = torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1)

        self.conv51 = Conv1d(512, 1024, kernel_size=k, padding=p)
        self.conv52 = Conv1d(1024, 1024, kernel_size=k, padding=p)
        self.bn5 = torch.nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1)

        self.uconv6 = ConvTranspose1d(1024, 512, kernel_size=2, stride=2)
        self.conv61 = Conv1d(1024, 512, kernel_size=k, padding=p)
        self.conv62 = Conv1d(512, 512, kernel_size=k, padding=p)
        self.bn6 = torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1)

        self.uconv7 = ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.conv71 = Conv1d(512, 256, kernel_size=k, padding=p)
        self.conv72 = Conv1d(256, 256, kernel_size=k, padding=p)
        self.bn7 = torch.nn.BatchNorm1d(256, eps=1e-05, momentum=0.1)

        self.uconv8 = ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.conv81 = Conv1d(256, 128, kernel_size=k, padding=p)
        self.conv82 = Conv1d(128, 128, kernel_size=k, padding=p)
        self.bn8 = torch.nn.BatchNorm1d(128, eps=1e-05, momentum=0.1)

        self.uconv9 = ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        self.conv91 = Conv1d(128, 64, kernel_size=k, padding=p)
        self.conv92 = Conv1d(64, 64, kernel_size=k, padding=p)
        self.bn9 = torch.nn.BatchNorm1d(64, eps=1e-05, momentum=0.1)

        self.conv93 = Conv1d(64, num_classes, kernel_size=1, padding=0)

        #self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv11(x)
        x = self.relu(x)
        x = self.conv12(x)
        x1d = self.relu(x)
        x1d = self.bn1(x1d)
        x = self.maxpool(x1d)
        #print('x1d.shape:', x1d.shape)

        x = self.conv21(x)
        x = self.relu(x)
        x = self.conv22(x)
        x2d = self.relu(x)
        x2d = self.bn2(x2d)
        x = self.maxpool(x2d)
        #print('x2d.shape', x2d.shape)

        x = self.conv31(x)
        x = self.relu(x)
        x = self.conv32(x)
        x3d = self.relu(x)
        x3d = self.bn3(x3d)
        x = self.maxpool(x3d)
        #print('x3d.shape:', x3d.shape)

        x = self.conv41(x)
        x = self.relu(x)
        x = self.conv42(x)
        x4d = self.relu(x)
        x4d = self.bn4(x4d)
        x = self.maxpool(x4d)
        #print('x4d.shape:', x4d.shape)

        x = self.conv51(x)
        x = self.relu(x)
        x = self.conv52(x)
        x5d = self.relu(x)
        x5d = self.bn5(x5d)
        # TODO: add maxpool layer here 
        #print(x5d.shape)

        x6u = self.uconv6(x5d)
        #print(x6u.shape, x4d.shape)
        x = torch.cat((x4d, x6u), 1)
        x = self.conv61(x)
        x = self.relu(x)
        x = self.conv62(x)
        x = self.relu(x)
        x = self.bn6(x)

        x7u = self.uconv7(x)
        x = torch.cat((x3d, x7u), 1)
        x = self.conv71(x)
        x = self.relu(x)
        x = self.conv72(x)
        x = self.relu(x)
        x = self.bn7(x)
        #print('x.shape at 7', x.shape)

        x8u = self.uconv8(x)
        x = torch.cat((x2d, x8u), 1)
        x = self.conv81(x)
        x = self.relu(x)
        x = self.conv82(x)
        x = self.relu(x)
        x = self.bn8(x)

        #print
        x9u = self.uconv9(x)
        x = torch.cat((x1d, x9u), 1)
        x = self.conv91(x)
        x = self.relu(x)
        x = self.conv92(x)
        x = self.relu(x)
        x = self.bn9(x)

        x = self.conv93(x)

        #x = self.sigmoid(x)

        return x

    def write_weights_to_hdf5(self, file_name):
        f = h5py.File(file_name, 'w')
        g1 = f.create_group("/model_weights")
        g = g1.create_group("sequential_1")

        g.create_dataset("conv1d_1_1.weight", data=np.array(self.conv11.weight.data))
        g.create_dataset("conv1d_1_1.bias", data=np.array(self.conv11.bias.data))
        g.create_dataset("conv1d_1_2.weight", data=np.array(self.conv12.weight.data))
        g.create_dataset("conv1d_1_2.bias", data=np.array(self.conv12.bias.data))
        g.create_dataset("bn_1.weight", data=np.array(self.bn1.weight.data)) # gamma
        g.create_dataset("bn_1.bias", data=np.array(self.bn1.bias.data))  # beta
        g.create_dataset("bn_1.running_mean", data=np.array(self.bn1.running_mean.data))
        g.create_dataset("bn_1.running_var", data=np.array(self.bn1.running_var.data))

        g.create_dataset("conv1d_2_1.weight", data=np.array(self.conv21.weight.data))
        g.create_dataset("conv1d_2_1.bias", data=np.array(self.conv21.bias.data))
        g.create_dataset("conv1d_2_2.weight", data=np.array(self.conv22.weight.data))
        g.create_dataset("conv1d_2_2.bias", data=np.array(self.conv22.bias.data))
        g.create_dataset("bn_2.weight", data=np.array(self.bn2.weight.data)) # gamma
        g.create_dataset("bn_2.bias", data=np.array(self.bn2.bias.data))  # beta
        g.create_dataset("bn_2.running_mean", data=np.array(self.bn2.running_mean.data))
        g.create_dataset("bn_2.running_var", data=np.array(self.bn2.running_var.data))

        g.create_dataset("conv1d_3_1.weight", data=np.array(self.conv31.weight.data))
        g.create_dataset("conv1d_3_1.bias", data=np.array(self.conv31.bias.data))
        g.create_dataset("conv1d_3_2.weight", data=np.array(self.conv32.weight.data))
        g.create_dataset("conv1d_3_2.bias", data=np.array(self.conv32.bias.data))
        g.create_dataset("bn_3.weight", data=np.array(self.bn3.weight.data)) # gamma
        g.create_dataset("bn_3.bias", data=np.array(self.bn3.bias.data))  # beta
        g.create_dataset("bn_3.running_mean", data=np.array(self.bn3.running_mean.data))
        g.create_dataset("bn_3.running_var", data=np.array(self.bn3.running_var.data))

        g.create_dataset("conv1d_4_1.weight", data=np.array(self.conv41.weight.data))
        g.create_dataset("conv1d_4_1.bias", data=np.array(self.conv41.bias.data))
        g.create_dataset("conv1d_4_2.weight", data=np.array(self.conv42.weight.data))
        g.create_dataset("conv1d_4_2.bias", data=np.array(self.conv42.bias.data))
        g.create_dataset("bn_4.weight", data=np.array(self.bn4.weight.data)) # gamma
        g.create_dataset("bn_4.bias", data=np.array(self.bn4.bias.data))  # beta
        g.create_dataset("bn_4.running_mean", data=np.array(self.bn4.running_mean.data))
        g.create_dataset("bn_4.running_var", data=np.array(self.bn4.running_var.data))

        g.create_dataset("conv1d_5_1.weight", data=np.array(self.conv51.weight.data))
        g.create_dataset("conv1d_5_1.bias", data=np.array(self.conv51.bias.data))
        g.create_dataset("conv1d_5_2.weight", data=np.array(self.conv52.weight.data))
        g.create_dataset("conv1d_5_2.bias", data=np.array(self.conv52.bias.data))
        g.create_dataset("bn_5.weight", data=np.array(self.bn5.weight.data)) # gamma
        g.create_dataset("bn_5.bias", data=np.array(self.bn5.bias.data))  # beta
        g.create_dataset("bn_5.running_mean", data=np.array(self.bn5.running_mean.data))
        g.create_dataset("bn_5.running_var", data=np.array(self.bn5.running_var.data))

        g.create_dataset("convTranspose1d_6_1.weight", data=np.array(self.uconv6.weight.data))
        g.create_dataset("convTranspose1d_6_1.bias", data=np.array(self.uconv6.bias.data))
        g.create_dataset("conv1d_6_1.weight", data=np.array(self.conv61.weight.data))
        g.create_dataset("conv1d_6_1.bias", data=np.array(self.conv61.bias.data))
        g.create_dataset("conv1d_6_2.weight", data=np.array(self.conv62.weight.data))
        g.create_dataset("conv1d_6_2.bias", data=np.array(self.conv62.bias.data))
        g.create_dataset("bn_6.weight", data=np.array(self.bn6.weight.data)) # gamma
        g.create_dataset("bn_6.bias", data=np.array(self.bn6.bias.data))  # beta
        g.create_dataset("bn_6.running_mean", data=np.array(self.bn6.running_mean.data))
        g.create_dataset("bn_6.running_var", data=np.array(self.bn6.running_var.data))

        g.create_dataset("convTranspose1d_7_1.weight", data=np.array(self.uconv7.weight.data))
        g.create_dataset("convTranspose1d_7_1.bias", data=np.array(self.uconv7.bias.data))
        g.create_dataset("conv1d_7_1.weight", data=np.array(self.conv71.weight.data))
        g.create_dataset("conv1d_7_1.bias", data=np.array(self.conv71.bias.data))
        g.create_dataset("conv1d_7_2.weight", data=np.array(self.conv72.weight.data))
        g.create_dataset("conv1d_7_2.bias", data=np.array(self.conv72.bias.data))
        g.create_dataset("bn_7.weight", data=np.array(self.bn7.weight.data)) # gamma
        g.create_dataset("bn_7.bias", data=np.array(self.bn7.bias.data))  # beta
        g.create_dataset("bn_7.running_mean", data=np.array(self.bn7.running_mean.data))
        g.create_dataset("bn_7.running_var", data=np.array(self.bn7.running_var.data))

        g.create_dataset("convTranspose1d_8_1.weight", data=np.array(self.uconv8.weight.data))
        g.create_dataset("convTranspose1d_8_1.bias", data=np.array(self.uconv8.bias.data))
        g.create_dataset("conv1d_8_1.weight", data=np.array(self.conv81.weight.data))
        g.create_dataset("conv1d_8_1.bias", data=np.array(self.conv81.bias.data))
        g.create_dataset("conv1d_8_2.weight", data=np.array(self.conv82.weight.data))
        g.create_dataset("conv1d_8_2.bias", data=np.array(self.conv82.bias.data))
        g.create_dataset("bn_8.weight", data=np.array(self.bn8.weight.data)) # gamma
        g.create_dataset("bn_8.bias", data=np.array(self.bn8.bias.data))  # beta
        g.create_dataset("bn_8.running_mean", data=np.array(self.bn8.running_mean.data))
        g.create_dataset("bn_8.running_var", data=np.array(self.bn8.running_var.data))

        g.create_dataset("convTranspose1d_9_1.weight", data=np.array(self.uconv9.weight.data))
        g.create_dataset("convTranspose1d_9_1.bias", data=np.array(self.uconv9.bias.data))
        g.create_dataset("conv1d_9_1.weight", data=np.array(self.conv91.weight.data))
        g.create_dataset("conv1d_9_1.bias", data=np.array(self.conv91.bias.data))
        g.create_dataset("conv1d_9_2.weight", data=np.array(self.conv92.weight.data))
        g.create_dataset("conv1d_9_2.bias", data=np.array(self.conv92.bias.data))
        g.create_dataset("conv1d_9_3.weight", data=np.array(self.conv93.weight.data))
        g.create_dataset("conv1d_9_3.bias", data=np.array(self.conv93.bias.data))
        g.create_dataset("bn_9.weight", data=np.array(self.bn9.weight.data)) # gamma
        g.create_dataset("bn_9.bias", data=np.array(self.bn9.bias.data))  # beta
        g.create_dataset("bn_9.running_mean", data=np.array(self.bn9.running_mean.data))
        g.create_dataset("bn_9.running_var", data=np.array(self.bn9.running_var.data))
        #print(self.conv93.bias.data)

        f.close()