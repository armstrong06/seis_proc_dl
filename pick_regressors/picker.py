import torch
from torch.nn import MaxPool1d, Conv1d, Linear
import h5py
import numpy as np
import os
from torch.utils.data.sampler import SubsetRandomSampler
import sys
sys.path.insert(0, "/home/armstrong/Research/git_repos/seis-proc-dl")
from model.base_model import BaseModel
from utils.model_helpers import NumpyDataset
from process_picker_data import randomize_start_times_and_normalize
from picker_trainer import PickerTrainer

class Picker(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device(self.config.train.torch_device)
        self.model = self.build(self.config.model.num_channels, self.config.model.max_dt_nn)

        # Training Config Params
        self.learning_rate = self.config.train.learning_rate
        self.batch_size = self.config.train.batch_size
        self.train_file = self.config.train.train_hdf5_file
        self.validation_file = self.config.train.validation_hdf5_file
        self.model_out_dir = self.config.train.model_out_directory
        self.train_epochs = self.config.train.epochs

        # Model Config Params
        self.phase_type = self.config.model.phase_type
        self.max_dt = self.config.model.max_dt
        self.freeze_convolutional_layers = self.config.model.freeze_convolutional_layers
        self.random_seed = self.config.model.random_seed
        np.random.seed(self.random_seed)

        # Data Config Params
        self.time_series_len = self.config.data.time_series_len
        self.dt = self.config.data.dt
        self.n_duplicates = self.config.data.n_duplicates

        self.model_path = self.model_out_dir  # self.make_model_path(self.model_out_dir)
        self.evaluator = None

        self.evaluation_epoch = -1  # Epoch that is being evaluate - -1 if model not trained/loaded\
        self.results_out_dir = None  # f"{self.model_out_dir}/results"


    def set_results_out_dir(self, test_type):
        outdir = f"{self.model_out_dir}/{test_type}_results"
        if (os.path.exists(outdir)):
            print(f"output directory {outdir} already exists.")
            outdir = f"{self.model_out_dir}/{test_type}_results1"

        self.results_out_dir = outdir

    @staticmethod
    def read_data(data_file):
        # TODO: This didn't work if data_file path is relative to the main script location
        with h5py.File(data_file, "r") as f:
            X = f['X'][:]
        return X

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

    def load_data(self, data_file, batch_size, n_dups, shuffle=True):
        # TODO: This didn't work if data_file path is relative to the main script location
        X = self.read_data(data_file)
        print("Randomizing start times")
        X, Y = randomize_start_times_and_normalize(X, time_series_len=self.time_series_len,
                                                               max_dt=self.max_dt, dt=self.dt, n_duplicate=n_dups,
                                                                radom_seed=self.random_seed)

        dataset = NumpyDataset(X, Y)

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle
        )

        return loader

    def build(self, num_channles, max_dt_nn):
        return CNNNet(num_channels=num_channles, min_lag = -max_dt_nn, max_lag = +max_dt_nn).to(self.device)

    def train(self):
        train_loader = self.load_data(self.train_file, self.batch_size, self.n_duplicates)
        validation_loader = None
        if self.validation_file is not None:
            validation_loader = self.load_data(self.validation_file, 512, 1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if (self.freeze_convolutional_layers):
            print("Freezing convolutional layers...")
            self.model.freeze_convolutional_layers()

        trainer = PickerTrainer(self.model, optimizer, model_path=self.model_path, device=self.device)
        trainer.train(train_loader, validation_loader, self.train_epochs)
        self.evaluation_epoch = self.train_epochs

    def evaluate(self):
        pass

class CNNNet(torch.nn.Module):
    def __init__(self, num_channels=3, min_lag=-0.75, max_lag=0.75):
        super(CNNNet, self).__init__()
        self.relu = torch.nn.ReLU()
        self.min_lag = min_lag
        self.max_lag = max_lag
        self.Hardtanh = torch.nn.Hardtanh(min_val=self.min_lag, max_val=self.max_lag)
        filter1 = 21
        filter2 = 15
        filter3 = 11

        linear_shape = 9600
        if num_channels == 1:
            linear_shape = 6400

        self.maxpool = MaxPool1d(kernel_size=2, stride=2)
        self.conv1 = Conv1d(num_channels, 32,
                            kernel_size=filter1, padding=filter1 // 2)
        self.bn1 = torch.nn.BatchNorm1d(32, eps=1e-05, momentum=0.1)
        # Output has dimension [300 x 32]

        self.conv2 = Conv1d(32, 64,
                            kernel_size=filter2, padding=filter2 // 2)
        self.bn2 = torch.nn.BatchNorm1d(64, eps=1e-05, momentum=0.1)
        # Output has dimension [150 x 64]

        self.conv3 = Conv1d(64, 128,
                            kernel_size=filter3, padding=filter3 // 2)
        self.bn3 = torch.nn.BatchNorm1d(128, eps=1e-05, momentum=0.1)
        # Output has dimension [75 x 128]

        self.fcn1 = Linear(linear_shape, 512)
        self.bn4 = torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1)

        self.fcn2 = Linear(512, 512)
        self.bn5 = torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1)

        self.fcn3 = Linear(512, 1)

    def forward(self, x):
        # N.B. Consensus seems to be growing that BN goes after nonlinearity
        # That's why this is different than Zach's original paper.
        # First convolutional layer
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        # Second convolutional layer
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpool(x)
        # Third convolutional layer
        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)
        x = self.maxpool(x)
        # Flatten
        x = x.flatten(1)  # torch.nn.flatten(x)
        # First fully connected layer
        x = self.fcn1(x)
        x = self.relu(x)
        x = self.bn4(x)
        # Second fully connected layer
        x = self.fcn2(x)
        x = self.relu(x)
        x = self.bn5(x)
        # Last layer
        x = self.fcn3(x)
        # Force linear layer to be between +/- 0.5
        x = self.Hardtanh(x)
        return x

    def freeze_convolutional_layers(self):
        self.conv1.weight.requires_grad = False
        self.conv1.bias.requires_grad = False
        self.bn1.weight.requires_grad = False
        self.bn1.bias.requires_grad = False
        # Second convolutional layer
        self.conv2.weight.requires_grad = False
        self.conv2.bias.requires_grad = False
        self.bn2.weight.requires_grad = False
        self.bn2.bias.requires_grad = False
        # Third convolutional layer
        self.conv3.weight.requires_grad = False
        self.conv3.bias.requires_grad = False
        self.bn3.weight.requires_grad = False
        self.bn3.bias.requires_grad = False

    def write_weights_to_hdf5(self, file_name):
        f = h5py.File(file_name, 'w')
        g1 = f.create_group("/model_weights")
        g = g1.create_group("sequential_1")

        g.create_dataset("conv1d_1.weight", data=np.array(self.conv1.weight.data.cpu()))
        g.create_dataset("conv1d_1.bias", data=np.array(self.conv1.bias.data.cpu()))
        g.create_dataset("bn_1.weight", data=np.array(self.bn1.weight.data.cpu()))  # gamma
        g.create_dataset("bn_1.bias", data=np.array(self.bn1.bias.data.cpu()))  # beta
        g.create_dataset("bn_1.running_mean", data=np.array(self.bn1.running_mean.data.cpu()))
        g.create_dataset("bn_1.running_var", data=np.array(self.bn1.running_var.data.cpu()))

        g.create_dataset("conv1d_2.weight", data=np.array(self.conv2.weight.data.cpu()))
        g.create_dataset("conv1d_2.bias", data=np.array(self.conv2.bias.data.cpu()))
        g.create_dataset("bn_2.weight", data=np.array(self.bn2.weight.data.cpu()))  # gamma
        g.create_dataset("bn_2.bias", data=np.array(self.bn2.bias.data.cpu()))  # beta
        g.create_dataset("bn_2.running_mean", data=np.array(self.bn2.running_mean.data.cpu()))
        g.create_dataset("bn_2.running_var", data=np.array(self.bn2.running_var.data.cpu()))

        g.create_dataset("conv1d_3.weight", data=np.array(self.conv3.weight.data.cpu()))
        g.create_dataset("conv1d_3.bias", data=np.array(self.conv3.bias.data.cpu()))
        g.create_dataset("bn_3.weight", data=np.array(self.bn3.weight.data.cpu()))  # gamma
        g.create_dataset("bn_3.bias", data=np.array(self.bn3.bias.data.cpu()))  # beta
        g.create_dataset("bn_3.running_mean", data=np.array(self.bn3.running_mean.data.cpu()))
        g.create_dataset("bn_3.running_var", data=np.array(self.bn3.running_var.data.cpu()))

        g.create_dataset("fcn_1.weight", data=np.array(self.fcn1.weight.data.cpu()))
        g.create_dataset("fcn_1.bias", data=np.array(self.fcn1.bias.data.cpu()))
        g.create_dataset("bn_4.weight", data=np.array(self.bn4.weight.data.cpu()))  # gamma
        g.create_dataset("bn_4.bias", data=np.array(self.bn4.bias.data.cpu()))  # beta
        g.create_dataset("bn_4.running_mean", data=np.array(self.bn4.running_mean.data.cpu()))
        g.create_dataset("bn_4.running_var", data=np.array(self.bn4.running_var.data.cpu()))

        g.create_dataset("fcn_2.weight", data=np.array(self.fcn2.weight.data.cpu()))
        g.create_dataset("fcn_2.bias", data=np.array(self.fcn2.bias.data.cpu()))
        g.create_dataset("bn_5.weight", data=np.array(self.bn5.weight.data.cpu()))  # gamma
        g.create_dataset("bn_5.bias", data=np.array(self.bn5.bias.data.cpu()))  # beta
        g.create_dataset("bn_5.running_mean", data=np.array(self.bn5.running_mean.data.cpu()))
        g.create_dataset("bn_5.running_var", data=np.array(self.bn5.running_var.data.cpu()))

        g.create_dataset("fcn_3.weight", data=np.array(self.fcn3.weight.data.cpu()))
        g.create_dataset("fcn_3.bias", data=np.array(self.fcn3.bias.data.cpu()))

        f.close()