#!/usr/bin/env python3
import numpy as np
import warnings
import os
import time
import h5py
import torch
import torch.utils.data
import sklearn as sk
from sklearn.metrics import confusion_matrix

warnings.simplefilter("ignore")

def get_n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def accuracy_precision_recall_up(y_true, y_pred):
    n_tp = 0
    n_tn = 0
    n_fp = 0
    n_fn = 0
    for i in range(len(y_pred)):
        if (y_true[i] < 2):
            if (y_true[i] == 0 and y_pred[i] == 0):
                n_tp = n_tp + 1
            elif (y_true[i] == 0 and y_pred[i] == 1):
                n_fn = n_fn + 1
            elif (y_true[i] == 1 and y_pred[i] == 0):
                n_fp = n_fp + 1
            else:
                n_tn = n_tn + 1
    accuracy = (n_tp + n_tn)/(n_tp + n_tn + n_fp + n_fn)
    precision = n_tp/(n_tp + n_fp)
    recall = n_tp/(n_tp + n_fn) 
    return accuracy, precision, recall

def accuracy_precision_recall_down(y_true, y_pred):
    n_tp = 0
    n_tn = 0
    n_fp = 0
    n_fn = 0
    for i in range(len(y_pred)):
        if (y_true[i] < 2):
            if (y_true[i] == 1 and y_pred[i] == 1):
                n_tp = n_tp + 1
            elif (y_true[i] == 1 and y_pred[i] == 0):
                n_fn = n_fn + 1
            elif (y_true[i] == 0 and y_pred[i] == 1):
                n_fp = n_fp + 1
            else:
                n_tn = n_tn + 1
    accuracy = (n_tp + n_tn)/(n_tp + n_tn + n_fp + n_fn)
    precision = n_tp/(n_tp + n_fp)
    recall = n_tp/(n_tp + n_fn)
    return accuracy, precision, recall
   
def target2target3c(y):
    unique_first_motions = np.unique(y)
    assert min(unique_first_motions) >-1, 'min should be 0'
    assert max(unique_first_motions) < 3, 'max should be 2'
    assert len(unique_first_motions) <= 3, 'should only be at most 3 groups'
    y3c = np.zeros([len(y), 3], dtype='int')
    for i in range(len(y)):
        y3c[i, y[i]] = 1
    return y3c

def randomize_start_times_and_normalize(X_in, time_series_len = 400,
                                        max_dt=0.5, dt=0.01):
    max_shift = int(max_dt/dt)
    n_obs = X_in.shape[0] 
    n_samples = X_in.shape[1]
    n_distribution = n_samples - time_series_len
    random_lag = np.random.random_integers(-max_shift, +max_shift, size=n_obs)
    X_out = np.zeros([len(random_lag), time_series_len])
    ibeg = int(n_samples/2) - int(time_series_len/2) # e.g., 100
    print("Beginning sample to which random lags are added:", ibeg)
    print("Min/max lag:", min(random_lag), max(random_lag))
    for iobs in range(n_obs):
        i1 = ibeg + random_lag[iobs]
        i2 = i1 + time_series_len
        X_out[iobs,:] = X_in[iobs, i1:i2]    
        # Remember to normalize
        xnorm = np.max(np.abs(X_out[iobs,:]))
        X_out[iobs,:] = X_out[iobs,:]/xnorm
    return X_out

class NumpyDataset(torch.utils.data.Dataset):

    def __init__(self, data, target, transform=None):
        n_obs = data.shape[0]
        n_samples = data.shape[1]
        self.data = torch.from_numpy(data.reshape([n_obs, 1, n_samples])).float()
        y3c = target2target3c(target)
        self.target = torch.from_numpy(y3c).float()

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)

class FMNet(torch.nn.Module):

    def __init__(self, num_channels=1, num_classes=3):
        super(FMNet, self).__init__()
        from torch.nn import MaxPool1d, Conv1d, Linear
        from torch.nn.functional import softmax
        self.relu = torch.nn.ReLU()
        self.softmax = softmax
        #k = 3 #7
        filter1 = 21
        filter2 = 15
        filter3 = 11

        self.maxpool = MaxPool1d(kernel_size=2, stride=2)
        self.conv1 = Conv1d(num_channels, 32,
                            kernel_size=filter1, padding=filter1//2)
        self.bn1 = torch.nn.BatchNorm1d(32, eps=1e-05, momentum=0.1)
        # Output has dimension [200 x 32]

        
        self.conv2 = Conv1d(32, 64,
                            kernel_size=filter2, padding=filter2//2)
        self.bn2 = torch.nn.BatchNorm1d(64, eps=1e-05, momentum=0.1)
        # Output has dimension [100 x 64] 

        self.conv3 = Conv1d(64, 128,
                            kernel_size=filter3, padding=filter3//2)
        self.bn3 = torch.nn.BatchNorm1d(128, eps=1e-05, momentum=0.1)
        # Output has dimension [50 x 128]

        self.fcn1 = Linear(6400, 512)
        self.bn4 = torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1)
  
        self.fcn2 = Linear(512, 512)
        self.bn5 = torch.nn.BatchNorm1d(512, eps=1e-05, momentum=0.1)

        self.fcn3 = Linear(512, num_classes)

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
        x = x.flatten(1) #torch.nn.flatten(x)
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
        
        x = self.softmax(x)  
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
        g.create_dataset("bn_1.weight", data=np.array(self.bn1.weight.data.cpu())) # gamma
        g.create_dataset("bn_1.bias", data=np.array(self.bn1.bias.data.cpu()))  # beta
        g.create_dataset("bn_1.running_mean", data=np.array(self.bn1.running_mean.data.cpu()))
        g.create_dataset("bn_1.running_var", data=np.array(self.bn1.running_var.data.cpu()))

        g.create_dataset("conv1d_2.weight", data=np.array(self.conv2.weight.data.cpu()))
        g.create_dataset("conv1d_2.bias", data=np.array(self.conv2.bias.data.cpu()))
        g.create_dataset("bn_2.weight", data=np.array(self.bn2.weight.data.cpu())) # gamma
        g.create_dataset("bn_2.bias", data=np.array(self.bn2.bias.data.cpu()))  # beta
        g.create_dataset("bn_2.running_mean", data=np.array(self.bn2.running_mean.data.cpu()))
        g.create_dataset("bn_2.running_var", data=np.array(self.bn2.running_var.data.cpu()))

        g.create_dataset("conv1d_3.weight", data=np.array(self.conv3.weight.data.cpu()))
        g.create_dataset("conv1d_3.bias", data=np.array(self.conv3.bias.data.cpu()))
        g.create_dataset("bn_3.weight", data=np.array(self.bn3.weight.data.cpu())) # gamma
        g.create_dataset("bn_3.bias", data=np.array(self.bn3.bias.data.cpu()))  # beta
        g.create_dataset("bn_3.running_mean", data=np.array(self.bn3.running_mean.data.cpu()))
        g.create_dataset("bn_3.running_var", data=np.array(self.bn3.running_var.data.cpu()))

        g.create_dataset("fcn_1.weight", data=np.array(self.fcn1.weight.data.cpu()))
        g.create_dataset("fcn_1.bias", data=np.array(self.fcn1.bias.data.cpu()))
        g.create_dataset("bn_4.weight", data=np.array(self.bn4.weight.data.cpu())) # gamma
        g.create_dataset("bn_4.bias", data=np.array(self.bn4.bias.data.cpu()))  # beta
        g.create_dataset("bn_4.running_mean", data=np.array(self.bn4.running_mean.data.cpu()))
        g.create_dataset("bn_4.running_var", data=np.array(self.bn4.running_var.data.cpu()))

        g.create_dataset("fcn_2.weight", data=np.array(self.fcn2.weight.data.cpu()))
        g.create_dataset("fcn_2.bias", data=np.array(self.fcn2.bias.data.cpu()))
        g.create_dataset("bn_5.weight", data=np.array(self.bn5.weight.data.cpu())) # gamma
        g.create_dataset("bn_5.bias", data=np.array(self.bn5.bias.data.cpu()))  # beta
        g.create_dataset("bn_5.running_mean", data=np.array(self.bn5.running_mean.data.cpu()))
        g.create_dataset("bn_5.running_var", data=np.array(self.bn5.running_var.data.cpu()))

        g.create_dataset("fcn_3.weight", data=np.array(self.fcn3.weight.data.cpu()))
        g.create_dataset("fcn_3.bias", data=np.array(self.fcn3.bias.data.cpu()))

        f.close()

class Model():
    def __init__(self, network, optimizer, model_path, device):
        self.network = network
        self.optimizer = optimizer
        self.model_path = model_path
        self.device = device

    def train(self, train_loader, val_loader, n_epochs):
        from torch.autograd import Variable

        self.network.train()
        print("Number of trainable parameters:", get_n_params(self.network))
        loss = torch.nn.BCEWithLogitsLoss()
        n_batches = len(train_loader)
        training_start_time = time.time()

        print_every = n_batches//10
        if (not os.path.exists(self.model_path)):
            os.makedirs(self.model_path)
       

        for epoch in range(n_epochs):
            print("Beginning epoch {}...".format(epoch+1))
            running_accuracy = 0
            running_loss = 0
            running_sample_count = 0
            total_training_loss = 0
            total_validation_loss = 0
            start_time = time.time()

            n_total_pred = 0
            y_true_all = np.zeros(len(train_loader.dataset), dtype='int')
            y_pred_all = np.zeros(len(train_loader.dataset), dtype='int')
            for i, data in enumerate(train_loader, 0):
                # Get inputs/outputs and wrap in variable object
                inputs, y_true = data
                inputs, y_true = Variable(
                    inputs.to(self.device)), Variable(
                    y_true.to(self.device))

                # Set gradients for all parameters to zero
                #print(inputs.shape)
                if (inputs.shape[0] < 2):
                    print("Skipping edge case")
                    continue
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.network(inputs)

                # Backward pass
                loss_value = loss(outputs, y_true)
                loss_value.backward()

                # Update parameters
                self.optimizer.step()

                # Print statistics
                with torch.no_grad():
                    running_loss += loss_value.item()
                    total_training_loss += loss_value.item()

                    # Calculate categorical accuracy
                    y_pred_idx = outputs.argmax(1)
                    y_true_idx = y_true.argmax(1)

                    running_accuracy += (y_pred_idx == y_true_idx).sum().item()
                    running_sample_count += torch.numel(y_true_idx)

                    for i_local_pred in range(len(y_true_idx)):
                        y_true_all[n_total_pred] = y_true_idx[i_local_pred]
                        y_pred_all[n_total_pred] = y_pred_idx[i_local_pred]
                        n_total_pred = n_total_pred + 1

                # Print every n'th batch of an epoch
                if ( (i + 1)%(print_every + 1) == 0):
                    print("Epoch {}, {:d}% \t train_loss: {:.4e} "
                          "train_accuracy: {:4.2f} took: {:.2f}s".format(
                          epoch + 1, int(100*(i + 1)/n_batches),
                          running_loss/print_every,
                          100*running_accuracy/running_sample_count,
                          time.time() - start_time) )
                    running_loss = 0
                    start_time = time.time()
                # end print epoch
            # Loop on no gradient
            # Resize 
            y_true_all = y_true_all[0:n_total_pred]
            y_pred_all = y_pred_all[0:n_total_pred]
            # Compute some metrics
            cm_train = confusion_matrix(y_true_all, y_pred_all)
            print('Train confusion matrix:\n', cm_train)
            accuracy_up, precision_up, recall_up \
                = accuracy_precision_recall_up(y_true_all, y_pred_all)
            accuracy_down, precision_down, recall_down \
                = accuracy_precision_recall_down(y_true_all, y_pred_all)
            print("Train Up class: (Accuracy, Precision, Recall):",
                  accuracy_up, precision_up, recall_up)
            print("Train Down class: (Accuracy, Precision, Recall):",
                  accuracy_down, precision_down, recall_down)

            # Validation
            running_sample_count = 0
            running_val_accuracy = 0
            n_total_pred = 0
            y_true_all = np.zeros(len(val_loader.dataset), dtype='int')
            y_pred_all = np.zeros(len(val_loader.dataset), dtype='int')
            with torch.no_grad():
                for inputs, y_true in val_loader:

                    # Wrap tensors in Variables
                    inputs, y_true = Variable(
                        inputs.to(device)), Variable(
                        y_true.to(device))

                    # Forward pass only
                    val_outputs = self.network(inputs)
                    val_loss = loss(val_outputs, y_true)
                    total_validation_loss += val_loss.item()

                    # Calculate categorical accuracy
                    y_pred = torch.zeros(val_outputs.data.size()).to(device)
                    y_pred_idx = val_outputs.argmax(1)
                    y_true_idx = y_true.argmax(1)

                    running_val_accuracy += (y_pred_idx == y_true_idx).sum().item()
                    running_sample_count += torch.numel(y_true_idx)
                    for i_local_pred in range(len(y_true_idx)):
                        y_true_all[n_total_pred] = y_true_idx[i_local_pred]
                        y_pred_all[n_total_pred] = y_pred_idx[i_local_pred]
                        n_total_pred = n_total_pred + 1
            # Loop on data in training
            total_validation_loss /= len(val_loader)
            total_validation_accuracy = running_val_accuracy/running_sample_count
            print("Validation loss = {:.4e}   acc = {:4.2f}%".format(
                    total_validation_loss,
                    100*total_validation_accuracy))
            accuracy_up, precision_up, recall_up \
                = accuracy_precision_recall_up(y_true_all, y_pred_all)
            accuracy_down, precision_down, recall_down \
                = accuracy_precision_recall_down(y_true_all, y_pred_all)
            cm_validate = confusion_matrix(y_true_all, y_pred_all)
            #                              display_labels=['Up', 'Down', 'Unknown'])
            print(cm_validate) 
            print("Validation Up class: (Accuracy, Precision, Recall):",
                  accuracy_up, precision_up, recall_up)
            print("Validation Down class: (Accuracy, Precision, Recall):",
                  accuracy_down, precision_down, recall_down)
            model_file_name = os.path.join(self.model_path,
                                           'models_%03d.pt'%(epoch+1))
            torch.save({
                       'epoch': epoch+1,
                       'model_state_dict': self.network.state_dict(),
                       'optimizer_state_dict': self.optimizer.state_dict(),
                       'loss': total_validation_loss,
                       }, model_file_name)
            self.network.write_weights_to_hdf5( os.path.join(self.model_path, 
                                                             'models_%03d.h5'%(epoch+1)) )
        # Loop on epochs

if __name__ == "__main__":
    device = torch.device("cuda:0")
    np.random.seed(82323) 
    time_series_len = 400
    n_epochs = 20
    batch_size = 32
    learning_rate = 0.0001
    max_dt = 0.5 # Allow +/- start time perturbation
    dt = 0.01 # Sampling period
    fine_tune = True #True
    freeze_convolutional_layers = False # Freeze convolutional layers during fine tuning?
    model_to_fine_tune = 'models/models_007.pt' #'models/models_011.pt'
    # Map
    polarity_map = {"Up": 0, "Down": 1, "Unknown": 2}

    if (not fine_tune):
        print("Loading California training data...")
        train_file = h5py.File('data/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5', 'r')
    else:
        print("Loading Utah training data...")
        #if (not train_magna):
        np.random.seed(92343)
        n_epochs = 35
        batch_size = 256
        learning_rate = 5.e-5
        train_file = h5py.File('uuss_data/uuss_train.h5', 'r')

    print('Train shape:', train_file['X'].shape)
    X_waves_train = train_file['X'][:]#[0:20000]
    Y_train = train_file['Y'][:]#[0:20000]
    # The targets use [-1,0,1] but Zach's architecture expects [0,1,2] for classes
    """ I've already done this
    if (fine_tune): #and not train_magna):
        for i in range(len(Y_train)):
            if (Y_train[i] == 1):
                Y_train[i] = 0
            elif (Y_train[i] ==-1):
                Y_train[i] = 1
            else:
                Y_train[i] = 2 
    """
    train_file.close()
    print("Randomizing start times...")
    X_train = randomize_start_times_and_normalize(X_waves_train,
                                                  time_series_len = time_series_len,
                                                  max_dt = max_dt, dt = dt)
    print("Creating training dataset...") 
    train_dataset = NumpyDataset(X_train, Y_train)
    indices = list(range(len(Y_train)))
    train_index = np.random.choice(indices, size=len(Y_train), replace=False)
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_index)
    print(train_sampler)
    params_train = {'batch_size': batch_size,
                    'shuffle': True}
    train_loader = torch.utils.data.DataLoader(train_dataset, **params_train)

    if (not fine_tune):
        print("Loading CA validation data...")
        validation_file = h5py.File('data/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5', 'r')
    else:
        print("Loading Utah validation data...")
        #if (not train_magna):
        validation_file = h5py.File('uuss_data/uuss_validation.h5', 'r')
        #else:
        #    validation_file = h5py.File('trainingMagna/utahValidate.h5', 'r')
    print('Validation shape:', validation_file['X'].shape)
    X_waves_validate = validation_file['X'][:]#[0:3200]
    Y_validate = validation_file['Y'][:]#[0:3200]
    # Again need classes need to be consistent with Zach's targets
    """ I did this step earlier
    if (fine_tune): # and not train_magna):
        for i in range(len(Y_validate)):
            if (Y_validate[i] == 1): 
                Y_validate[i] = 0
            elif (Y_validate[i] ==-1):
                Y_validate[i] = 1
            else:
                Y_validate[i] = 2
    """
    validation_file.close()
    print("Randomizing start times...")
    X_validate = randomize_start_times_and_normalize(X_waves_validate,
                                                     time_series_len = time_series_len,
                                                     max_dt = max_dt, dt = dt)
    print("Creating validation dataset...")
    validation_dataset = NumpyDataset(X_validate, Y_validate)
    indices = list(range(len(Y_validate))) 
    validation_index = np.random.choice(indices, size=len(Y_validate), replace=False)
    validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(validation_index)
    params_validate = {'batch_size': 512,
                       'shuffle': True}
    validation_loader = torch.utils.data.DataLoader(validation_dataset, **params_validate)

    #print(X_validate.shape)
    #print(Y_validate[0:10])
    #print(np.sum(Y_validate == 0)) # Up
    #print(np.sum(Y_validate == 1)) # Down
    #print(np.sum(Y_validate == 2)) # Unknown
    #stop
    # Create the network and optimizer
    fmnet = FMNet().to(device)
    print("Number of model parameters:", get_n_params(fmnet))
    optimizer = torch.optim.Adam(fmnet.parameters(), lr=learning_rate)
    model_path = './models'
    if (fine_tune):
        print("Will fine tune:", model_to_fine_tune)
        if (not model_to_fine_tune is None):
            print("Loading model: ", model_to_fine_tune)
            check_point = torch.load(model_to_fine_tune)
            fmnet.load_state_dict(check_point['model_state_dict'])
        #if (not train_magna):
        model_path = './finetuned'
        #else:
        #    model_path = './magna_finetuned_models'
        print("Will write models to:", model_path)
        if (freeze_convolutional_layers):
            print("Freezing convolutional layers...")
            fmnet.freeze_convolutional_layers()
    model = Model(fmnet, optimizer, model_path=model_path, device=device)
    print("Number of trainable parameters:", get_n_params(fmnet))
    print("Starting training...")
    model.train(train_loader, validation_loader, n_epochs)