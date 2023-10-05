#!/usr/bin/env python3
import numpy as np
import warnings
import os
import h5py
import torch
import torch.utils.data
#import apex.amp as amp
#from threadpoolctl import threadpool_limits

warnings.simplefilter("ignore")
#phase_type = "P"

device = torch.device("cuda:0")

class NumpyDataset(torch.utils.data.Dataset):

    def __init__(self, data, target, transform=None):
        print(data.shape)
        self.data = torch.from_numpy(data.transpose((0, 2, 1))).float()
        print(target.shape)
        if (len(target.shape) == 2):
            self.target = torch.from_numpy(target).float()
        else:
            self.target = torch.from_numpy(target.transpose((0, 2, 1))).float()

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        return x, y

    def __len__(self):
        return len(self.data)


class UNet(torch.nn.Module):

    def __init__(self, num_channels=3, num_classes=1, k=3):
        super(UNet, self).__init__()
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

    """
    def predict(self, Z, N, E): 
        self.eval()
        zmax = np.max(np.abs(Z))
        nmax = np.max(np.abs(N))
        emax = np.max(np.abs(E))
        norm = max(max(zmax, nmax), emax)
        l = 16
        while (l < len(Z)): 
            l = l + 16
        X = np.zeros([1, l, 3]) 
        data = torch.from_numpy(X.transpose((0, 2, 1))).float()
        X[0,:,0] = Z[0:l]/norm
        X[0,:,1] = N[0:l]/norm
        X[0,:,2] = E[0:l]/norm
        r = torch.sigmoid(self.network.forward(data))
        r = r.numpy()
        return r
    """

class Model():
    def __init__(self, network, optimizer, model_path):
        self.network = network
        self.optimizer = optimizer
        self.model_path = model_path
        if (not os.path.exists(self.model_path)):
            os.makedirs(self.model_path)

    def train(self, train_loader, val_loader, n_epochs):
        from torch.autograd import Variable
        import time

        self.network.train()
        loss = torch.nn.BCEWithLogitsLoss()
        n_batches = len(train_loader)
        training_start_time = time.time()

        for epoch in range(n_epochs):
            running_loss = 0.0
            running_acc = 0
            running_val_acc = 0
            print_every = n_batches // 10
            start_time = time.time()
            total_train_loss = 0
            total_val_loss = 0
            total_val_acc = 0
            running_sample_count = 0

            for i, data in enumerate(train_loader, 0):
                # Get inputs/outputs and wrap in variable object
                inputs, y_true = data
                inputs, y_true = Variable(
                    inputs.to(device)), Variable(
                    y_true.to(device))

                # Set gradients for all parameters to zero
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.network(inputs)
                #y_true = y_true[:,None,:]

                # Backward pass
                loss_value = loss(outputs, y_true)
                loss_value.backward()
                #loss_ = loss(outputs, y_true)
                #with amp.scale_loss(loss_, optimizer) as loss_value:
                #    loss_value.backward()

                # Update parameters
                self.optimizer.step()

                # Print statistics
                with torch.no_grad():
                    outputs = torch.sigmoid(outputs)
                    running_loss += loss_value.item()
                    total_train_loss += loss_value.item()

                    # Calculate categorical accuracy
                    y_pred = torch.zeros(outputs.data.size()).to(device)
                    y_pred[outputs >= 0.5] = 1

                    running_acc += (y_pred == y_true).sum().item()
                    running_sample_count += torch.numel(y_true)

                # Print every 10th batch of an epoch
                if (i + 1) % (print_every + 1) == 0:
                    print("Epoch {}, {:d}% \t train_loss: {:.4e} "
                        "train_acc: {:4.2f}% took: {:.2f}s".format(
                        epoch + 1, int(100 * (i + 1) / n_batches),
                        running_loss / print_every,
                        100*running_acc / running_sample_count,
                        time.time() - start_time))
                    # Reset running loss and time
                    running_loss = 0.0
                    start_time = time.time()

            total_train_loss /= len(train_loader)
            print("Training loss:", total_train_loss)

            """
            running_sample_count = 0
            y_pred_all, y_true_all = [], []

            with torch.no_grad():
                for inputs, y_true in val_loader:

                    # Wrap tensors in Variables
                    inputs, y_true = Variable(
                        inputs.to(device)), Variable(
                        y_true.to(device))

                    # Forward pass only
                    val_outputs = self.network(inputs)
                    val_outputs = torch.sigmoid(val_outputs)
                    val_loss = loss(val_outputs, y_true)
                    total_val_loss += val_loss.item()

                    # Calculate categorical accuracy
                    y_pred = torch.zeros(val_outputs.data.size()).to(device)
                    y_pred[val_outputs >= 0.5] = 1

                    # Compute the pick times
                    y_pred = y_pred.cpu().numpy()
                    y_true = y_true.cpu().numpy()

                    y_pred_all.append(y_pred.flatten())
                    y_true_all.append(y_true.flatten())

                    #y_pred_all.append(y_pred.cpu().numpy().flatten())
                    #y_true_all.append(y_true.cpu().numpy().flatten())

                    running_val_acc += (y_pred == y_true).sum().item()
                    running_sample_count += torch.numel(y_true)

            y_pred_all = np.concatenate(y_pred_all)
            y_true_all = np.concatenate(y_true_all)

            from sklearn.metrics import classification_report

            total_val_loss /= len(val_loader)
            total_val_acc = running_val_acc / running_sample_count
            print(
                "Validation loss = {:.4e}   acc = {:4.2f}%".format(
                    total_val_loss,
                    100*total_val_acc))

            print(classification_report(y_true_all, y_pred_all))
            """
            print('%s/model_%s_%03d.pt' % (self.model_path, phase_type, epoch))
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': total_train_loss, #total_val_loss,
            }, '%s/model_%s_%03d.pt' % \
                (self.model_path, phase_type, epoch))
            #self.network.write_weights_to_hdf5(os.path.join(self.model_path,
            #                                   'pymodel_%s_%03d'%(phase_type, epoch)))

        print(
            "Training finished, took {:.2f}s".format(
                time.time() -
                training_start_time))


if __name__ == "__main__":
    """
    np.random.seed(3940)
    torch.manual_seed(3940)
    unet = UNet(3, 1)#.to(device)
    unet.write_weights_to_hdf5("weights_file.h5")
    X = np.zeros([1,240,3])
    for i in range(X.shape[1]):
        if (i%2 == 0):
            X[0,i,:] = 1
        else:
            X[0,i,:] =-1

    data = torch.from_numpy(X.transpose((0, 2, 1))).float()
    r = torch.sigmoid(unet.forward(data))
    print(r.shape)
    print(r)
    """
    #with threadpool_limits(limits=2, user_api='blas'):
    phase_type = "P"
    duration = 10
    n_dup = 1
    n_epochs = 35
    if (phase_type == "S"):
        learning_rate = 0.1
    else:
        learning_rate = 0.001 
    batch_size = 256
    model_path = "/home/armstrong/Research/mew_threecomp/%s_models_%s_%s"%(phase_type, batch_size, learning_rate)
    print(model_path)
    print("learning rate", learning_rate)
    import h5py
    with h5py.File("data/train%s.%ds.%ddup.synthetic.multievent.h5"%(phase_type, duration, n_dup)) as f:
         X_train = f['X'][:]
         Y_train = f['Y'][:]
         T_train = f['Pick_index'][:]
    #with h5py.File("validate%s.%ds.%ddup.h5"%(phase_type, duration, n_dup)) as f:
    #   X_train = f['X'][:]
    #   Y_train = f['Y'][:]
    #   T_train = f['Pick_index'][:]


    train_dataset = NumpyDataset(X_train, Y_train)
    #test_dataset = NumpyDataset(X_test, Y_test)
    n_train_samples = len(train_dataset)
    #n_test_samples = len(test_dataset)
    #n_test = int(0.1*n_samples)
    #print(X.shape, Y.shape)
    #print(n_samples, n_test)
    #stop

    train_indices = list(range(n_train_samples))
    #test_indices = list(range(n_test_samples))

    #validation_idx = np.random.choice(indices, size=n_test, replace=False)
    #train_idx = list(set(indices) - set(validation_idx))
    # Randomize rows since I packed the data then the noise
    train_idx = np.random.choice(train_indices, size=n_train_samples, replace=False)
    #validation_idx = np.random.choice(test_indices, size=n_test_samples, replace=False)

    from torch.utils.data.sampler import SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_idx)
    #validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
    )
    val_loader = None
    """
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False,
        sampler=validation_sampler
    )
    """

    unet = UNet(num_channels=3, num_classes=1).to(device)
    #unet = torch.nn.DataParallel(unet)
    optimizer = torch.optim.Adam(unet.parameters(), lr = learning_rate)

    #amp.register_float_function(torch, 'sigmoid')
    #unet, optimizer = amp.initialize(unet, optimizer, opt_level='O1',
    #    loss_scale="dynamic")

    model = Model(unet, optimizer, model_path=model_path)
    model.train(train_loader, val_loader, n_epochs)
