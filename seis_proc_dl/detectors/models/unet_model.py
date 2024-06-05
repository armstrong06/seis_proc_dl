import torch
import h5py
import numpy as np

class UNetModel(torch.nn.Module):

    def __init__(self, apply_last_sigmoid=False, num_channels=3, num_classes=1, k=3):
        super(UNetModel, self).__init__()
        from torch.nn import MaxPool1d, Conv1d, ConvTranspose1d
        self.relu = torch.nn.ReLU()
        self.lsigmoid = apply_last_sigmoid
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

        self.sigmoid = torch.nn.Sigmoid()

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

        if self.lsigmoid:
            # Logic from Ben:
            # Sigmoid is 1/(1 + exp(-x))
            # Float max: float: 3.40282e+38
            # log(float max): 88.7
            # Used min=-70, max=None in Armstrong et al, 2023
            x = torch.clamp(x, -87, 87)
            x = self.sigmoid(x)
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