import torch

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