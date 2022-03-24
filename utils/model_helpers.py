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

def clamp_presigmoid_values(presigmoid, min_presigmoid_value):
    remove_too_neg = torch.clamp(presigmoid, min=min_presigmoid_value, max=None)
    # TODO: should I do this here? Would need to update CUDA and tensorflow
    #remove_nan = torch.nan_to_num(remove_too_neg, nan=0.0)
    return remove_too_neg