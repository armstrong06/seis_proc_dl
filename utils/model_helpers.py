import torch
import numpy as np

def get_n_params(model):
    """
    Computes the number of trainable model parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_outer_fence_mean_standard_deviation(residuals):
    """
    Computes the mean and standard deviation using the outer fence method.
    The outerfence is [25'th percentile - 1.5*IQR, 75'th percentile + 1.5*IQR]
    where IQR is the interquartile range.

    Parameters
    ----------
    residuals : The travel time residuals in seconds.

    Results
    -------
    mean : The mean (seconds) of the residuals in the outer fence.
    std : The standard deviation (seconds) of the residuals in the outer fence.
    """
    q1, q3 = np.percentile(residuals, [25,75])
    iqr = q3 - q1
    of1 = q1 - 3.0*iqr
    of3 = q3 + 3.0*iqr
    trimmed_residuals = residuals[(residuals > of1) & (residuals < of3)]
    #print(len(trimmed_residuals), len(residuals), of1, of3)
    mean = np.mean(trimmed_residuals)
    std = np.std(trimmed_residuals)
    return mean, std

def clamp_presigmoid_values(presigmoid, min_presigmoid_value, max_presigmoid_value=None):
    remove_too_neg = torch.clamp(presigmoid, min=min_presigmoid_value, max=max_presigmoid_value)
    # TODO: should I do this here? Would need to update CUDA and tensorflow
    #remove_nan = torch.nan_to_num(remove_too_neg, nan=0.0)
    return remove_too_neg

class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, data, target, transform=None):
        print(data.shape)
        self.data = torch.from_numpy(data.transpose((0, 2, 1))).float()
        print(target.shape)
        if (len(target.shape) == 2):
            self.target = torch.from_numpy(target).float()
        elif (len(target.shape) == 1):
            # For Pickers
            self.target = torch.from_numpy(target.reshape([data.shape[0], 1])).float()
        else:
            self.target = torch.from_numpy(target.transpose((0, 2, 1))).float()

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]

        return x, y

    def __len__(self):
        return len(self.data)
