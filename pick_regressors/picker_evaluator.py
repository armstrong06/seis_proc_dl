import torch
from train_s_picker import CNNNetS, compute_outer_fence_mean_standard_deviation, randomize_start_times_and_normalize
from scipy.stats import gaussian_kde
import h5py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def PickerEvaluator():
    def scatter_to_density(snr, residual,
                           #xlim = [-1,1], ylim = [-20,40], s=20,
                           xbins = [-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5],
                           ybins = [-20,-15,-10,-5,0,5,10,15,20,25,30,35,45,50],
                           title = "All Residuals vs. SNR",
                           filename = None):
        """
        xy = np.vstack([residual, snr])
        z = gaussian_kde(xy)(xy)
        # Sort by density os the density points are plotted last
        idx = z.argsort()
        x, y, z, = x[idx], y[idx], z[idx]
        plt.scatter(x, y, c = z, s = s)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid(grid)
        """
        plt.figure(figsize=(8,8))
        plt.hist2d(residual, snr, bins = (xbins, ybins))
        plt.xlabel("Residual (s)")
        plt.ylabel("SNR")
        plt.title(title)#"Residual vs. SNR")
        plt.xlim(min(xbins), max(xbins))
        plt.ylim(min(ybins), max(ybins))
        if (filename is None):
            plt.show()
        else:
            plt.savefig(filename)