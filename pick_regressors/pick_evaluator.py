import torch
from scipy.stats import gaussian_kde
import h5py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random # don't use for anything but setting the seed ...

import sys
sys.path.append("/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis-proc-dl/utils")
from utils.model_helpers import compute_outer_fence_mean_standard_deviation

class PickEvaluator():
    def __init__(self, model, device, batch_size, model_dir, outdir, random_seed=None):
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.outdir = outdir
        self.model_dir = model_dir
        
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            random.seed(random_seed)

    def apply_model(self, X_test, y_test, epochs, test_type, df=None, do_shift=True):
        results_df_name = f"{self.outdir}/{test_type}_results.csv"
        residuals_outname_pref = f"{self.outdir}/{test_type}_residuals.txt"
        predictions_outname_pref = f"{self.outdir}/{test_type}_predictions.txt"
        
        figure_dir = f'{self.outdir}/{test_type}_figures/'
        if (not os.path.exists(figure_dir)):
            os.makedirs(figure_dir)

        if df is not None:
            current_eq_rows = np.arange(0, len(df))[ (df['event_type'] == 'le') & (df['evid'] >= 60000000) ]
            current_blast_rows = np.arange(0, len(df))[df['event_type'] == 'qb']
            historical_eq_rows = np.arange(0, len(df))[(df['event_type'] == 'le') & (df['evid'] < 60000000) ]
        else:
            current_eq_rows = np.arange(X_test.shape[0])
            current_blast_rows = []
            historical_eq_rows = []

        n_rows = len(y_test)
        y_pred = np.zeros(len(y_test))

        all_results = []
        all_residuals = np.zeros((len(epochs), n_rows))
        all_predictions = np.zeros((len(epochs), n_rows))

        n_model = 0
        for epoch in epochs: #imodel in range(1):
            model_to_test = '%s/model_%03d.pt'%(self.model_dir, epoch)
            results = {}
            results["model"] = model_to_test

            print("Loading model: ", model_to_test)
            check_point = torch.load(model_to_test)
            print("Model has training loss/rms:",
                check_point['training_loss'], check_point['training_rms']) 
            
            results["training_loss"] = check_point["training_loss"]
            results["training_rms"] = check_point["training_rms"]

            print("Model has validation loss/rms:",
                check_point['validation_loss'], check_point['validation_rms'])
            
            results["validation_loss"] = check_point["validation_loss"]
            results["validation_rms"] = check_point["validation_rms"]

            print("Model has training/test of mean:", 
                check_point['training_mean_of'], check_point['validation_mean_of'])

            results["training_mean_of"] = check_point["training_mean_of"]
            results["validation_mean_of"] = check_point["validation_mean_of"]

            model_mean = check_point['validation_mean_of']
            self.model.load_state_dict(check_point['model_state_dict'])
            self.model.eval()

            n_total_pred = 0
            for batch in range(0, n_rows, self.batch_size):
                i1 = batch
                i2 = min(i1 + self.batch_size, n_rows)
                X_batch = torch.tensor(X_test[i1:i2,:,:], dtype=torch.float).to(self.device)
                y_est_batch = self.model.forward(X_batch)
                for i in range(len(y_est_batch)):
                    y_pred[n_total_pred]  = y_est_batch[i].detach().cpu().numpy()
                    n_total_pred = n_total_pred + 1

            # Loop on batches
            if do_shift:
                print("Shifting residuals by:", model_mean)
            else:
                model_mean = 0

            y_pred = y_pred + model_mean
            residuals = y_test - y_pred

            # save residuals & predictions 
            all_predictions[n_model, :] = y_pred
            all_residuals[n_model, :] = residuals
            n_model += 1
            
            # fname = model_to_test.replace('.pt', '.resid_snr.jpg')
            # fname = fname.replace('/', '_')
            # figure_name = os.path.join(figure_dir, fname) 
            # print(figure_name)
            # self.scatter_to_density(snr, residuals,
            #                 #xlim = [-1,1], ylim = [-20,40], s = 20,
            #                 xbins = np.linspace(-0.5,0.5,61), #[-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5],
            #                 ybins = np.linspace(-20,50,71), #[-20,-15,-10,-5,0,5,10,15,20,25,30,35,45,50],
            #                 title = "All Residuals vs. SNR",
            #                 filename = figure_name)

            mean_of, std_of = compute_outer_fence_mean_standard_deviation(residuals)
            print("Mean/std of all residuals:", np.mean(residuals), np.std(residuals))
            print("Mean/std of outer fence residuals:", mean_of, std_of)
            results["mean_all_residuals"] = np.mean(residuals)
            results["std_all_residuals"] = np.std(residuals)
            results["mean_outer_fence_residuals"] = mean_of
            results["std_outer_fence_residuals"] = std_of

            if len(historical_eq_rows) > 0 and len(current_blast_rows) > 0:
                mean_of_eqc, std_of_eqc = compute_outer_fence_mean_standard_deviation(residuals[current_eq_rows])
                print("Mean/std of current earthquake residuals:", np.mean(residuals[current_eq_rows]), np.std(residuals[current_eq_rows]))
                print("Mean/std of outer fence current earthquake residuals:", mean_of_eqc, std_of_eqc)

                results["mean_eqc_residuals"] = np.mean(residuals[current_eq_rows])
                results["std_eqc_residuals"] = np.std(residuals[current_eq_rows])
                results["mean_eqc_outer_fence_residuals"] = mean_of_eqc
                results["std_eqc_outer_fence_residuals"] = std_of_eqc

                mean_of_eqa, std_of_eqa = compute_outer_fence_mean_standard_deviation(residuals[historical_eq_rows])
                print("Mean/std of historical earthquake residuals:", np.mean(residuals[historical_eq_rows]), np.std(residuals[historical_eq_rows]))
                print("Mean/std of outer fence historical earthquake residuals:", mean_of_eqa, std_of_eqa)

                results["mean_eqh_residuals"] = np.mean(residuals[historical_eq_rows])
                results["std_eqh_residuals"] = np.std(residuals[historical_eq_rows])
                results["mean_eqh_outer_fence_residuals"] = mean_of_eqa
                results["std_eqh_outer_fence_residuals"] = std_of_eqa

                mean_of_blc, std_of_blc = compute_outer_fence_mean_standard_deviation(residuals[current_blast_rows])
                print("Mean/std of current blast residuals:", np.mean(residuals[current_blast_rows]), np.std(residuals[current_blast_rows]))
                print("Mean/std of outer fence current blast residuals:", mean_of_blc, std_of_blc)

                results["mean_blc_residuals"] = np.mean(residuals[current_blast_rows])
                results["std_blc_residuals"] = np.std(residuals[current_blast_rows])
                results["mean_blc_outer_fence_residuals"] = mean_of_blc
                results["std_blc_outer_fence_residuals"] = std_of_blc
            elif len(historical_eq_rows) > 0:
                mean_of_eqc, std_of_eqc = compute_outer_fence_mean_standard_deviation(residuals[current_eq_rows])
                print("Mean/std of current earthquake residuals:", np.mean(residuals[current_eq_rows]), np.std(residuals[current_eq_rows]))
                print("Mean/std of outer fence current earthquake residuals:", mean_of_eqc, std_of_eqc)

                results["mean_eqc_residuals"] = np.mean(residuals[current_eq_rows])
                results["std_eqc_residuals"] = np.std(residuals[current_eq_rows])
                results["mean_eqc_outer_fence_residuals"] = mean_of_eqc
                results["std_eqc_outer_fence_residuals"] = std_of_eqc

                mean_of_eqa, std_of_eqa = compute_outer_fence_mean_standard_deviation(residuals[historical_eq_rows])
                print("Mean/std of historical earthquake residuals:", np.mean(residuals[historical_eq_rows]), np.std(residuals[historical_eq_rows]))
                print("Mean/std of outer fence historical earthquake residuals:", mean_of_eqa, std_of_eqa)

                results["mean_eqh_residuals"] = np.mean(residuals[historical_eq_rows])
                results["std_eqh_residuals"] = np.std(residuals[historical_eq_rows])
                results["mean_eqh_outer_fence_residuals"] = mean_of_eqa
                results["std_eqh_outer_fence_residuals"] = std_of_eqa
            elif len(current_blast_rows) > 0:
                mean_of_eqc, std_of_eqc = compute_outer_fence_mean_standard_deviation(residuals[current_eq_rows])
                print("Mean/std of current earthquake residuals:", np.mean(residuals[current_eq_rows]), np.std(residuals[current_eq_rows]))
                print("Mean/std of outer fence current earthquake residuals:", mean_of_eqc, std_of_eqc)

                results["mean_eqc_residuals"] = np.mean(residuals[current_eq_rows])
                results["std_eqc_residuals"] = np.std(residuals[current_eq_rows])
                results["mean_eqc_outer_fence_residuals"] = mean_of_eqc
                results["std_eqc_outer_fence_residuals"] = std_of_eqc

                mean_of_blc, std_of_blc = compute_outer_fence_mean_standard_deviation(residuals[current_blast_rows])
                print("Mean/std of current blast residuals:", np.mean(residuals[current_blast_rows]), np.std(residuals[current_blast_rows]))
                print("Mean/std of outer fence current blast residuals:", mean_of_blc, std_of_blc)

                results["mean_blc_residuals"] = np.mean(residuals[current_blast_rows])
                results["std_blc_residuals"] = np.std(residuals[current_blast_rows])
                results["mean_blc_outer_fence_residuals"] = mean_of_blc
                results["std_blc_outer_fence_residuals"] = std_of_blc

            all_results.append(results)

            if df is not None:
                zero_weight = abs(df['pick_quality'] - 1.00) < 1.e-4
                one_weight  = abs(df['pick_quality'] - 0.75) < 1.e-4
                two_weight  = abs(df['pick_quality'] - 0.50) < 1.e-4
                zero_close_weight = (abs(df['pick_quality'] - 0.75) < 1.e-4) & (df['source_receiver_distance'] <= 10) & (df['event_type'] == 'le')
                zero_far_weight = (abs(df['pick_quality'] - 0.75) < 1.e-4) & (df['source_receiver_distance'] > 10) & (df['event_type'] == 'le')

            fname = model_to_test.split("/")[-1].replace('.pt', '_resid_quality.jpg')
            #fname = fname.replace('/', '_')
            figure_name = os.path.join(figure_dir, fname) 

            plt.figure(figsize=(8,8))
            plt.hist(residuals, range=(-0.3,0.3), bins=61, align='mid', edgecolor='black', color='black', alpha = 0.85, label='All Residuals')
            #plt.hist(residuals[zero_weight], range=(-0.3,0.3), bins=61, align='mid', edgecolor='black', color='orange', alpha = 0.85, label='Zero Weight')
            #plt.hist(residuals[zero_far_weight], range=(-0.3,0.3), bins=61, align='mid', edgecolor='black', color='blue', alpha = 0.85, label='Zero Weight >= 10km')
            #plt.hist(residuals[zero_close_weight], range=(-0.3,0.3), bins=61, align='mid', edgecolor='black', color='red', alpha = 0.85, label='Zero Weight < 10km')
            if df is not None:
                plt.hist(residuals[zero_weight], range=(-0.5,0.5), bins=101, align='mid', edgecolor='black', color='blue', alpha = 1, label='Zero Weight')
                plt.hist(residuals[one_weight], range=(-0.5,0.5), bins=101, align='mid', edgecolor='black', color='red', alpha = 1, label='One Weight')
                plt.hist(residuals[two_weight], range=(-0.5,0.5), bins=101, align='mid', edgecolor='black', color='yellow', alpha = 1, label='Two Weight')
            xticks = [-0.30, -0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.20, 0.30]
            plt.xlim([min(xticks), max(xticks)])
            plt.xticks(xticks)
            plt.title("Pick Residual vs. Jiggle Quality")#title) #"CNN P Pick Residuals January 2018 - January 2021")
            plt.legend()
            plt.xlabel("Residual (s)")
            plt.ylabel("Counts")
            plt.grid(True)
            #plt.show()
            if (figure_name is None):
                plt.show()
            else:
                print("Saving...", figure_name)
                plt.savefig(figure_name)
                plt.close()
        # Loop on models

        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_df_name, index=False)

        np.savetxt(residuals_outname_pref, all_residuals)
        np.savetxt(predictions_outname_pref, all_predictions)

    @staticmethod
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