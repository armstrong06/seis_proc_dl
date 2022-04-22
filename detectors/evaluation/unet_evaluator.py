from math import pi
from xml.sax.handler import all_properties
import numpy as np
import torch
import h5py
from sklearn.metrics import confusion_matrix
import os
import pandas as pd

from utils.model_helpers import clamp_presigmoid_values

class UNetEvaluator():

    def __init__(self, batch_size, device, center_window=None, apply_sigmoid=True, minimum_presigmoid_value=None, debug_model_output=False):
        self.min_presigmoid_value = minimum_presigmoid_value
        self.model = None
        self.batch_size = batch_size
        self.apply_sigmoid = apply_sigmoid
        self.device = device
        self.center_window = center_window
        self.debug_model_output = debug_model_output

    def get_n_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def set_model(self, model):
        self.model = model

    @staticmethod
    def save_posterior_probs(posterior_probs, Y_proba, T_est_index, proba_out_dir, epoch):
        if (not os.path.exists(proba_out_dir)):
            os.makedirs(proba_out_dir)
        proba_out_file = f"{proba_out_dir}/posterior_probs.h5"
        probafile = h5py.File(proba_out_file, "w")
        probafile.create_group("ModelOutputs")
        probafile.create_dataset("%s.Y_est"%epoch, data=posterior_probs)
        probafile.create_dataset("%s.Y_max_proba"%epoch, data=Y_proba)
        probafile.create_dataset("%s.T_est_index"%epoch, data=T_est_index)
        probafile.close()

    def apply_model_to_batch(self, X_batch):
        if self.model is None:
            print("No model set!")
            return 

        # Get predictions
        Y_est = self.model.forward(X_batch)
        # Save the presigmoid values
        if self.debug_model_output:
            Y_presigmoid = Y_est.squeeze().to('cpu').detach().numpy()

        if (self.apply_sigmoid):
            if self.min_presigmoid_value is not None:
                Y_est = clamp_presigmoid_values(Y_est, self.min_presigmoid_value)
            Y_est = torch.sigmoid(Y_est)

        Y_est = Y_est.squeeze()

        if self.debug_model_output:
            return Y_est, Y_presigmoid
        return Y_est, None

    def apply_model(self, X_test, pick_method=None):
        """Apply the model to the entire dataset.
        
        Returns:
            model_output_tuple: Returns a tuple with the posterior probabilities for every trace. 
                                If in debug mode, the tuple also includes the presigmoid values.
            estimate_pick_tuple: If pick_method is None, returns None. If pick_method is "single" 
                                returns the most probability and index of the most probable pick. 
                                If pick_method is "multiple" returns the list of possible picks for each
                                trace, their probabilities, and their boxcar widths. 
            """

        n_samples = X_test.shape[1]
        n_examples =  X_test.shape[0]

        # Initialize
        pred_pick_index = np.zeros(n_examples, dtype='i')
        pred_pick_prob = np.zeros(n_examples)
        all_posterior_probs = np.zeros([n_examples, n_samples])

        if pick_method == "multiple":
            all_widths = []
            pred_pick_index = []
            pred_pick_prob = []

        if self.debug_model_output:
            all_presigmoid_values = np.zeros([n_examples, n_samples])

        for i1 in range(0, n_examples, self.batch_size):
            i2 = min(n_examples, i1 + self.batch_size)

            # Gather batch
            X_temp = np.copy(X_test[i1:i2, :, :]) 
            X = torch.from_numpy(X_temp.transpose((0, 2, 1))).float().to(self.device)
            
            # Get predictions
            if self.debug_model_output:
                Y_est, Y_presigmoid = self.apply_model_to_batch(X)
                all_presigmoid_values[i1:i2, :] = Y_presigmoid
            else:
                Y_est, _ = self.apply_model_to_batch(X)
                
            # Pick indices and probabilities
            if pick_method == "single":
                values, indices = self.get_single_picks(Y_est, n_samples)
                for k in range(i1,i2):
                    pred_pick_prob[k] = values[k-i1]
                    pred_pick_index[k] = indices[k-i1]
            elif pick_method == "multiple":
                for k in range(i1, i2):
                    values, indices, widths = self.sliding_window_phase_arrival_estimator(Y_est[k-i1], thresh=0.1)
                    pred_pick_prob.append(values)
                    pred_pick_index.append(indices)
                    all_widths.append(widths)

            all_posterior_probs[i1:i2, :] = Y_est.to('cpu').detach().numpy()

        output1 = (all_posterior_probs)
        if self.debug_model_output:
            output1 = (all_posterior_probs, Y_presigmoid)

        output2 = None
        if pick_method == "single":
            output2 = (pred_pick_index, pred_pick_prob)
        elif pick_method == "multiple":
            output2 = (pred_pick_index, pred_pick_prob, widths)

        return output1, output2 

    def get_single_picks(self, Y_est, n_samples):
        # Pick indices and probabilities
        if (self.center_window is None):
            values, indices = Y_est.max(dim=1)
            values = values.to('cpu').flatten()
            indices = indices.to('cpu').flatten()
        else:
            j1 = int(n_samples/2 - self.center_window)
            j2 = int(n_samples/2 + self.center_window)
            Y_sub = Y_est[:,j1:j2]
            values, indices = Y_sub.max(dim=1) 
            values = values.to('cpu').flatten()
            indices = indices.to('cpu').flatten() + j1

        return values, indices


    def tabulate_metrics(self, true_pick_index, est_pick_proba, est_pick_index, model_epoch,
                        tols = [0.1, 0.25, 0.5, 0.75, 0.9]):
        results = []

        # Only look at signal, not noise
        Y_obs = (true_pick_index >= 0)*1
        n_picks = np.sum(Y_obs)

        # Iterate over different tolerance thresholds for making a pick
        for tol in tols:
            Y_est = (est_pick_proba > tol)*1
            index_resid = np.zeros(len(true_pick_index), dtype='int')

            # Find where there are matching picks (any pick in the window) and get residuals
            j = 0
            for i in range(len(true_pick_index)):
                if (Y_obs[i] == 1 and Y_est[i] == 1):
                    index_resid[j] = true_pick_index[i] - est_pick_index[i] 
                    j = j + 1
            index_resid = np.copy(index_resid[0:j])
            if (len(index_resid) > 0):
                trimmed_mean, trimmed_std = self.compute_outer_fence_mean_standard_deviation(index_resid)
                residual_mean = np.mean(index_resid)
                residual_std = np.std(index_resid)
            else:
                trimmed_mean = 0
                trimmed_std = 0
                residual_mean = 0
                residual_std = 0
            
            # I had to add label into this or it breaks when 100% accuracte. If there are more than 2 classes, will need to edit this. 
            tn, fp, fn, tp = confusion_matrix(Y_obs, Y_est, labels=[0, 1]).ravel()
            
            if (tp + tn + fp + fn) == 0:
                acc = 0
            else:
                acc  = (tn + tp)/(tp + tn + fp + fn)
            
            if (tp + fp) == 0:
                prec = 0
            else:
                prec   = tp/(tp + fp)

            if (tp + fn) == 0:
                recall = 0
            else:
                recall = tp/(tp + fn)

            dic = {"epoch": model_epoch,
                "n_picks": n_picks,
                "n_picked": len(index_resid),
                "tolerance": tol,
                "accuracy": acc,
                "precision": prec,
                "residual_mean": residual_mean,
                "residual_std": residual_std,
                "trimmed_residual_mean": trimmed_mean,
                "trimmed_residual_std": trimmed_std,
                "recall": recall}
            results.append(dic)
        return results

    @staticmethod
    def calculate_residuals(T_test, T_est_index, Y_proba, epoch):
        resids = []
        for i in range(len(T_test)):
            if (T_test[i] < 0):
                break
            resids.append( {'epoch': epoch,
                            'true_lag': T_test[i],
                            'residual': T_test[i] - T_est_index[i],
                            'probability': Y_proba[i],})
                            #'snr': snrs[i] } ) 
        return resids
    
    @staticmethod
    def save_result(result, outfile):
        df = pd.DataFrame(result) 
        df.to_csv(outfile, index=False)

    @staticmethod
    def compute_outer_fence_mean_standard_deviation(residuals):
        
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1 
        of1 = q1 - 1.5*iqr
        of3 = q3 + 1.5*iqr
        trimmed_residuals = residuals[(residuals > of1) & (residuals < of3)]
        #print(len(trimmed_residuals), len(residuals), of1, of3)
        if len(trimmed_residuals) == 0:
            return 0, 0
        xmean = np.mean(trimmed_residuals)
        xstd = np.std(trimmed_residuals) 
        return xmean, xstd 

    @staticmethod
    def sliding_window_phase_arrival_estimator(Y, window_size=100, thresh=0.1, end_thresh_diff=0.05):
        """
        Find potential phase arrivals in the probability time-series output from the UNet given a minimum threshold value.
        Chooses window to look for detection based on when the proba goes above thresh and then below again
        :param Y: Probability time series
        :param window_size: step size for sliding window
        :param thresh: minimum probability threshold to record picks
        :param min_boxcar_width: approx. minimum width of a boxcar detection allowed
        :return: list of the samples in Y with an expected phase arrival (given thresh)
        """
        i1 = 0
        picks = []
        proba_values = []
        widths = []
        end_thresh = thresh-end_thresh_diff
        while i1 < Y.shape[0]:
            i2 = i1 + np.min([(Y.shape[0] - i1), window_size])
            if np.any(Y[i1:i2] >= thresh):
                # find first ind in window above thresh (start looking for max proba)
                start_win = i1 + np.where(Y[i1:i2] >= thresh)[0][0]
                # find end ind where proba has gone below thresh - if start and end inds are too close together, find a new end index
                search_win_size = 100
                possible_win_lengths = np.where(Y[start_win:start_win+search_win_size] < end_thresh)[0]
                while len(possible_win_lengths) < 1:
                    search_win_size += 10
                    possible_win_lengths = np.where(Y[start_win:start_win+search_win_size] < thresh)[0]
                    if start_win + search_win_size > Y.shape[0]:
                        break
                if len(possible_win_lengths) == 0:
                    end_win = Y.shape[0]
                else:
                    end_win = start_win + possible_win_lengths[0]
                proba = np.max(Y[start_win:end_win])
                pick = start_win + np.where(Y[start_win:end_win] == proba)[0][0]
                widths.append(end_win-start_win)
                picks.append(pick)
                proba_values.append(proba)
                i1 = end_win
            else:
                i1 += window_size

        return np.array(proba_values), np.array(picks), np.array(widths)