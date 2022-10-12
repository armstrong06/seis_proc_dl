from math import pi
from xml.sax.handler import all_properties
import numpy as np
import torch
import h5py
from sklearn.metrics import confusion_matrix
import os
import pandas as pd
import time 
import glob

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
                # get_single_picks uses pytorch tensor
                all_posterior_probs[i1:i2, :] = Y_est.to('cpu').detach().numpy()
            elif pick_method == "multiple":
                Y_est = Y_est.to('cpu').detach().numpy()
                for k in range(i1, i2):
                    values, indices, widths = self.sliding_window_phase_arrival_estimator(Y_est[k-i1], thresh=0.1)
                    pred_pick_prob.append(values)
                    pred_pick_index.append(indices)
                    all_widths.append(widths)
                # sliding_window_phase_arrival_estimator uses numpy array
                all_posterior_probs[i1:i2, :] = Y_est

        output1 = (all_posterior_probs)
        if self.debug_model_output:
            output1 = (all_posterior_probs, Y_presigmoid)

        output2 = None
        if pick_method == "single":
            output2 = (pred_pick_index, pred_pick_prob)
        elif pick_method == "multiple":
            output2 = (pred_pick_index, pred_pick_prob, all_widths)

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
                        tols = [0.1, 0.25, 0.5, 0.75, 0.9], df=None):
        results = []

        if df is not None:
            current_eq_rows = np.arange(0, len(df))[ (df['event_type'] == 'le') & (df['evid'] >= 60000000) ]
            current_blast_rows = np.arange(0, len(df))[df['event_type'] == 'qb']
            historical_eq_rows = np.arange(0, len(df))[(df['event_type'] == 'le') & (df['evid'] < 60000000) ]

        # Make a binary array of the picks (1=singal, 0=noise)
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
            
            def compute_acc_prec_recall(tn, fp, fn, tp):
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
                
                return acc, prec, recall

            def compute_subset_stats(rows):
                Y_obs_subset = Y_obs[rows]
                Y_est_subset =  Y_est[rows]
                tn_sub, fp_sub, fn_sub, tp_sub = confusion_matrix(Y_obs_subset, Y_est_subset, labels=[0, 1]).ravel()
                acc, prec, recall = compute_acc_prec_recall(tn_sub, fp_sub, fn_sub, tp_sub)
                return acc, prec, recall

            # I had to add label into this or it breaks when 100% accuracte. If there are more than 2 classes, will need to edit this. 
            tn, fp, fn, tp = confusion_matrix(Y_obs, Y_est, labels=[0, 1]).ravel()
            acc, prec, recall = compute_acc_prec_recall(tn, fp, fn, tp)

            if df is not None:
                ceq_acc, ceq_prec, ceq_recall = compute_subset_stats(current_eq_rows)
                heq_acc, heq_prec, heq_recall = compute_subset_stats(historical_eq_rows)
                cbl_acc, cbl_prec, cbl_recall = compute_subset_stats(current_blast_rows)
                noise_acc, noise_prec, noise_recall = None, None, None
                if len(true_pick_index) > len(df):
                    noise_acc, noise_prec, noise_recall = compute_subset_stats(np.arange(len(df), len(true_pick_index)))

                dic = {"epoch": model_epoch,
                    "n_picks": n_picks,
                    "n_picked": len(index_resid),
                    "tolerance": tol,
                    "accuracy": acc,
                    "precision": prec,
                    "recall": recall, 
                    "residual_mean": residual_mean,
                    "residual_std": residual_std,
                    "trimmed_residual_mean": trimmed_mean,
                    "trimmed_residual_std": trimmed_std,
                    "tp": tp, 
                    "tn": tn, 
                    "fp": fp, 
                    "fn":fn, 
                    "ceq_accuracy": ceq_acc,
                    "ceq_precision": ceq_prec,
                    "ceq_recall": ceq_recall,
                    "heq_accuracy": heq_acc,
                    "heq_precision": heq_prec,
                    "heq_recall": heq_recall,
                    "cbl_accuracy": cbl_acc,
                    "cbl_precision": cbl_prec,
                    "cbl_recall": cbl_recall,
                    "noise_accuracy": noise_acc,
                    "noise_precision": noise_prec,
                    "noise_recall": noise_recall}
            else:
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
                    "recall": recall, 
                    "tp": tp, 
                    "tn": tn, 
                    "fp": fp, 
                    "fn":fn}

            results.append(dic)
        return results

    def tabulate_metrics_mew(self, T_test, T_test2, Y_proba, T_est_index, epoch, Y_data_est, Y_data_act,
                        tols=[0.1, 0.25, 0.5, 0.75, 0.9], df=None):
        results_p1 = []
        results_p2 = []
        results_comb = []
        results_js = []
        js_arrays = []
        # makes a binary array of the real picks
        Y_obs = (T_test >= 0) * 1
        Y_obs2 = (T_test2 >= 0) * 1
        n_picks1 = np.sum(Y_obs)
        n_picks2 = np.sum(Y_obs2)
        n_noise = len(T_test) - n_picks1

        if df is not None:
            current_eq_rows = np.arange(0, len(df))[ (df['event_type'] == 'le') & (df['evid'] >= 60000000) ] + len(T_test2)
            current_blast_rows = np.arange(0, len(df))[df['event_type'] == 'qb'] + len(T_test2)
            historical_eq_rows = np.arange(0, len(df))[(df['event_type'] == 'le') & (df['evid'] < 60000000) ] + len(T_test2)

            assert np.min([np.min(current_eq_rows), np.min(current_blast_rows), np.min(historical_eq_rows)]) >= 0, "Index is less than 0"

        for tol in tols:
            # Y_est = (Y_proba > tol) * 1

            ## Residual information will not mean much here because only calculate pick residuals for those close together
            ## Can look at how many residuals have nan values though
            index_resid = np.full(len(T_test)+len(T_test2), np.nan)
            jaccard_sims = np.full(len(T_test), np.nan)
            jaccard_sims_proba = np.full(len(T_test), np.nan)
            resids1 = np.full(len(T_test), np.nan)
            resids2 = np.full(len(T_test2), np.nan)
            noise_class = np.zeros(n_noise)
            j = 0
            j1 = 0
            j2 = 0

            est_picks_cnt = 0
            est_picks_cnt_p2 = 0
            for i in range(len(T_test)):               
                Y_est = (Y_proba[i] > tol) *1             

                # Get picks with probability > threshold 
                est_picks = T_est_index[i][np.where(Y_proba[i] > tol)[0]]
                est_picks_cnt += len(est_picks)

                # Keep track of how many picks total picks for MEW waveforms
                if i == len(Y_obs2):
                    est_picks_cnt_p2 = est_picks_cnt
                
                # TODO: check if this should be < or <=
                if i < len(Y_obs2):
                    actual_picks = [T_test[i], T_test2[i]]
                else:
                    actual_picks = [T_test[i]]
                    
                js, resids = self.calculate_pick_similiarity(est_picks, actual_picks)

                # If a pick is missed, the resid should be nan. noise will already have nan value
                if (Y_obs[i] == 1 and np.any(Y_est == 1)):
                    # TODO: I feel like the J indexing may be wrong - want picks to match up with Y_obs
                    if i < len(Y_obs2):
                        index_resid[j:j+2] = resids[:]
                        j = j + 2
                        resids1[i] = resids[0]
                        resids2[i] = resids[1]
                        j1 += 1
                        j2 += 1
                    else:
                        index_resid[j] = resids[0]
                        j = j + 1
                        resids1[i] = resids[0]
                        j1 += 1
                elif (Y_obs[i] == 0 and np.any(Y_est == 1)):
                    # Keep track of where there were picks in a noise window so we can look at the TN rate
                    noise_class[i-n_picks1] = 1

                # TODO: add in other way of measuring JS
                jaccard_sims[i] = js
                jaccard_sims_proba[i] = self.calculate_jaccard_similarity_proba(Y_data_act[i], Y_data_est[i], tol)

            index_resid = np.copy(index_resid[0:j])

            # assign 0 and ones based on where there was a residual calculated or not
            Y_est1 = (~np.isnan(resids1)) * 1
            Y_est2 = (~np.isnan(resids2)) * 1
            # (takes into account picks that are made but not in the right locations?)
            fp_all = est_picks_cnt - sum(Y_est1) - sum(Y_est2)  # number of picks made - number of picks assigned to p1 or p2 
            #fp = est_picks_cnt - n_picks1 - n_picks2 # number of picks made - number of known picks 
            assert fp_all >= 0, "FP is negative"

            # Calcualte the FP for MEW wabeforms. Total number of picks for MEW - P2 picks - P1 picks on MEW waveforms
            fp_p2 = est_picks_cnt_p2 - sum(Y_est2) - sum(~np.isnan(resids1[:len(Y_obs2)])*1)

            # Since FP is being calculated seperately, all noise observations are being assigned noise by default, even if the model made a pick.
            # Change estimated noise classification to 1 in there were any proba over threshold, so we can get an accurate estimate of TN 
            Y_est1[-n_noise:] = noise_class

            def calc_stats(Y_obs, Y_est, n_picks, residuals, fp, df=None):
                residuals = residuals[np.where(~np.isnan(residuals))[0]]

                trimmed_mean, trimmed_std = self.compute_outer_fence_mean_standard_deviation(residuals)

                # I had to add label into this or it breaks when 100% accuracte. If there are more than 2 classes, will need to edit this.
                tn, fp_nothing, fn, tp = confusion_matrix(Y_obs, Y_est, labels=[0, 1]).ravel()
                # TODO: I'm not sure if this is right
                acc = (tn + tp) / (tp + tn + fp + fn)
                prec = tp / (tp + fp)
                recall = tp / (tp + fn)

                def compute_subset_stats(rows):
                    Y_obs_subset = Y_obs[rows]
                    Y_est_subset =  Y_est[rows]
                    tn_sub, fp_sub, fn_sub, tp_sub = confusion_matrix(Y_obs_subset, Y_est_subset, labels=[0, 1]).ravel()
                    acc = (tn_sub + tp_sub) / (tp_sub + tn_sub + fp_sub + fn_sub)
                    prec = tp_sub / (tp_sub + fp_sub)
                    recall = tp_sub / (tp_sub + fn_sub)
                    return acc, prec, recall

                if df is not None:
                    ceq_acc, ceq_prec, ceq_recall = compute_subset_stats(current_eq_rows)
                    heq_acc, heq_prec, heq_recall = compute_subset_stats(historical_eq_rows)
                    cbl_acc, cbl_prec, cbl_recall = compute_subset_stats(current_blast_rows)
                    
                    noise_acc, noise_prec, noise_recall = None, None, None
                    if len(T_test) > len(df):
                        noise_acc, noise_prec, noise_recall = compute_subset_stats(np.arange(len(df), len(T_test)))
                    
                    dic = {"epoch": epoch,
                        "n_picks": n_picks,
                        "n_picked": len(index_resid),
                        "tolerance": tol,
                        "accuracy": acc,
                        "precision": prec,
                        "recall": recall, 
                        "residual_mean": np.mean(residuals),
                        "residual_std": np.std(residuals),
                        "trimmed_residual_mean": trimmed_mean,
                        "trimmed_residual_std": trimmed_std,
                        "tp": tp, 
                        "tn": tn, 
                        "fp": fp, 
                        "fn":fn, 
                        "fp_notused": fp_nothing,
                        "ceq_accuracy": ceq_acc,
                        "ceq_precision": ceq_prec,
                        "ceq_recall": ceq_recall,
                        "heq_accuracy": heq_acc,
                        "heq_precision": heq_prec,
                        "heq_recall": heq_recall,
                        "cbl_accuracy": cbl_acc,
                        "cbl_precision": cbl_prec,
                        "cbl_recall": cbl_recall,
                        "noise_accuracy": noise_acc,
                        "noise_precision": noise_prec,
                        "noise_recall": noise_recall}
                else:
                    dic = {"epoch": epoch,
                        "n_picks": n_picks,
                        "n_picked": len(residuals),
                        "tolerance": tol,
                        "accuracy": acc,
                        "precision": prec,
                        "residual_mean": np.mean(residuals),
                        "residual_std": np.std(residuals),
                        "trimmed_residual_mean": trimmed_mean,
                        "trimmed_residual_std": trimmed_std,
                        "recall": recall, 
                        "tp": tp, 
                        "tn": tn, 
                        "fp": fp, 
                        "fn": fn, 
                        "fp_notused": fp_nothing}

                return dic

            # Results for individual picks 
            dict_p1 = calc_stats(Y_obs, Y_est1, n_picks1, resids1, fp_all, df=df)
            dict_p2 = calc_stats(Y_obs2, Y_est2, n_picks2, resids2, fp_p2)
            # Results when having all the picks for a waveform counts as a success 
            tmp = np.full(len(Y_est1), 1)
            tmp[0:len(Y_est2)] = Y_est2
            combined_Yest = Y_est1 * tmp
            dict_comb = calc_stats(Y_obs, combined_Yest, n_picks1+n_picks2, index_resid, fp_all)

            results_p1.append(dict_p1)
            results_p2.append(dict_p2)
            results_comb.append(dict_comb)
            # save JS seperatley since it is the same for all picks
            results_js.append(
                    {"epoch": epoch,
                    "tolerance": tol,
                    "js_mean": np.mean(jaccard_sims),
                    "js_min": np.min(jaccard_sims),
                    "js_max": np.max(jaccard_sims),
                    "js_dist": np.histogram(jaccard_sims),
                    "js_proba_mean": np.mean(jaccard_sims_proba),
                    "js_proba_min": np.min(jaccard_sims_proba),
                    "js_proba_max": np.max(jaccard_sims_proba),
                    "js_proba_dist": np.histogram(jaccard_sims_proba)})

            js_arrays.append({"epoch":epoch, 
                                "tol":tol,
                                "js_pick": jaccard_sims,
                                "js_proba":jaccard_sims_proba})

        return results_p1, results_p2, results_comb, results_js, js_arrays

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

    @staticmethod
    def calculate_pick_similiarity(Y_est, Y_act, allowed_pick_diff=30):
        intersect = 0
        resids = np.full(len(Y_act), np.nan)
        for i in range(len(Y_act)):
            closeT = np.where(abs(Y_act[i]-Y_est) < allowed_pick_diff)[0]
            if len(closeT) > 0:
                if len(closeT) > 1:
                    print("Muliple matching detections")
                    print("act", Y_act)
                    print("est", Y_est)
                    closeT = closeT[np.where(abs(Y_act[i]-Y_est[closeT]) == np.min(abs(Y_act[i]-Y_est[closeT])))[0]]
                    # TODO: I should probably incorporate probabilities into this so it can break a tie
                    if len(closeT) > 1:
                        print("There were two picks the same distance away - randomly choose one")
                        closeT = closeT[np.random.randint(0, len(closeT), 1)]
                    print("Choosing", Y_est[closeT])
                assert len(closeT) <= 1, "Multiple matching detections"
                intersect += 1
                resids[i] = (Y_act[i]-Y_est[closeT])

        union = len(Y_est) + len(Y_act) - intersect

        if union == 0:
            assert len(Y_est)==0 and len(Y_act)==0, "union is 0 but pick array sizes are not"
            assert intersect == 0
            return 1, resids
        # print("jaccard similarity:", intersect/union)
        return intersect/union, resids

    @staticmethod
    def calculate_jaccard_similarity_proba(Y1, Y2, thresh):
        Y1 = Y1.reshape((Y1.shape[0], 1))
        Y2= Y2.reshape((Y2.shape[0], 1))
        Y1_bin = (Y1 >= thresh) * 1
        Y2_bin = (Y2 >= thresh) * 1
        compare = Y1_bin * Y2_bin
        intersect = len(np.where(compare)[0])
        union = len(np.where(Y1_bin == 1)[0]) + len(np.where(Y2_bin == 1)[0]) - intersect

        if union == 0:
            assert len(np.where(Y1_bin == 1)[0]) == 0 and len(np.where(Y2_bin == 1)[0]) == 0, "Union is 0 but lengths are not"
            assert intersect == 0
            return 1

        return intersect/union

    ### Functions from MultiModelEval
    def load_model_state(self, model_in):
        if (not os.path.exists(model_in)):
            print("Model", model_in, " does not exist")
            raise Exception
        print(f"Loading {model_in}")
        check_point = torch.load(model_in)
        training_loss = {'epoch': check_point['epoch'], 'training_loss': check_point['loss']}
        #print(training_loss)
        self.model.load_state_dict(check_point['model_state_dict'])
        self.model.eval()

        return training_loss

    @staticmethod
    def load_dataset(h5f, mew=False):
        with h5py.File(h5f, "r") as f:
            X_test = f['X'][:]
            T_test = f['Pick_index'][:]
            if mew:
                Y_test = f["Y"][:]
                T_test2 = f["Pick_index2"][:]

        if mew:
            return X_test, Y_test, T_test, T_test2

        return X_test, T_test

    def evaluate_over_models(self, data_path, epochs, model_states_path, output_dir, 
                                tols, pick_method, df=None, save_proba=False):

        # Not going to save all posterior probs here because that seems uneccessary, 
        # just choose a model and then evaluate using evalutor and save


        if (not os.path.exists(output_dir)):
            os.makedirs(output_dir)
            
        start = time.time()
        resids = []
        metrics = []
        X_test, T_test = self.load_dataset(data_path)

        if save_proba:
            probafile = h5py.File(f'{output_dir}/proba.h5', "w")
            probafile.create_group("ModelOutputs")

        for epoch in epochs:
            model_to_test = glob.glob(os.path.join(model_states_path, f"*{epoch:03}.pt"))
            assert len(model_to_test)==1, "Wrong number of model paths found"
            training_loss = self.load_model_state(model_to_test[0])
            if epoch < 0:
                training_loss.update({"epoch":epoch})
            #self.set_model(self.model)
            post_probs, pick_info = self.apply_model(X_test, pick_method=pick_method)
            Y_proba = pick_info[1]
            T_est_index = pick_info[0]

            for i in range(len(T_test)):
                # Removing this becuase I want to be able to calculate confusion matrices after the fact 
                # - don't know if I actually need to do this
                # if (T_test[i] < 0):
                #     break
                resids.append({'model': epoch,
                                'true_lag': T_test[i],
                                'residual': T_test[i] - T_est_index[i],
                                'probability': Y_proba[i]})

            metric = self.tabulate_metrics(T_test, Y_proba, T_est_index, epoch, tols=tols, df=df)
            for m in metric:
                m.update(training_loss)
                metrics.append(m)

            if save_proba:
                probafile.create_dataset("%s.Y_est"%epoch, data=post_probs)
                probafile.create_dataset("%s.Y_max_proba"%epoch, data=Y_proba)
                probafile.create_dataset("%s.T_est_index"%epoch, data=T_est_index)


        end = time.time()
        print("Total time:", end-start)

        if save_proba:
            probafile.close()
            
        df = pd.DataFrame(metrics) 
        df.to_csv(f'{output_dir}/metrics.csv', index=False)

        df_resid = pd.DataFrame(resids)
        df_resid.to_csv(f'{output_dir}/residuals.csv', index=False)