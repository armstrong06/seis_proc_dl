#!/usr/bin/env python3
from operator import truediv
import numpy as np
import pandas as pd
import warnings
import os
import h5py
import torch
import torch.utils.data
from sklearn.metrics import confusion_matrix
import time

import sys
sys.path.append('/home/armstrong/Research/mew_threecomp')
#sys.path.append('/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/TestGpdUnet/jaccard_test_dev')

from train_gpd_unet import UNet, NumpyDataset

# from scipy import stats
# import sys
# sys.path.append('/home/jfarrell/MACHINE_LEARNING')
# from zach_unet import ZUNet
# from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
# from sklearn.metrics import log_loss
# import matplotlib.pyplot as plt


warnings.simplefilter("ignore")
device = torch.device("cuda:0")

def get_n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_outer_fence_mean_standard_deviation(residuals):
     
     q1, q3 = np.percentile(residuals, [25, 75])
     iqr = q3 - q1 
     of1 = q1 - 1.5*iqr
     of3 = q3 + 1.5*iqr
     trimmed_residuals = residuals[(residuals > of1) & (residuals < of3)]
     #print(len(trimmed_residuals), len(residuals), of1, of3)
     xmean = np.mean(trimmed_residuals)
     xstd = np.std(trimmed_residuals) 
     return xmean, xstd 

def apply_model(unet, X_test, Y_test, lsigmoid=True, batch_size=128,
                center_window = None,
                dev = torch.device("cuda:0")):
    T_est_index = np.zeros(X_test.shape[0], dtype='i')
    Y_proba = np.zeros(X_test.shape[0])
    n_samples = Y_test.shape[1]
    #print(Y_test.shape)
    #Y_proba = np.zeros(Y_test.shape)
    #Y_pred = np.zeros(Y_test.shape, dtype='int')
    Y_est_all = np.zeros([X_test.shape[0], X_test.shape[1]])

    for i1 in range(0, X_test.shape[0], batch_size):
        i2 = min(X_test.shape[0], i1 + batch_size)
        X_temp = np.copy(X_test[i1:i2, :, :])
        Y_temp = np.copy(Y_test[i1:i2, :, :])
        X = torch.from_numpy(X_temp.transpose((0, 2, 1))).float().to(dev)
        #Y_obs = torch.from_numpy(Y_temp.transpose((0, 2, 1))).float().to(dev)
        if (lsigmoid):
            Y_est = torch.sigmoid(unet.forward(X))
        else:
            Y_est = unet.forward(X)
        Y_est = Y_est.squeeze()
        Y_est_all[i1:i2, :] = Y_est.to('cpu').detach().numpy()

        # Pick indices and probabilities
        if (center_window is None):
            values, indices = Y_est.max(dim=1)
            values = values.to('cpu').flatten()
            indices = indices.to('cpu').flatten()
            for k in range(i1,i2):
                Y_proba[k] = values[k-i1]
                T_est_index[k] = indices[k-i1]
                #print(Y_proba[k], T_est_index[k])
        else:
            j1 = int(n_samples/2 - center_window)
            j2 = int(n_samples/2 + center_window)
            Y_sub = Y_est[:,j1:j2]
            #print(Y_sub.shape, j1, j2)
            #print(Y_test.shape)
            values, indices = Y_sub.max(dim=1)
            values = values.to('cpu').flatten()
            indices = indices.to('cpu').flatten() + j1
            for k in range(i1,i2):
                Y_proba[k] = values[k-i1]
                T_est_index[k] = indices[k-i1]

    return Y_proba, T_est_index, Y_est_all


def apply_model_mew(unet, X_test, Y_test, min_tol, sliding_window_size=100, lsigmoid=True, batch_size=128,
                    dev=torch.device("cuda:0")):
    n_samples = Y_test.shape[1]
    # Max number of picks possible - likely an overestimate?
    # added +1 because I got an error at line 134 saying "cannot copy sequence with size 11 to array axis with dimension 10" 
    max_pick_cnt = n_samples//sliding_window_size + 1
    # initialize array to hold pick indices - use this method so can store arrays of picks instead of single values
    # Will probably always be extra values space in second dim but speed benefit of initializing array before hand likely
    # > than additional memory usage? May better way to do this??
    T_est_index = np.full((X_test.shape[0], max_pick_cnt), -1, dtype='i')
    # Initialize the array to hold the max probability from the model corresponding to the pick time
    Y_proba = np.full((X_test.shape[0], max_pick_cnt), -1, dtype='float32')

    # Array to hold the probability output for all tested waveforms
    Y_est_all = np.zeros([X_test.shape[0], X_test.shape[1]])

    for i1 in range(0, X_test.shape[0], batch_size):
        i2 = min(X_test.shape[0], i1 + batch_size)
        # grab the X data for waveforms in this batch
        X_temp = np.copy(X_test[i1:i2, :, :])
        # Y_temp = np.copy(Y_test[i1:i2, :, :])
        X = torch.from_numpy(X_temp.transpose((0, 2, 1))).float().to(dev)
        # Y_obs = torch.from_numpy(Y_temp.transpose((0, 2, 1))).float().to(dev)

        # Put results in sigmoid function if required
        if (lsigmoid):
            Y_est = torch.sigmoid(unet.forward(X))
        else:
            Y_est = unet.forward(X)

        Y_est = Y_est.squeeze()
        # put Y_est in cpu format since using my function and not PyTorch max function
        Y_est = Y_est.to('cpu').detach().numpy()
        # store the Y probability arrays
        Y_est_all[i1:i2, :] = Y_est

        for k in range(i1, i2):
            proba_values, pick_indices = sliding_window_phase_arrival_estimator(Y_est[k-i1], sliding_window_size, thresh=min_tol)
            Y_proba[k, 0:len(proba_values)] = proba_values[:]
            T_est_index[k, 0:len(pick_indices)] = pick_indices[:]

    return Y_proba, T_est_index, Y_est_all

def sliding_window_phase_arrival_estimator(Y, window_size, thresh=0.1, min_boxcar_width=20):
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
    while i1 < Y.shape[0]:
        i2 = i1 + np.min([(Y.shape[0] - i1), window_size])
        if np.any(Y[i1:i2] >= thresh):
            # find first ind in window above thresh (start looking for max proba)
            start_win = i1 + np.where(Y[i1:i2] >= thresh)[0][0]
            # find end ind where proba has gone below thresh - if start and end inds are too close together, find a new end index
            search_win_size = 100
            possible_win_lengths = np.where(Y[start_win:start_win+search_win_size] < thresh)[0]
            
            while len(possible_win_lengths) < 1:
                search_win_size += 10
                possible_win_lengths = np.where(Y[start_win:start_win+search_win_size] < thresh)[0]

            end_win = start_win + possible_win_lengths[0]
            win_end_ind = 1
            while end_win - start_win < min_boxcar_width:
                # changed this to while loop - hopefully it doesn't break everything
                while win_end_ind >= len(possible_win_lengths):
                    search_win_size += 10
                    # TODO: add in case where increasing by 10 does not increase possible_win_lengths
                    possible_win_lengths = np.where(Y[start_win:start_win+search_win_size] < thresh)[0]
                
                end_win = start_win + possible_win_lengths[win_end_ind]
                win_end_ind += 1

            if end_win - start_win < min_boxcar_width:
                print("Y", Y[start_win:start_win+150])
                print("P", possible_win_lengths)
                print(thresh)
                print(end_win-start_win)

            assert end_win - start_win >= min_boxcar_width, "too narrow boxcar"
            proba = np.max(Y[start_win:end_win])
            pick = start_win + np.where(Y[start_win:end_win] == proba)[0][0]

#            if len(picks) > 0 and pick - picks[-1] < 21:
#                print("picks close together", pick, picks[-1])
#                print("probas", proba, proba_values[-1])
#                if proba > proba_values[-1]:
#                    picks[-1] = pick
#                    proba_values[-1] = proba
#                    print(pick)
#                else:
#                    print(picks[-1])
#            else:
            picks.append(pick)
            proba_values.append(proba)
            
            i1 += (end_win - i1)
        else:
            i1 += window_size


    return proba_values, picks

def tabulate_metrics(T_test, Y_proba, T_est_index, epoch,
                     tols = [0.1, 0.25, 0.5, 0.75, 0.9]):
    results = [] 
    Y_obs = (T_test >= 0)*1
    n_picks = np.sum(Y_obs)
    for tol in tols:
        Y_est = (Y_proba > tol)*1

        index_resid = np.zeros(len(T_test), dtype='int')
        j = 0
        for i in range(len(T_test)):
            if (Y_obs[i] == 1 and Y_est[i] == 1):
                index_resid[j] = T_test[i] - T_est_index[i] 
                j = j + 1
        index_resid = np.copy(index_resid[0:j])
        if (len(index_resid) > 0):
            trimmed_mean, trimmed_std = compute_outer_fence_mean_standard_deviation(index_resid)
        else:
            trimmed_mean = 0
            trimmed_std = 0
        
        # I had to add label into this or it breaks when 100% accuracte. If there are more than 2 classes, will need to edit this. 
        tn, fp, fn, tp = confusion_matrix(Y_obs, Y_est, labels=[0, 1]).ravel()
        acc  = (tn + tp)/(tp + tn + fp + fn)
        prec   = tp/(tp + fp)
        recall = tp/(tp + fn)
        dic = {"epoch": epoch,
               "n_picks": n_picks,
               "n_picked": len(index_resid),
               "tolerance": tol,
               "accuracy": acc,
               "precision": prec,
               "residual_mean": np.mean(index_resid),
               "residual_std": np.std(index_resid),
               "trimmed_residual_mean": trimmed_mean,
               "trimmed_residual_std": trimmed_std,
               "recall": recall}
        #print(dic)
        results.append(dic)
        #print(tol, acc, prec, recall)
        #print(accuracy_score(Y_obs, Y_est))
        #print(precision_score(Y_obs, Y_est))
        #print(recall_score(Y_obs, Y_est))
    return results

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

def tabulate_metrics_mew(T_test, T_test2, Y_proba, T_est_index, epoch, Y_data_est, Y_data_act,
                     tols=[0.1, 0.25, 0.5, 0.75, 0.9]):
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

    for tol in tols:
        Y_est = (Y_proba > tol) * 1

        ## Residual information will not mean much here because only calculate pick residuals for those close together
        ## Can look at how many residuals have nan values though
        index_resid = np.full(len(T_test)+len(T_test2), np.nan)
        jaccard_sims = np.full(len(T_test), np.nan)
        jaccard_sims_proba = np.full(len(T_test), np.nan)
        resids1 = np.full(len(T_test), np.nan)
        resids2 = np.full(len(T_test2), np.nan)
        j = 0
        j1 = 0
        j2 = 0
        est_picks_cnt = 0
        for i in range(len(T_test)):               
            # Get picks with probability > threshold 
            est_picks = T_est_index[i][np.where(Y_proba[i] > tol)[0]]
            est_picks_cnt += len(est_picks)

            # TODO: check if this should be < or <=
            if i < len(Y_obs2):
                actual_picks = [T_test[i], T_test2[i]]
            else:
                actual_picks = [T_test[i]]
                
            js, resids = calculate_pick_similiarity(est_picks, actual_picks)

            if (Y_obs[i] == 1 and np.any(Y_est[i] == 1)):
                if i < len(Y_obs2):
                    index_resid[j:j+2] = resids[:]
                    j = j + 2
                    resids1[j1] = resids[0]
                    resids2[j2] = resids[1]
                    j1 += 1
                    j2 += 1
                else:
                    index_resid[j] = resids[0]
                    j = j + 1
                    resids1[j1] = resids[0]
                    j1 += 1

            # TODO: add in other way of measuring JS
            jaccard_sims[i] = js
            jaccard_sims_proba[i] = calculate_jaccard_similarity_proba(Y_data_act[i], Y_data_est[i], tol)


        index_resid = np.copy(index_resid[0:j])

        # assign 0 and ones based on where there was a resiudal calculated or not
        Y_est1 = (~np.isnan(resids1)) * 1
        Y_est2 = (~np.isnan(resids2)) * 1
        fp = est_picks_cnt - sum(Y_est1) - sum(Y_est2)
        assert fp >= 0, "FP is negative"

        def calc_stats(Y_obs, Y_est, n_picks, residuals, fp):
            residuals = residuals[np.where(~np.isnan(residuals))[0]]

            trimmed_mean, trimmed_std = compute_outer_fence_mean_standard_deviation(residuals)

            # I had to add label into this or it breaks when 100% accuracte. If there are more than 2 classes, will need to edit this.
            tn, fp_nothing, fn, tp = confusion_matrix(Y_obs, Y_est, labels=[0, 1]).ravel()
            # TODO: I'm not sure if this is right
            acc = (tn + tp) / (tp + tn + fp + fn)
            prec = tp / (tp + fp)
            recall = tp / (tp + fn)
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
                   "recall": recall}
            return dic

        # Results for individual picks 
        dict_p1 = calc_stats(Y_obs, Y_est1, n_picks1, resids1, fp)
        dict_p2 = calc_stats(Y_obs2, Y_est2, n_picks2, resids2, fp)
        # Results when having all the picks for a waveform counts as a success 
        tmp = np.full(len(Y_est1), 1)
        tmp[0:len(Y_est2)] = Y_est2
        combined_Yest = Y_est1 * tmp
        dict_comb = calc_stats(Y_obs, combined_Yest, n_picks1+n_picks2, index_resid, fp)

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

def compute_snr(X, Y, T, phase_type = "P", azims = None):
    """
    A simple SNR computation routine.  For z this computes the 10*log_10 of 
    the ratio of the expectations of the squared amplitudes.
    """
    n_rows = len(T)
    snrs = np.zeros(n_rows) - 999.0 
    if (phase_type != "P" and azims is None):
        print("Warning azims is none - don't use these SNRs")
    for i in range(n_rows):
        # Noise window
        if (T[i] < 0):  
            continue
        y_idx = np.where(Y[i,:] == 1)[0]
        i0 = max(0, y_idx[0] - len(y_idx))
        i1 = y_idx[0]
        if (phase_type == "P"):
            x_signal = X[i, y_idx, 2] # Index 2 should be Z
            x_noise = X[i, i0:i1, 2]
            x_signal = x_signal - np.mean(x_signal)
            x_noise = x_noise - np.mean(x_noise)
        else:
            if (azims is None):
                alpha = 0
            else:
                alpha = azims[i]*np.pi/180
            n_signal = X[i, y_idx, 0]
            n_noise  = X[i, i0:i1, 0]
            e_signal = X[i, y_idx, 1]
            e_noise  = X[i, i0:i1, 1]
            # transverse signal
            t_signal = -n_signal*np.sin(alpha) + e_signal*np.cos(alpha)
            t_noise  = -n_noise*np.sin(alpha)  + e_noise*np.cos(alpha)
            x_signal = t_signal - np.mean(t_signal)
            x_noise = t_noise - np.mean(t_noise)
        s2 = np.multiply(x_signal, x_signal) # element-wise
        n2 = np.multiply(x_noise, x_noise) # element-wise
        snr = np.mean(s2)/np.mean(n2) # Ratio of exepctations
        snr = 10*np.log10(snr)
        snrs[i] = snr
    return snrs
 

if __name__ == "__main__":
    phase_type = "P"
    duration = 10
    n_dup = 1
    n_epochs = 35
    batch_size = 128 #32 #512 #1024
    center_window = None # Force picks to be in +/- 250 samples of central part of window
    tols = np.linspace(0.4, 0.95, 10) # np.linspace(0.05, 0.95, 21) #np.linspace(0.05, 0.95, 10)
    test_yellowstone = False # Test with the NGB data - I edited this
    test_train = False # Compute pick quality on train set
    save_proba = True
    save_js = True

    pref = '/home/armstrong/Research/mew_threecomp'
    model_path = "%s/%s_models_256_0.001" % (pref, phase_type)

   # pref = '/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/TestGpdUnet/jaccard_test_dev'
   # model_path = "%s/mew_model_P_010_test.pt" % (pref, phase_type)

    #if (test_yellowstone):
    #    model_path = '../../yellowstoneMLPicker/training/python_models/'
    test_type = "test"
    outpath = model_path + "/test/mew/no_mew"
    #csv_meta = "%s/data/%s%s.%ds.%ddup.synthetic.multievent.df.csv"%(pref, test_type,  phase_type, duration, n_dup)
    
    #csv_meta = "%s/data/only_mew/%s%s.%ds.%ddup_synthetic_multievent_catalog.df.csv"%(pref, test_type,  phase_type, duration, n_dup)
    #csv_summary_p1 = '%s/%s%s.%ds.%ddup.synthetic.multievent.p1.summary.csv'%(outpath,test_type, phase_type, duration, n_dup)
    #csv_summary_p2 = '%s/%s%s.%ds.%ddup_p2_synthetic_multievent_catalog_summary.csv'%(outpath, test_type, phase_type, duration, n_dup)
    #csv_summary_comb = '%s/%s%s.%ds.%ddup_combinedpicks_synthetic_multievent_catalog_summary.csv'%(outpath, test_type, phase_type, duration, n_dup)
    #csv_summary_js = '%s/%s%s.%ds.%ddup_synthetic_multievent_catalog_js_summary.csv'%(outpath, test_type, phase_type, duration, n_dup)
    #resid_summary = '%s/%s%s.%ds.%ddup_synthetic_multievent_catalog_resid.csv'%(outpath, test_type, phase_type, duration, n_dup)
    
    csv_meta = "%s/data/%s%s.%ds.%ddup.df.csv"%(pref, test_type,  phase_type, duration, n_dup)
    csv_summary_p1 = '%s/%s%s.%ds.%ddup.p1.summary.csv'%(outpath,test_type, phase_type, duration, n_dup)
    csv_summary_p2 = '%s/%s%s.%ds.%ddup.p2.summary.csv'%(outpath, test_type, phase_type, duration, n_dup)
    csv_summary_comb = '%s/%s%s.%ds.%ddup.combinedpicks.catalog.summary.csv'%(outpath, test_type, phase_type, duration, n_dup)
    csv_summary_js = '%s/%s%s.%ds.%ddup.js.summary.csv'%(outpath, test_type, phase_type, duration, n_dup)
    resid_summary = '%s/%s%s.%ds.%ddup.catalog.resid.csv'%(outpath, test_type, phase_type, duration, n_dup)
    
    if save_proba:
        proba_out = '%s/%s%s.%ds.%ddup.proba.h5'%(outpath, test_type, phase_type, duration, n_dup)
    
    if save_js:
        js_out = '%s/%s%s.%ds.%ddup.jaccard.similarity.h5'%(outpath, test_type, phase_type, duration, n_dup)

    if (test_yellowstone):
        # NGB.testPMeta.10s.1dup.df.csv 
        csv_meta = "%s/data/NGB.test%sMeta.%ds.%ddup.df.csv"%(pref, phase_type, duration, n_dup)
        csv_summary_p1 = '%s/test%s.NGB.%ds.%ddup.p1.summary.csv'%(outpath, phase_type, duration, n_dup)
        csv_summary_p2 = '%s/test%s.NGB.%ds.%ddup.p2.summary.csv'%(outpath, phase_type, duration, n_dup)
        csv_summary_comb = '%s/test%s.NGB.%ds.%ddup.combinedpicks.summary.csv'%(outpath, phase_type, duration, n_dup)
        csv_summary_js = '%s/test%s.NGB.%ds.%ddup.js.summary.csv'%(outpath, phase_type, duration, n_dup)

        resid_summary = '%s/test%s.NGB.%ds.%ddup.resid.csv'%(outpath, phase_type, duration, n_dup)
    
        if save_proba:
            proba_out = '%s/test%s.NGB.%ds.%ddup.proba.h5'%(outpath, phase_type, duration, n_dup)

        if save_js:
            js_out = '%s/test%s.%ds.%ddup.synthetic.multievent.jaccard.similarity.h5'%(outpath, phase_type, duration, n_dup)

    elif(test_train):
        csv_meta = "%s/data/train%s.%ds.%ddup.synthetic.multievent.df.csv"%(pref, phase_type, duration, n_dup)
        csv_summary_p1 = '%s/train%s.%ds.%ddup.synthetic.multievent.p1.summary.csv'%(outpath, phase_type, duration, n_dup)
        csv_summary_p2 = '%s/train%s.%ds.%ddup.synthetic.multievent.p2.summary.csv'%(outpath, phase_type, duration, n_dup)
        csv_summary_comb = '%s/train%s.%ds.%ddup.synthetic.multievent.combinedpicks.summary.csv'%(outpath, phase_type, duration, n_dup)
        csv_summary_js = '%s/train%s.%ds.%ddup.synthetic.multievent.js.summary.csv'%(outpath, phase_type, duration, n_dup)
        resid_summary = '%s/train%s.%ds.%ddup.synthetic.multievent.resid.csv'%(outpath, phase_type, duration, n_dup)
    
        if save_proba:
            proba_out = '%s/train%s.%ds.%ddup.synthetic.multievent.proba.h5'%(outpath, phase_type, duration, n_dup)

        if save_js:
            js_out = '%s/train%s.%ds.%ddup.synthetic.multievent.jaccard.similarity.h5'%(outpath, phase_type, duration, n_dup)


    print(csv_meta)
    print(csv_summary_p1)
    print(csv_summary_p2)
    print(csv_summary_comb)
    print(csv_summary_js)
    print(resid_summary)

    print("Loading data...")
    if (test_train):
        h5f = "%s/data/train%s.%ds.%ddup.synthetic.multievent.h5"%(pref, phase_type, duration, n_dup)
        print(h5f)
        with h5py.File(h5f) as f:
            X_test = f['X'][:]
            Y_test = f['Y'][:]
            T_test = f['Pick_index'][:]
            T_test2 = f['Pick_index2'][:]
    else:
        h5f = "%s/data/%s%s.%ds.%ddup.h5"%(pref, test_type, phase_type, duration, n_dup)
        #h5f = "%s/data/%s%s.%ds.%ddup.synthetic.multievent.h5"%(pref, test_type, phase_type, duration, n_dup)
       # h5f = "%s/data/only_mew/%s%s.%ds.%ddup_synthetic_multievent_waveforms.h5"%(pref, test_type, phase_type, duration, n_dup)
        print(h5f)
        if (test_yellowstone):
            # NGB.testP.10s.1dup.h5 
            h5f = "%s/data/NGB.test%s.%ds.%ddup.h5"%(pref, phase_type, duration, n_dup)

        print(h5f)
        
        with h5py.File(h5f) as f:
            print("keys", f.keys())
            X_test = f['X'][:]
            Y_test = f['Y'][:]
            T_test = f['Pick_index'][:]
            #T_test2 = f['Pick_index2'][:]
            T_test2 = []

    df_meta = pd.read_csv(csv_meta)
    print(df_meta.head())
    # TODO: only need these lines when reading in file with a line for both combined events
    #odd_inds = np.arange(1, len(csv_meta), 2)
    #df_meta = df_meta.drop(odd_inds, axis=0)
    print(df_meta.head())
    azims = df_meta['source_receiver_azimuth'].to_numpy()
    # azims = None
    if (test_train and phase_type == "S"):
        azims = None
    print(X_test.shape, Y_test.shape, T_test.shape, phase_type)
    snrs = compute_snr(X_test, Y_test, T_test, phase_type, azims)
    pick_min = np.amin(T_test)/100
    pick_max = np.amax(T_test)/100
    print(np.amin(T_test), np.amax(T_test))
    # Compute
    print(T_test.shape)
     
    n_picks = np.sum([T_test > 1])
    T_est_index = np.zeros(T_test.shape[0]) - 1
    
    if (not os.path.exists(outpath)):
            os.makedirs(outpath)

    start = time.time()
    print("Initializing unet...")
    # metrics = []
    metrics_p1 = []
    metrics_p2 = []
    metrics_comb = []
    metrics_js = []
    residual_info = []
    unet = UNet(num_channels=3, num_classes=1).to(device)
    print("Number of parameters in my model:", get_n_params(unet))
    if save_proba:
        probafile = h5py.File(proba_out, "w")
        probafile.create_group("ModelOutputs")

    if save_js:
        js_file = h5py.File(js_out, "w")

    for epoch in range(n_epochs):
        print("Testing:", epoch)
        model_in = os.path.join(model_path, 'model_%s_%03d.pt'%(phase_type,epoch))
        if (not os.path.exists(model_in)):
            print("Model", model_in, " does not exist")
            continue
        check_point = torch.load(model_in)
        training_loss = {'epoch': check_point['epoch'], 'training_loss': check_point['loss']}
        #print(training_loss)
        unet.load_state_dict(check_point['model_state_dict'])
        unet.eval()

        #y_pred, y_true, y_pred_all, y_true_all, T_est_index, y_proba, val_acc = apply_model(unet, X_test, Y_test, tol=tol, lsigmoid=True, batch_size=batch_size, dev=device)
        # Y_proba, T_est_index, Y_est_all = apply_model(unet, X_test, Y_test,
        #                                            lsigmoid=True, batch_size=batch_size, dev=device,
        #                                            center_window = center_window)

        Y_proba, T_est_index, Y_est_all = apply_model_mew(unet, X_test, Y_test,
                                                          min_tol=tols[0], sliding_window_size=100, lsigmoid=True,
                                                          batch_size=batch_size, dev=device)

        print(Y_proba.shape, Y_est_all.shape)

        for i in range(len(T_test)):
            if T_test[i] < 0:
                break

            if i < len(T_test2):
                actual_picks = [T_test[i], T_test2[i]]
                true_lag2 = T_test2[i]
            else:
                actual_picks = [T_test[i]]
                true_lag2 = None

            print(true_lag2)

            est_picks = T_est_index[i][T_est_index[i] > 0]
            est_proba = Y_proba[i][Y_proba[i] > 0]

            #js, resids = calculate_pick_similiarity(est_picks, actual_picks)

            if len(est_picks) == 0:
                residual_info.append( {'epoch': epoch,
                'waveform_index':i,
                'pick_index': None,
                'probability': None,
                'true_lag1': T_test[i],
                'true_lag2': true_lag2,
                'residual1': None,
                'residual2': None,
                'is_match': False,
                'snr': snrs[i] })
            else:
                # Didn't put the jaccard similairty in here because it would just be the jaccard similarity using the minimum tolerace value
                all_resids1 = np.full(len(est_picks), np.nan)
                all_resids2 = np.full(len(est_picks), np.nan)
                for pind in range(len(est_picks)):
                    all_resids1[pind] = (T_test[i] - est_picks[pind])
                    if i < len(T_test2):
                        all_resids2[pind] = (T_test2[i]-est_picks[pind])


                match1 = None
                match2 = None
                # TODO: Do I need to handle it when these are the same (usually when only one pick)??
                if len(all_resids1) > 0 and ~np.isnan(np.unique(all_resids1))[0]:
                    match1 = np.where(abs(all_resids1) == np.min(abs(all_resids1)))[0][0]
                if len(all_resids2) > 0 and ~np.isnan(np.unique(all_resids2))[0]:
                    match2 = np.where(abs(all_resids2) == np.min(abs(all_resids2)))[0][0]
                
                for pind in range(len(est_picks)):
                    # TODO: Is match doesn't really mean anything - no minimum proximity required just that it is the closest pick 
                    is_match = False
                    if pind == match1 or pind == match2:
                        is_match = True

                    residual_info.append( {'epoch': epoch,
                                    'waveform_index':i,
                                    'pick_index': est_picks[pind],
                                    'probability': est_proba[pind],
                                    'true_lag1': T_test[i],
                                    'true_lag2': true_lag2,
                                    'residual1': all_resids1[pind],
                                    'residual2': all_resids2[pind],
                                    'is_match': is_match,
                                    'snr': snrs[i] })

        metric_p1, metric_p2, metric_comb, metric_js, js_arrays = tabulate_metrics_mew(T_test, T_test2, Y_proba, T_est_index, epoch=epoch, Y_data_est=Y_est_all,
                                                                             Y_data_act=Y_test, tols=tols)

        # TODO: check this - I'm not sure what update is doing
        for m in metric_p1:
            m.update(training_loss)
            metrics_p1.append(m)

        for m in metric_p2:
            m.update(training_loss)
            metrics_p2.append(m)

        for m in metric_comb:
            m.update(training_loss)
            metrics_comb.append(m)

        for js in metric_js:
            metrics_js.append(js)

        if save_proba:
            probafile.create_dataset("%s.Y_est"%epoch, data=Y_est_all)
            probafile.create_dataset("%s.Y_max_proba"%epoch, data=Y_proba)
            probafile.create_dataset("%s.T_est_index"%epoch, data=T_est_index)

        if save_js:
            group = js_file.create_group("epoch.%02d"%epoch)
            for t in range(len(tols)):
                group.create_dataset("tol.%0.2f.pick"%tols[t], data=js_arrays[t]["js_pick"])
                group.create_dataset("tol.%0.2f.proba"%tols[t], data=js_arrays[t]["js_proba"])


    if save_proba:
        probafile.close()

    if save_js:
        js_file.close()

    # Loop
    end = time.time()
    print("Total time:", end-start)

    #if (not os.path.exists(outpath)):
    #        os.makedirs(outpath)

    df = pd.DataFrame(metrics_p1)
    df.to_csv(csv_summary_p1, index=False)

    df = pd.DataFrame(metrics_p2)
    df.to_csv(csv_summary_p2, index=False)

    df = pd.DataFrame(metrics_comb)
    df.to_csv(csv_summary_comb, index=False)

    df = pd.DataFrame(metrics_js)
    df.to_csv(csv_summary_js, index=False)

    df_resid = pd.DataFrame(residual_info)
    df_resid.to_csv(resid_summary, index=False)
#!/usr/bin/env python3
