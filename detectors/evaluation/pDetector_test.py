#!/usr/bin/env python3
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
    epochs_to_test = [21]
    batch_size = 128 #32 #512 #1024
    center_window = 250 # Force picks to be in +/- 250 samples of central part of window
    tols = np.linspace(0.05, 0.95, 21) #np.linspace(0.05, 0.95, 10)
    test_yellowstone = True # Test with the NGB data - I edited this
    test_train = False # Compute pick quality on train set
    save_proba = True

    pref = '/home/armstrong/Research/mew_threecomp'
    model_path = "%s/%s_models_256_0.001" % (pref, phase_type)

    #if (test_yellowstone):
    #    model_path = '../../yellowstoneMLPicker/training/python_models/'
    test_type = "test"
    outpath = model_path + "/test/NGB"
   # csv_meta = "%s/data/%s%s.%ds.%ddup_synthetic_multievent_catalog.df.csv"%(pref, test_type,  phase_type, duration, n_dup)
   # csv_summary = '%s/%s%s.%ds.%ddup_synthetic_multievent_catalog_summary.csv'%(outpath, test_type, phase_type, duration, n_dup)
   # resid_summary = '%s/%s%s.%ds.%ddup_synthetic_multievent_catalog_resid.csv'%(outpath, test_type, phase_type, duration, n_dup)
    
  #  csv_meta = "%s/data/%s%s.%ds.%ddup.synthetic.multievent.df.csv"%(pref, test_type,  phase_type, duration, n_dup)
  #  csv_summary = '%s/%s%s.%ds.%ddup.synthetic.multievent.summary.csv'%(outpath, test_type, phase_type, duration, n_dup)
  #  resid_summary = '%s/%s%s.%ds.%ddup.synthetic.multievent.resid.csv'%(outpath, test_type, phase_type, duration, n_dup)
    csv_meta = "%s/data/%s%s.%ds.%ddup.df.csv"%(pref, test_type,  phase_type, duration, n_dup)
    csv_summary = '%s/%s%s.%ds.%ddup.summary.csv'%(outpath, test_type, phase_type, duration, n_dup)
    resid_summary = '%s/%s%s.%ds.%ddup.resid.csv'%(outpath, test_type, phase_type, duration, n_dup)
    
    if save_proba:
        #proba_out = '%s/%s%s.%ds.%ddup.proba.synthetic.multievent.h5'%(outpath, test_type, phase_type, duration, n_dup)
        proba_out = '%s/%s%s.%ds.%ddup.proba.h5'%(outpath, test_type, phase_type, duration, n_dup)

    if (test_yellowstone):
        # NGB.testPMeta.10s.1dup.df.csv 
        csv_meta = "%s/data/test%s.NGB.%ds.%ddup.df.csv"%(pref, phase_type, duration, n_dup)
        csv_summary = '%s/test/test%s.NGB.%ds.%ddup.summary.csv'%(model_path, phase_type, duration, n_dup)
        resid_summary = '%s/test/test%s.NGB.%ds.%ddup.resid.csv'%(model_path, phase_type, duration, n_dup)
    
        if save_proba:
            proba_out = '%s/test%s.NGB.%ds.%ddup.proba.h5'%(outpath, phase_type, duration, n_dup)

    elif(test_train):
        csv_meta = "%s/data/train%s.%ds.%ddup.synthetic.multievent.df.csv"%(pref, phase_type, duration, n_dup)
        csv_summary = '%s/train%s.%ds.%ddup.summary.csv'%(outpath, phase_type, duration, n_dup)
        resid_summary = '%s/train%s.%ds.%ddup.resid.csv'%(outpath, phase_type, duration, n_dup)
    
        if save_proba:
            proba_out = '%s/train%s.%ds.%ddup.proba.h5'%(outpath, phase_type, duration, n_dup)

    print(csv_meta)
    print(csv_summary)
    print(resid_summary)
    
    print("Loading data...")
    if (test_train):
        h5f = "%s/data/train%s.%ds.%ddup.synthetic.multievent.h5"%(pref, phase_type, duration, n_dup)
        print(h5f)
        with h5py.File(h5f) as f:
            X_test = f['X'][:]
            Y_test = f['Y'][:]
            T_test = f['Pick_index'][:]
    else:
        #h5f = "%s/data/%s%s.%ds.%ddup_synthetic_multievent_waveforms.h5"%(pref, test_type, phase_type, duration, n_dup)
        #h5f = "%s/data/%s%s.%ds.%ddup.synthetic.multievent.h5"%(pref, test_type, phase_type, duration, n_dup)
        h5f = "%s/data/%s%s.%ds.%ddup.h5"%(pref, test_type, phase_type, duration, n_dup)
        if (test_yellowstone):
            # NGB.testP.10s.1dup.h5 
            h5f = "%s/data/test%s.NGB.%ds.%ddup.h5"%(pref, phase_type, duration, n_dup)

        print(h5f)
        
        with h5py.File(h5f) as f:
            print("keys", f.keys())
            X_test = f['X'][:]
            Y_test = f['Y'][:]
            T_test = f['Pick_index'][:]
            #T_test2 = f['Pick_index2'][:]

    df_meta = pd.read_csv(csv_meta)
    print(df_meta.head())
    #odd_inds = np.arange(1, len(csv_meta), 2)
    #df_meta = df_meta.drop(odd_inds, axis=0)
    #print(df_meta.head())
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

    """
    zunet = ZUNet(num_channels=3, num_classes=1).to(device)
    print("Number of parameters in Zach's model:", get_n_params(zunet))
    if (phase_type == "P"):
        print(zach_p_model)
        check_point = torch.load(zach_p_model, map_location=device)
    else:
        check_point = torch.load(zach_s_model)
    zunet.load_state_dict(check_point['model_state_dict'])
    print("Applying Zach's model...")
    zunet.eval()

    y_pred, y_true, y_pred_all, y_true_all, T_est_index, y_proba, val_acc = apply_model(zunet, X_test, Y_test, tol=tol, batch_size=batch_size, lsigmoid=False, dev=device)
    print("Testing Zach's model...")
    tabulate_metrics(y_pred, y_true, y_pred_all, y_true_all, T_test, T_est_index, y_proba, tol)
    """

    """
    I commented this out
    zunet = ZUNet(num_channels=3, num_classes=1).to(device)
    print("Number of parameters in Zach's model:", get_n_params(zunet))
    if (phase_type == "P"):
        check_point = torch.load(zach_p_model, map_location=device)
    else:
        check_point = torch.load(zach_s_model)
    zunet.load_state_dict(check_point['model_state_dict'])
    print("Applying Zach's model...")
    zunet.eval()
    Y_proba, T_est_index = apply_model(zunet, X_test, Y_test,
                                       lsigmoid=True, batch_size=batch_size, dev=device,
                                       center_window = center_window)
    resids_zach = []
    for i in range(len(T_test)):
        if (T_test[i] < 0):
            break
        resids_zach.append( {'epoch': 0, #epoch,
                             'true_lag': T_test[i],
                             'residual': T_test[i] - T_est_index[i],
                             'probability': Y_proba[i],
                             'snr': snrs[i] } )
    zach_metrics = tabulate_metrics(T_test, Y_proba, T_est_index, epoch=0, tols=tols)
    df_zach = pd.DataFrame(zach_metrics)
    df_zach.to_csv('zach_' + csv_summary, index=False)
    df_resid_zach = pd.DataFrame(zach_metrics)
    df_resid_zach.to_csv('zach_' + resid_summary,  index=False)
    """
    
    if (not os.path.exists(outpath)):
            os.makedirs(outpath)

    start = time.time()
    print("Initializing unet...")
    metrics = []
    resids = []
    unet = UNet(num_channels=3, num_classes=1).to(device)
    print("Number of parameters in my model:", get_n_params(unet))
    if save_proba:
        probafile = h5py.File(proba_out, "w")
        probafile.create_group("ModelOutputs")

    for epoch in epochs_to_test:
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
        Y_proba, T_est_index, Y_est_all = apply_model(unet, X_test, Y_test,
                                                   lsigmoid=True, batch_size=batch_size, dev=device,
                                                   center_window = center_window)
        print(Y_proba.shape, Y_est_all.shape)
        
        for i in range(len(T_test)):
            if (T_test[i] < 0):
                break
            resids.append( {'epoch': epoch,
                            'true_lag': T_test[i],
                            'residual': T_test[i] - T_est_index[i],
                            'probability': Y_proba[i],
                            'snr': snrs[i] } ) 
        metric = tabulate_metrics(T_test, Y_proba, T_est_index, epoch=epoch, tols=tols)
        for m in metric:
            m.update(training_loss)
            metrics.append(m)

        if save_proba:
            probafile.create_dataset("%s.Y_est"%epoch, data=Y_est_all)
            probafile.create_dataset("%s.Y_max_proba"%epoch, data=Y_proba)
            probafile.create_dataset("%s.T_est_index"%epoch, data=T_est_index)


    if save_proba:
        probafile.close()
    # Loop
    end = time.time()
    print("Total time:", end-start)

    #if (not os.path.exists(outpath)):
    #        os.makedirs(outpath)

    df = pd.DataFrame(metrics) 
    df.to_csv(csv_summary, index=False)

    df_resid = pd.DataFrame(resids)
    df_resid.to_csv(resid_summary, index=False)
