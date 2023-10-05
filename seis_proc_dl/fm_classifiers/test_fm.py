#!/usr/bin/env python3
import os
import glob
import h5py
import pandas as pd
from train_fm import FMNet
from train_fm import randomize_start_times_and_normalize
import numpy as np
import torch
from sklearn import metrics

def accuracy_precision_recall_up(y_true, y_pred):
    n_tp = 0 
    n_tn = 0 
    n_fp = 0 
    n_fn = 0 
    for i in range(len(y_pred)):
        if (y_true[i] == 1 and y_pred[i] == 1): 
            n_tp = n_tp + 1 
        elif (y_true[i] == 1 and y_pred[i] ==-1): 
            n_fn = n_fn + 1 
        elif (y_true[i] ==-1 and y_pred[i] == 1):
            n_fp = n_fp + 1 
        elif (y_true[i] ==-1 and y_pred[i] ==-1):
            n_tn = n_tn + 1 
    accuracy = (n_tp + n_tn)/(n_tp + n_tn + n_fp + n_fn)
    precision = n_tp/(n_tp + n_fp)
    recall = n_tp/(n_tp + n_fn) 
    return accuracy, precision, recall

def accuracy_precision_recall_down(y_true, y_pred):
    n_tp = 0 
    n_tn = 0 
    n_fp = 0 
    n_fn = 0 
    for i in range(len(y_pred)):
        if (y_true[i] ==-1 and y_pred[i] ==-1): 
            n_tp = n_tp + 1 
        elif (y_true[i] ==-1 and y_pred[i] == 1):
            n_fn = n_fn + 1 
        elif (y_true[i] == 1 and y_pred[i] ==-1): 
            n_fp = n_fp + 1 
        elif (y_true[i] == 1 and y_pred[i] == 1):
            n_tn = n_tn + 1 
    accuracy = (n_tp + n_tn)/(n_tp + n_tn + n_fp + n_fn)
    precision = n_tp/(n_tp + n_fp)
    recall = n_tp/(n_tp + n_fn)
    return accuracy, precision, recall



if __name__ == "__main__":
    np.random.seed(88834)
 
    device = torch.device("cuda:0")
   # model_dir = "magna_finetuned_models"
    model_dir = "finetuned" 
    print("change model dir", model_dir)
    h5_test_fname = "uuss_data/uuss_test.h5"
    h5_valid_fname = "uuss_data/uuss_validation.h5"
    valid_summary = 'validation_summary.csv'
    test_summary = 'test_summary.csv'
    valid_metrics = 'validation_metrics.csv'
    test_metrics = 'test_metrics.csv'
    fine_tune_model = 'ca_models/models_007.pt'
    validation_df = pd.read_csv('uuss_data/uuss_validation.csv', dtype = {"location" : object})
    test_df = pd.read_csv('uuss_data/uuss_test.csv', dtype = {"location" : object})
    n_epochs = 35
    batch_size = 256

    validation_rows = np.arange(0, len(validation_df))
    validation_eq_rows         = validation_rows[ (validation_df.event_type == 'le') & (validation_df.evid >= 60000000) ]
    validation_blast_rows      = validation_rows[ (validation_df.event_type == 'qb') ]
    validation_historical_rows = validation_rows[ (validation_df.event_type == 'le') & (validation_df.evid < 60000000) ]

    test_rows = np.arange(0, len(test_df))
    test_eq_rows         = test_rows[ (test_df.event_type == 'le') & (test_df.evid >= 60000000) ]
    test_blast_rows      = test_rows[ (test_df.event_type == 'qb') ]
    test_historical_rows = test_rows[ (test_df.event_type == 'le') & (test_df.evid < 60000000) ]

    print("Loading data...", h5_valid_fname)
    h5valid = h5py.File(h5_valid_fname, 'r')
    X_valid = h5valid['X'][:]
    Y_valid = h5valid['Y'][:]
    h5valid.close()
    assert len(Y_valid) == len(validation_df), 'y validation size != validation df size'

    h5test = h5py.File(h5_test_fname, 'r')
    X_test = h5test['X'][:]
    Y_test = h5test['Y'][:]
    h5test.close()
    assert len(Y_test) == len(test_df), 'y test size != test df size'

    print("Randomizing start times...")
    X_valid = randomize_start_times_and_normalize(X_valid)
    X_test = randomize_start_times_and_normalize(X_test)
 
    polarity = np.asarray([1,-1,0])

    fmnet = FMNet().to(device)
 
    d_valid = []
    d_test = []
    d_valid_metrics = []
    d_test_metrics = []
    for epoch in range(-1, n_epochs):
        if (epoch ==-1):
            model_in = fine_tune_model
        else:
            model_in = os.path.join(model_dir, 'models_%03d.pt'%(epoch+1))
        print("Loading model:", model_in)
        check_point = torch.load(model_in)
        training_loss = {'epoch': check_point['epoch'],
                         'training_loss': check_point['loss']}
        fmnet.load_state_dict(check_point['model_state_dict'])
        fmnet.eval()

        n_obs = X_valid.shape[0]
        p_up = np.zeros(n_obs) + 10
        p_down = np.zeros(n_obs) + 10
        p_unknown = np.zeros(n_obs) + 10
        y_hat = np.zeros(n_obs, dtype='int')
        for b1 in range(0, n_obs, batch_size): 
            b2 = min(n_obs, b1 + batch_size)
            nb = b2 - b1
            X = np.zeros([nb, 1, 400])
            X[:,0,:] = X_valid[b1:b2,:]
            X_hat = torch.from_numpy(X).float().to(device)
            p = fmnet.forward(X_hat)
            pcpu = p.cpu().data.numpy()
            for s in range(nb):
                df_row = validation_df.iloc[b1 + s]
                #print(df_row.first_motion, polarity[Y_valid[b1 + s]])
                assert df_row.first_motion == Y_valid[b1 + s], 'metadata doesnt match targets'
                y_pred = polarity[np.argmax(pcpu[s,:])] # change this if using [0, 1, 2]
                d = {'epoch': epoch+1,
                     'training_loss': training_loss['training_loss'],
                     'p_up': float(pcpu[s,0]),
                     'p_down': float(pcpu[s,1]),
                     'p_unknown': float(pcpu[s,2]),
                     'y_pred': y_pred,
                     'y': Y_valid[b1+s],
                     'event_type': df_row.event_type,
                     'evid': df_row.evid,
                     'source_receiver_distance': df_row.source_receiver_distance,
                     'network': df_row.network,
                     'station': df_row.station,
                     'channel': df_row.channelz,
                     'location': df_row.location,
                     'magnitude': df_row.magnitude }
                y_hat[b1 + s] = y_pred
                d_valid.append(d)
 
            p_up[b1:b2] = pcpu[0:nb,0]
            p_down[b1:b2] = pcpu[0:nb,1]
            p_unknown[b1:b2] = pcpu[0:nb,2]

        assert np.max(p_up) <= 1, 'failed to do elements'
        
        validation_acc = metrics.accuracy_score(Y_valid, y_hat)
        validation_acc_eqc = metrics.accuracy_score(Y_valid[validation_eq_rows], y_hat[validation_eq_rows])
        validation_acc_eqc_up, validation_prec_eqc_up, validation_recall_eqc_up = accuracy_precision_recall_up(Y_valid[validation_eq_rows], y_hat[validation_eq_rows])
        
        validation_acc_blc = metrics.accuracy_score(Y_valid[validation_blast_rows], y_hat[validation_blast_rows])
        validation_acc_blc_up, validation_prec_blc_up, validation_recall_blc_up = accuracy_precision_recall_up(Y_valid[validation_blast_rows], y_hat[validation_blast_rows])

        validation_acc_eqa =  metrics.accuracy_score(Y_valid[validation_historical_rows], y_hat[validation_historical_rows])
        validation_acc_eqa_up, validation_prec_eqa_up, validation_recall_eqa_up = accuracy_precision_recall_up(Y_valid[validation_historical_rows], y_hat[validation_historical_rows])


        print("Validation accuracy:", validation_acc) #metrics.accuracy_score(Y_valid, y_hat))
        print("Validation current eq accuracy:", validation_acc_eqc) #metrics.accuracy_score(Y_valid[validation_eq_rows], y_hat[validation_eq_rows]))
        print("Validation up/down accuracy/prec/recall:", validation_acc_eqc_up, validation_prec_eqc_up, validation_recall_eqc_up) #accuracy_precision_recall_up(Y_valid[validation_eq_rows], y_hat[validation_eq_rows]))

        print("Validation current qb accuracy:", validation_acc_blc) #metrics.accuracy_score(Y_valid[validation_blast_rows], y_hat[validation_blast_rows]))
        print("Validation up/down accuracy/prec/recall:", validation_acc_blc_up, validation_prec_blc_up, validation_recall_blc_up) #accuracy_precision_recall_up(Y_valid[validation_blast_rows], y_hat[validation_blast_rows]))

        print("Validation historical eq accuracy:", validation_acc_eqa) #metrics.accuracy_score(Y_valid[validation_historical_rows], y_hat[validation_historical_rows])) 
        print("Validation up/down accuracy/prec/recall:", validation_acc_eqa_up, validation_prec_eqa_up, validation_recall_eqa_up) #accuracy_precision_recall_up(Y_valid[validation_historical_rows], y_hat[validation_historical_rows]))

        print("Confusion matrix:\n", metrics.confusion_matrix(Y_valid, y_hat, labels=[1, -1, 0])) #change labels if using [0, 1, 2]
        
        d_valid_metrics.append({
            'epoch':epoch+1, 
            'accuracy': validation_acc, 
            'current_eq_accuracy': validation_acc_eqc,
            'current_eq_acc_up': validation_acc_eqc_up, 
            'current_eq_prec_up': validation_prec_eqc_up, 
            'current_eq_recall_up': validation_prec_eqc_up, 
            'current_bl_accuracy': validation_acc_blc,
            'current_bl_acc_up': validation_acc_blc_up,
            'current_bl_prec_up': validation_prec_blc_up,
            'current_bl_recall_up': validation_prec_blc_up,
            'historic_eq_accuracy': validation_acc_eqa,                                                            
            'historic_eq_acc_up': validation_acc_eqa_up,
            'historic_eq_prec_up': validation_prec_eqa_up,
            'historic_eq_recall_up': validation_prec_eqa_up,
            })
        
        n_obs = X_test.shape[0]
        p_up_test = np.zeros(n_obs) + 10
        p_down_test = np.zeros(n_obs) + 10
        p_unknown_test = np.zeros(n_obs) + 10
        y_hat = np.zeros(n_obs, dtype='int')
        for b1 in range(0, n_obs, batch_size):
            b2 = min(n_obs, b1 + batch_size)
            nb = b2 - b1
            X = np.zeros([nb, 1, 400])
            X[:,0,:] = X_test[b1:b2,:]
            X_hat = torch.from_numpy(X).float().to(device)
            p = fmnet.forward(X_hat)
            pcpu = p.cpu().data.numpy()
            for s in range(nb):
                df_row = test_df.iloc[b1 + s]
                assert df_row.first_motion == Y_test[b1+s], 'metadata doesnt match targets' # this breaks if using [0, 1, 2]
                y_pred = polarity[np.argmax(pcpu[s,:])] # Change this if using [0, 1, 2]
                d = {'epoch': epoch+1,
                     'training_loss': training_loss['training_loss'],
                     'p_up': float(pcpu[s,0]),
                     'p_down': float(pcpu[s,1]),
                     'p_unknown': float(pcpu[s,2]),
                     'y_pred': y_pred,
                     'y': Y_test[b1+s],
                     'event_type': df_row.event_type,
                     'evid': df_row.evid,
                     'source_receiver_distance': df_row.source_receiver_distance,
                     'network': df_row.network,
                     'station': df_row.station,
                     'channel': df_row.channelz,
                     'location': df_row.location,
                     'magnitude': df_row.magnitude}
                y_hat[b1 + s ] = y_pred
                d_test.append(d)

            p_up_test[b1:b2] = pcpu[0:nb,0]
            p_down_test[b1:b2] = pcpu[0:nb,1]
            p_unknown_test[b1:b2] = pcpu[0:nb,2]
        assert np.max(p_up_test) <= 1, 'failed to do elements'

        test_acc = metrics.accuracy_score(Y_test, y_hat)
        test_acc_eqc = metrics.accuracy_score(Y_test[test_eq_rows], y_hat[test_eq_rows])
        test_acc_eqc_up, test_prec_eqc_up, test_recall_eqc_up = accuracy_precision_recall_up(Y_test[test_eq_rows], y_hat[test_eq_rows])

        test_acc_blc = metrics.accuracy_score(Y_test[test_blast_rows], y_hat[test_blast_rows])
        test_acc_blc_up, test_prec_blc_up, test_recall_blc_up = accuracy_precision_recall_up(Y_test[test_blast_rows], y_hat[test_blast_rows])

        test_acc_eqa =  metrics.accuracy_score(Y_test[test_historical_rows], y_hat[test_historical_rows])
        test_acc_eqa_up, test_prec_eqa_up, test_recall_eqa_up = accuracy_precision_recall_up(Y_test[test_historical_rows], y_hat[test_historical_rows])

        print("Test accuracy:", test_acc) #metrics.accuracy_score(Y_test, y_hat))
        print("Test current eq accuracy:", test_acc_eqc) #metrics.accuracy_score(Y_test[test_eq_rows], y_hat[test_eq_rows]))
        print("Test up/down accuracy/prec/recall:", test_acc_eqc_up, test_prec_eqc_up, test_recall_eqc_up) #accuracy_precision_recall_up(Y_test[test_eq_rows], y_hat[test_eq_rows]))

        print("Test current qb accuracy:", test_acc_blc) #metrics.accuracy_score(Y_test[test_blast_rows], y_hat[test_blast_rows]))
        print("Test up/down accuracy/prec/recall:", test_acc_blc_up, test_prec_blc_up, test_recall_blc_up) #accuracy_precision_recall_up(Y_test[test_blast_rows], y_hat[test_blast_rows]))

        print("Test historical eq accuracy:", test_acc_eqa) #metrics.accuracy_score(Y_test[test_historical_rows], y_hat[test_historical_rows]))
        print("Test up/down accuracy/prec/recall:", test_acc_eqa_up, test_prec_eqa_up, test_recall_eqa_up) #accuracy_precision_recall_up(Y_test[test_historical_rows], y_hat[test_historical_rows]))
        print("Confusion matrix:\n", metrics.confusion_matrix(Y_test, y_hat, labels=[1, -1, 0])) #change this if using [0, 1, 2]

        print("")

        d_test_metrics.append({
            'epoch':epoch+1,
            'accuracy': test_acc,
            'current_eq_accuracy': test_acc_eqc,
            'current_eq_acc_up': test_acc_eqc_up,
            'current_eq_prec_up': test_prec_eqc_up,
            'current_eq_recall_up': test_prec_eqc_up,
            'current_bl_accuracy': test_acc_blc,
            'current_bl_acc_up': test_acc_blc_up,
            'current_bl_prec_up': test_prec_blc_up,
            'current_bl_recall_up': test_prec_blc_up,
            'historic_eq_accuracy': test_acc_eqa,
            'historic_eq_acc_up': test_acc_eqa_up,
            'historic_eq_prec_up': test_prec_eqa_up,
            'historic_eq_recall_up': test_prec_eqa_up,
            })

    # Loop
    df = pd.DataFrame(d_valid)
    df.to_csv(valid_summary, index = False)

    df = pd.DataFrame(d_test)
    df.to_csv(test_summary, index = False)

    df = pd.DataFrame(d_valid_metrics)
    df.to_csv(valid_metrics, index=False)

    df = pd.DataFrame(d_test_metrics)
    df.to_csv(test_metrics, index=False)
