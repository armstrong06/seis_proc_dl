import sys
sys.path.append('..')
from data_processing import extract_events
from detectors.data_processing import make_yboxcar as boxcar
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(49230)

def sample_on_evid(df, train_frac = 0.8):
    """
    Samples the waveforms on event IDs.  The idea is to prevent the possibility
    of our classifier learning a source time function.
    """
    print("Splitting data...")
    evids = df['evid'].values
    unique_evids = np.unique(evids)
    evids_train, evids_test = train_test_split(unique_evids, train_size=train_frac)

    rows_train = np.where(np.isin(evids, evids_train))[0]
    rows_test = np.where(np.isin(evids, evids_test))[0]

    assert len(rows_test) + len(rows_train) == len(evids), "Did not get all events"

    # Make sure I got all the rows
    frac = len(rows_train)/len(evids) + len(rows_test)/len(evids)
    assert abs(frac - 1.0) < 1.e-10, 'failed to split data'
    # Ensure no test evid is in the training set
    for row_test in rows_test:
        r = np.isin(row_test, rows_train, assume_unique=True)
        assert not r, 'failed to split data'
    print("N_train: %d, Training fraction: %.2f"%(
          len(rows_train), len(rows_train)/len(evids)))
    print("N_test: %d, Testing fraction: %.2f"%(
          len(rows_test), len(rows_test)/len(evids)))
    return rows_train, rows_test #print(rows_train, len(rows_train), len(rows_train)/len(evids))

def compute_pad_length(window_duration = 8.0, dt = 0.01):
    n_samples = int(window_duration/dt + 0.5) + 1
    while (n_samples%16 != 0):
        n_samples = n_samples + 1
    return n_samples

def augment_data(X, Y, n_duplicate,
                 window_duration = 8.0, dt = 0.01,
                 target_shrinkage = None,
                 lrand = True, for_python=True, meta_for_boxcar=None):
    # Determine the sizes of the data
    print("initial", X.shape, Y.shape)
    n_obs = X.shape[0]
    n_samples = X.shape[1]
    n_comps = X.shape[2]
    # Compute a safe window size for the network architecture
    n_samples_in_window = compute_pad_length(window_duration, dt)
    if (target_shrinkage is None):
        n_samples_in_target = n_samples_in_window
        start_y = 0
        end_y = 0
    else:
        assert target_shrinkage > 0 and target_shrinkage < 1, 'target shrinkage must be in range (0,1)'
        n_samples_in_target = int(target_shrinkage*n_samples_in_window)
        start_y = (n_samples_in_window - n_samples_in_target)//2 
        end_y = n_samples_in_window - n_samples_in_target - start_y
    
    # Define the lags
    #max_lag = int(0.9*n_samples_in_window/2) # TODO: May want to multiply by like 0.95 - worst case is starting half-way into pick window
    # TODO: do not hard code this
    max_lag = 250
    
    if (lrand):
        lags = np.random.randint(-max_lag, max_lag+1, size=n_obs*n_duplicate)
    else:
        lags = np.zeros(n_obs*n_duplicate, dtype='i')
    
    # Without lags this will center the picks
    start_sample = int(n_samples/2 - n_samples_in_window/2)
    pick_window_index = int(n_samples_in_window/2)
    print(start_sample, n_samples_in_window)

    # Allocate space
    t = np.zeros([n_samples_in_window, n_comps])
    T_index = np.zeros(n_obs*n_duplicate, dtype='int')
    X_new = np.zeros([n_obs*n_duplicate, n_samples_in_window, n_comps])

    if (for_python):
        Y_new = np.zeros([n_obs*n_duplicate, n_samples_in_target, 1])
    else:
        Y_new = np.zeros([n_obs*n_duplicate, n_samples_in_target]) 

    # Sample and augment data
    for idup in range(n_duplicate):
        for iobs in range(n_obs):
            # I added this condition
            if n_samples == 1008:
                start_index = 0
            else:
                start_index = start_sample + lags[idup*n_obs+iobs]
            assert start_index >= 0, 'this should never happen'
            end_index = start_index + n_samples_in_window
            
            T_index[idup*n_obs+iobs] = pick_window_index - lags[idup*n_obs+iobs] 

            if X[iobs, start_index:end_index, :].shape[0] != 1008:
                print('here is a problem')
            t[:,:] = np.copy(X[iobs, start_index:end_index, :])
            
            # min/max rescale
            tscale = np.amax(np.abs(t))
            t = t/tscale
            
            # Copy
            X_new[idup*n_obs+iobs, :, :] = t[:,:]
            # if (for_python):
            #     # Noise and signal Y had slightly different formats for some reason. Noise does not need 0 index,
            #     # signal does
            #     try:
            #         Y_new[idup*n_obs+iobs, :, 0] = Y[iobs, start_index+start_y:end_index-end_y]
            #     except:
            #         try:
            #             Y_new[idup * n_obs + iobs, :, 0] = Y[iobs, start_index + start_y:end_index - end_y, 0]
            #         except:
            #             assert Exception("this didn't work either")
            # else:
            #     Y_new[idup*n_obs+iobs, :] = Y[iobs, start_index+start_y:end_index-end_y]

    # TODO: I dont know if this will work correct when there are duplicates 
    if meta_for_boxcar is not None:
        Y_new = boxcar.add_boxcar(meta_for_boxcar, {0: 21, 1: 31, 2: 51}, X_new, Y, T_index)

    print("final", X_new.shape, Y_new.shape, T_index.shape)
    return X_new, Y_new, T_index