#!/usr/bin/env python3
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
    # rows_train = None
    # rows_test = None
    # for unique_evid in evids_train: #unique_evids:
    #     temp = np.where(evids == unique_evid)[0]
    #     assert len(temp) > 0, 'shouldnt happen'
    #     if (not rows_train is None):
    #         rows_train = np.concatenate((rows_train, temp), axis=0)
    #     else:
    #         rows_train = np.asarray(temp[:]) #np.where(evids_train == unique_evid)
    # for unique_evid in evids_test:
    #     temp = np.where(evids == unique_evid)[0]
    #     assert len(temp) > 0, 'shouldnt happen either'
    #     if (not rows_test is None):
    #         rows_test = np.concatenate((rows_test, temp), axis=0)
    #     else:
    #         rows_test = np.asarray(temp[:])

    #rows_train = np.sort(rows_train)
    #rows_test = np.sort(rows_test)
    # hoping this is a faster version of the commented out code above
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

def read_ps_meta_data(csv_file):
    """
    Reads the P and S meta data and puts it into a dataframe.
    """
    df = pd.read_csv(csv_file)
    return df
    #df = pd.read_csv(csv_file, header=None)
    ## e.g.,: 60000005,WY,YMR,HHE,HHN,HHZ,01,P,1349112442.957352,1,0
    #df.columns = ['evid', 'network', 'station', 'channel1', 'channel2',
    #              'channel3', 'location', 'phase', 'pick_time',
    #              'jiggle_weight', 'polarity']
    #return df

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
    n_comp = X.shape[2]
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
    # Allocate space
    print(start_sample, n_samples_in_window)
    t = np.zeros([n_samples_in_window, 3])
    T_index = np.zeros(n_obs*n_duplicate, dtype='int')
    X_new = np.zeros([n_obs*n_duplicate, n_samples_in_window, 3])
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
                # print("changing start_index")
                start_index = start_sample + lags[idup*n_obs+iobs]
            assert start_index >= 0, 'this should never happen'
            #start_index = max(0, start_index)
            end_index = start_index + n_samples_in_window
            T_index[idup*n_obs+iobs] = pick_window_index - lags[idup*n_obs+iobs] 
            #print(start_index, end_index, end_index - start_index)
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

def join_h5_files(h5_files, n_samples_in_window):
    X = None
    Y = None
    for h5_f in h5_files:
        print("Reading:", h5_f)
        h5_file = h5py.File(h5_f, 'r')
        print("file length", h5_file["X"].shape)

        file_x = np.asarray(h5_file["X"])
        file_y = np.asarray(h5_file['Y'])
        if file_x.shape[1] > n_samples_in_window:
            file_x = file_x[:, :n_samples_in_window, :]
            if len(file_y.shape) < 3:
                new_shape = file_y.shape + (1,)
                file_y = file_y.reshape(new_shape)
                # tmp=file_x
            file_y = file_y[:, :n_samples_in_window, :]

        print(file_x[0])

        if (X is None):
            X = file_x
            Y = file_y
        else:
            # print(np.where(tmp[0, :, 0] == file_x[0, 0, 0]))
            X = np.concatenate((X, file_x), axis=0)
            Y = np.concatenate((Y, file_y), axis=0)
        h5_file.close()
        print(X.shape)

    return X, Y

def combine(meta_csv_file,    # metadata for the arrivals
            h5_phase_file,    # name of archive with the picked waveforms 
            h5_noise_file,    # name of archive with the noise waveforms
            train_file_name,  # name of the training file
            validate_file_name, # name of validation file
            test_file_name,   # name of the test file
            validate_df_name, # name of the csv file with information on the validation rows
            test_df_name,     # name of the csv file with information on the test rows 
            train_frac = 0.8, # e.g., keep 80 pct of signals for training and 20 pct for validation and testing
            test_frac = 0.5,  # e.g., keep 50 pct of signals for validation and 50 pct of signals for testing
            noise_train_frac = 0.8, # e.g., keep 0.8 pct of noise for training and 0.2 pct for validation
            n_duplicate_train = 3, # randomly repeat waveforms in training dataset but randomize starting location
            target_shrinkage = None,  # make it so that we classifiy at this fraction of central points in the window
            train_df_name=None,  # I added this, so can have a metadata file for training data - probably another way to do this
            window_duration=10,
            dt=0.01,
            reduce_stead_noise=False,
            noise_meta_file=None,
            extract_events_params=None):
    """
    Combines the noise and observed phase waveforms then samples them into
    testing and validation test sets.
    """
    print("Loading signal waveforms...")
    if (type(h5_phase_file) is list):
        X = None
        Y = None
        for h5_phase_f in h5_phase_file:
            print("Reading:", h5_phase_f)
            h5_file = h5py.File(h5_phase_f, 'r')
            if (X is None):
                X = np.asarray(h5_file['X'])
                Y = np.asarray(h5_file['Y'])
            else:
                X = np.concatenate((X, np.asarray(h5_file['X'])), axis=0 )
                Y = np.concatenate((Y, np.asarray(h5_file['Y'])), axis=0 )
            h5_file.close()
            print(X.shape)
    else:
        print("Reading:", h5_phase_file)
        h5_file = h5py.File(h5_phase_file, 'r')
        X = np.asarray(h5_file['X'])
        Y = np.asarray(h5_file['Y'])
        h5_file.close()
    print("Loading metadata")

    if (type(meta_csv_file) is list):
        meta_df = None
        for meta_csv_f in meta_csv_file:
            if (meta_df is None):
                meta_df = read_ps_meta_data(meta_csv_f)
            else:
                meta_df = meta_df.append(read_ps_meta_data(meta_csv_f))
            print(meta_csv_f, meta_df.shape)
        #loop
    else:
        meta_df = read_ps_meta_data(meta_csv_file)

    if np.isin("source_id", meta_df.columns):
        column_names = np.copy(meta_df.columns.values)
        column_names[np.where(column_names == "source_id")[0][0]] = "evid"
        meta_df.columns = column_names

    if extract_events_params is not None:
        print("Extracting events...")
        # Start processing steps...
        extracted_event_meta, kept_event_meta = extract_events.separate_events(meta_df, extract_events_params["bounds"])
        print(X.shape, Y.shape)
        # Grab all the data not including the extracted events
        kept_event_X, kept_event_Y = extract_events.grab_from_h5files(X, Y, kept_event_meta)
        # Grab the data for the extracted events
        extracted_event_X, extracted_event_Y = extract_events.grab_from_h5files(X, Y, extracted_event_meta)
        # Just some sanity checks
        assert X.shape[0] - kept_event_X.shape[0] == extracted_event_X.shape[0]
        assert Y.shape[0] - kept_event_Y.shape[0] == extracted_event_Y.shape[0]
        assert extracted_event_X.shape[0] < kept_event_X.shape[0]
        assert extracted_event_Y.shape[0] < kept_event_Y.shape[0]

        print("Processing extracted events...")
        X_ext, Y_ext, T_ext = augment_data(extracted_event_X[:], extracted_event_Y[:],
                                                         1, window_duration, dt,
                                                         target_shrinkage=None, lrand=True, for_python=True,
                                                         meta_for_boxcar=extracted_event_meta)

        print("Saving extracted event data...")
        extract_events.write_h5file(X_ext, Y_ext, f'{extract_events_params["outfile_root"]}.h5', T=T_ext)
        extracted_event_meta.to_csv(f'{extract_events_params["outfile_root"]}.df.csv', index=False)

        print("Updating data...")
        X = kept_event_X
        Y = kept_event_Y
        meta_df = kept_event_meta
        print(X.shape, Y.shape)

    assert meta_df.shape[0] == X.shape[0], 'number of waveforms dont match csv'
    assert meta_df.shape[0] == Y.shape[0], 'number of responses dont match csv'

    print("Loading noise waveforms...")
    if (type(h5_noise_file) is list):
        print("Combining noise files")
        n_samples_in_window = compute_pad_length(window_duration, dt)
        X_noise, Y_noise = join_h5_files(h5_noise_file, n_samples_in_window)
    else:
        noise_h5_file = h5py.File(h5_noise_file, 'r')
        X_noise = np.asarray(noise_h5_file['X'])
        Y_noise = np.asarray(noise_h5_file['Y'])
        noise_h5_file.close()

    if noise_meta_file is not None:
        noise_meta_df = pd.read_csv(noise_meta_file)

    print("Input data shape:", X.shape, Y.shape)
    print("Input noise shape:", X_noise.shape, Y_noise.shape)
    print("Selecting noise rows...")

    if reduce_stead_noise:
        print("Filtering STEAD noise rows...")
        print("Original noise shape:", X_noise.shape)
        noise_rows = np.arange(0, len(X_noise), 3)

        if len(noise_rows) < len(meta_df):
            rows_to_sample = np.full(len(X_noise), 1)
            rows_to_sample[noise_rows] = 0
            noise_rows = np.append(noise_rows, np.random.choice(np.arange(0, len(X_noise))[np.where(rows_to_sample)], len(meta_df)-len(noise_rows)))
            noise_rows = np.sort(noise_rows)

        X_noise = X_noise[noise_rows, :, :]
        Y_noise = Y_noise[noise_rows]
        if noise_meta_file is not None:
            noise_meta_df = noise_meta_df.iloc[noise_rows]

        print("New noise shape:", X_noise.shape)

    n_noise = X_noise.shape[0]
    noise_train_rows, noise_test_rows = train_test_split(np.arange(0, n_noise-1, 1), 
                                                         train_size = noise_train_frac)

    noise_validate_rows = None
    if (test_frac > 0 and test_frac < 1):
        noise_validate_rows, noise_test_rows = train_test_split(noise_test_rows,
                                                                train_size = test_frac)
    else:
        if (test_frac == 1):
            noise_test_rows = np.copy(noise_test_rows)
        else:
            noise_validate_rows = np.copy(noise_test_rows) 

    print("Sampling rows...")
    train_rows, test_rows = sample_on_evid(meta_df, train_frac)

    validate_rows = None
    if (test_frac > 0 and test_frac < 1):
        validate_rows, test_rows = train_test_split(test_rows, train_size = test_frac) 
    else:
        if (test_frac == 1):
            test_rows = np.copy(test_rows)
        else:
            validate_rows = np.copy(test_rows)

    X_noise_train = np.zeros([0,0,0])
    Y_noise_train = np.zeros([0,0,0])
    T_noise_train = np.zeros([0])
    X_noise_validate = np.zeros([0,0,0])
    Y_noise_validate = np.zeros([0,0,0])
    T_noise_validate = np.zeros([0])
    X_noise_test = np.zeros([0,0,0])
    Y_noise_test = np.zeros([0,0,0])
    T_noise_test = np.zeros([0])

    X_train = np.zeros([0,0,0])
    Y_train = np.zeros([0,0,0])
    T_train = np.zeros([0])
    X_validate = np.zeros([0,0,0])
    Y_validate = np.zeros([0,0,0])
    T_validate = np.zeros([0])
    X_test = np.zeros([0,0,0])
    Y_test = np.zeros([0,0,0])
    T_test = np.zeros([0])


    print("Creating noise training dataset...")
    X_noise_train, Y_noise_train, T_noise_train \
        = augment_data(X_noise[noise_train_rows,:,:], Y_noise[noise_train_rows],
                       1, window_duration, dt,
                       target_shrinkage = target_shrinkage, lrand=False, for_python=True)
    T_noise_train[:] =-1 # Don't care
    if (noise_validate_rows.size > 0):
        print("Creating noise validation dataset...")
        X_noise_validate, Y_noise_validate, T_noise_validate \
            = augment_data(X_noise[noise_validate_rows,:,:], Y_noise[noise_validate_rows],
                           1, window_duration, dt, 
                           target_shrinkage = target_shrinkage, lrand=False, for_python=True)
        T_noise_validate[:] =-1 # Don't care
    if (noise_test_rows.size > 0):
        print("Creating test dataset...")
        X_noise_test, Y_noise_test, T_noise_test \
            = augment_data(X_noise[noise_test_rows,:,:], Y_noise[noise_test_rows],
                           1, window_duration, dt,
                           target_shrinkage = target_shrinkage, lrand=False, for_python=True)
        T_noise_test[:] =-1 # Don't care

    print("Creating signal training dataset...")
    X_train, Y_train, T_train \
        = augment_data(X[train_rows, :, :], Y[train_rows],
                       n_duplicate_train, window_duration, dt,
                       target_shrinkage = target_shrinkage, lrand=True, for_python=True, meta_for_boxcar=meta_df.iloc[train_rows])
    if (validate_rows is not None):
        print("Creating signal validation dataset...")
        X_validate, Y_validate, T_validate \
            = augment_data(X[validate_rows, :, :], Y[validate_rows],
                           1, window_duration, dt, 
                           target_shrinkage = target_shrinkage, lrand=True, for_python=True, meta_for_boxcar=meta_df.iloc[validate_rows])
    if (test_rows.size > 0):
        print("Creating signal test dataset...")
        X_test, Y_test, T_test \
            = augment_data(X[test_rows, :, :], Y[test_rows],
                           1, window_duration, dt,
                           target_shrinkage = target_shrinkage, lrand=True, for_python=True, meta_for_boxcar=meta_df.iloc[test_rows])

    print("Concatenating training dataset...")
    X_train = np.concatenate((X_train, X_noise_train), axis=0 )
    Y_train = np.concatenate((Y_train, Y_noise_train), axis=0 )
    T_train = np.concatenate((T_train, T_noise_train), axis=0 )
    print("Training shape:", X_train.shape)
    if (X_validate.size > 0):
        print("Concatenating validating dataset...")
        X_validate = np.concatenate((X_validate, X_noise_validate), axis=0 )
        Y_validate = np.concatenate((Y_validate, Y_noise_validate), axis=0 )
        T_validate = np.concatenate((T_validate, T_noise_validate), axis=0 )
        print("Validation shape:", X_validate.shape)
    print(X_test.shape, X_noise_test.shape)
    if (X_test.size > 0):
        print("Concatenating test dataset...")
        X_test = np.concatenate((X_test, X_noise_test), axis=0 )
        Y_test = np.concatenate((Y_test, Y_noise_test), axis=0 )
        T_test = np.concatenate((T_test, T_noise_test), axis=0 )
        print("Testing shape:", X_test.shape)

    # Write it
    print("Writing archives...")
    ofl = h5py.File(train_file_name, 'w')
    ofl.create_dataset("X", data=X_train)
    ofl.create_dataset("Y", data=Y_train)
    ofl.create_dataset("Pick_index", data=T_train)
    ofl.close()

    # I added this
    if train_df_name is not None:
        meta_df.iloc[train_rows].to_csv(train_df_name, index=False)

    if noise_meta_file is not None:
        output_pref = f'{"/".join(train_df_name.split("/")[:-1])}'
        noise_meta_df.iloc[noise_train_rows].to_csv(f'{output_pref}/noise_train.csv')
        noise_meta_df.iloc[noise_test_rows].to_csv(f'{output_pref}/noise_test.csv')
        noise_meta_df.iloc[noise_validate_rows].to_csv(f'{output_pref}/noise_validate.csv')

    if (X_validate.size > 0):
        ofl = h5py.File(validate_file_name, 'w')
        ofl.create_dataset("X", data=X_validate)
        ofl.create_dataset("Y", data=Y_validate)
        ofl.create_dataset("Pick_index", data=T_validate)
        ofl.close()
        meta_df.iloc[validate_rows].to_csv(validate_df_name, index=False)
    if (X_test.size > 0):
        ofl = h5py.File(test_file_name, 'w')
        ofl.create_dataset("X", data=X_test)
        ofl.create_dataset("Y", data=Y_test)
        ofl.create_dataset("Pick_index", data=T_test)
        ofl.close()
        meta_df.iloc[test_rows].to_csv(test_df_name, index=False)

if __name__ == "__main__":
    np.random.seed(49230)
    window_duration = 10.0
    n_duplicate_p_train = 1 #2
    n_duplicate_s_train = 2 #3
    dt = 0.01 # Sampling period (seconds)

    pref ='/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/'
    combined_pref = pref + 'magna2020_srl/combined_signal/'
    p_h5_file_name = combined_pref + 'allP.10s.h5'
    s_h5_file_name = combined_pref + 'allS.10s.h5'
    p_meta_csv_file = combined_pref + 'allPMetaData.df.csv'
    s_meta_csv_file = combined_pref + 'allSMetaData.df.csv'
    noise_h5_file_name = pref + 'yellowstone/allNoiseYellowstoneWaveforms.h5'

    p_train_python_file_name = '%sresampled/trainP.%ds.%ddup.h5'%(pref, int(window_duration), n_duplicate_p_train)
    p_validate_python_file_name = '%sresampled/validateP.%ds.%ddup.h5'%(pref, int(window_duration), n_duplicate_p_train)
    p_test_python_file_name = '%sresampled/testP.%ds.%ddup.h5'%(pref, int(window_duration), n_duplicate_p_train)
    p_validate_df_name = '%sresampled/validateP.%ds.%ddup.df.csv'%(pref, int(window_duration), n_duplicate_p_train)
    p_test_df_name = '%sresampled/testP.%ds.%ddup.df.csv'%(pref, int(window_duration), n_duplicate_p_train)
    # I added this
    p_train_df_name = '%sresampled/trainP.%ds.%ddup.df.csv'%(pref, int(window_duration), n_duplicate_p_train)

    s_train_python_file_name = '%sresampled/trainS.%ds.%ddup.h5'%(pref, int(window_duration), n_duplicate_s_train)
    s_validate_python_file_name = '%sresampled/validateS.%ds.%ddup.h5'%(pref, int(window_duration), n_duplicate_s_train)
    s_test_python_file_name = '%sresampled/testS.%ds.%ddup.h5'%(pref, int(window_duration), n_duplicate_s_train)
    s_validate_df_name = '%sresampled/validateS.%ds.%ddup.df.csv'%(pref, int(window_duration), n_duplicate_s_train)
    s_test_df_name = '%sresampled/testS.%ds.%ddup.df.csv'%(pref, int(window_duration), n_duplicate_s_train)
    # I added this
    s_train_df_name = '%sresampled/trainS.%ds.%ddup.df.csv'%(pref, int(window_duration), n_duplicate_s_train)

    train_frac = 0.8
    noise_train_frac = 0.8

    p_meta_csv_files = p_meta_csv_file
    h5_file_names = p_h5_file_name
    combine(p_meta_csv_files,
            h5_file_names,    # name of archive with the picked waveforms
            noise_h5_file_name,    # name of archive with the noise waveforms
            p_train_python_file_name,  # name of the training file
            p_validate_python_file_name, 
            p_test_python_file_name,   # name of the test file
            p_validate_df_name, # name of the csv file with information on the validation rows
            p_test_df_name,     # name of the csv file with information on the test rows 
            phase='P',        # phase label
            train_frac = 0.8, # e.g., keep 80 pct of signals for training and 0.2 pct for validation 
            test_frac = 1,    # e.g., keep 100 pct of remaining 20 pct of data for testing as opposed to validation
            noise_train_frac = 0.8, # e.g., keep 80 pct of noise for training and 0.2 pct for validation
            n_duplicate_train = n_duplicate_p_train, # randomly repeat waveforms in training dataset but randomize starting location
            train_df_name=p_train_df_name,
            window_duration=window_duration) # Name of csv file with information on training rows - I added this

    s_meta_csv_files = s_meta_csv_file
    h5_file_names = s_h5_file_name
    combine(s_meta_csv_files,  #s_meta_csv_file,
            h5_file_names, # s_h5_file_name,    # name of archive with the picked waveforms 
            noise_h5_file_name,    # name of archive with the noise waveforms
            s_train_python_file_name,  # name of the training file
            s_validate_python_file_name, 
            s_test_python_file_name,   # name of the test file
            s_validate_df_name, # name of the csv file with information on the validation rows
            s_test_df_name,     # name of the csv file with information on the test rows 
            phase='S',        # phase label
            train_frac = 0.8, # e.g., keep 80 pct of signals for training and 0.2 pct for validation 
            test_frac = 1,    # e.g., keep 100 pct of remaining 20 pct of data for testing as opposed to validation
            noise_train_frac = 0.8, # e.g., keep 80 pct of noise for training and 0.2 pct for validation
            n_duplicate_train = n_duplicate_s_train, # randomly repeat waveforms in training dataset but randomize starting location
            train_df_name=s_train_df_name) # Name of csv file with information on training rows - I added this
