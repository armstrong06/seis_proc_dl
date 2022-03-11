import sys
sys.path.append('..')
from data_processing import extract_events
from detectors.data_processing import make_yboxcar as boxcar
import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
np.random.seed(49230)

class Split_Detector_Data():

    def __init__(self, window_duration, dt, max_pick_shift):
        self.window_duration = window_duration
        self.dt = dt
        self.max_pick_shift = max_pick_shift

    @staticmethod
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

    def compute_pad_length(self):
        n_samples = int(self.window_duration/self.dt + 0.5) + 1
        while (n_samples%16 != 0):
            n_samples = n_samples + 1
        return n_samples

    def augment_data(self, X, Y, n_duplicate,
                    target_shrinkage = None,
                    lrand = True, for_python=True, meta_for_boxcar=None):
        # Determine the sizes of the data
        print("initial", X.shape, Y.shape)
        n_obs = X.shape[0]
        n_samples = X.shape[1]
        n_comps = X.shape[2]
        # Compute a safe window size for the network architecture
        n_samples_in_window = self.compute_pad_length()
        if (target_shrinkage is None):
            n_samples_in_target = n_samples_in_window
            start_y = 0
            end_y = 0
        else:
            # TODO: Figure out what target_shrinkage does
            assert target_shrinkage > 0 and target_shrinkage < 1, 'target shrinkage must be in range (0,1)'
            n_samples_in_target = int(target_shrinkage*n_samples_in_window)
            start_y = (n_samples_in_window - n_samples_in_target)//2 
            end_y = n_samples_in_window - n_samples_in_target - start_y
        
        # Define the lags
        #max_lag = int(0.9*n_samples_in_window/2) # TODO: May want to multiply by like 0.95 - worst case is starting half-way into pick window
        
        # If max_pick_shift is a less than 1.0 - use it as a fraction of 1/2 the window length. Otherwise just use the max_lag value
        max_lag = self.max_pick_shift
        if self.max_pick_shift <= 1.0:
            max_lag = int(self.max_pick_shift*n_samples_in_window/2)

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
                if (for_python):
                    # Noise and signal Y had slightly different formats for some reason. Noise does not need 0 index,
                    # signal does
                    #try:
                    Y_new[idup*n_obs+iobs, :, 0] = Y[iobs, start_index+start_y:end_index-end_y]
                    # except:
                    #     try:
                    #         Y_new[idup * n_obs + iobs, :, 0] = Y[iobs, start_index + start_y:end_index - end_y, 0]
                    #     except:
                    #         assert Exception("this didn't work either")
                else:
                    Y_new[idup*n_obs+iobs, :] = Y[iobs, start_index+start_y:end_index-end_y]

        # TODO: I dont know if this will work correct when there are duplicates 
        if meta_for_boxcar is not None:
            Y_new = boxcar.add_boxcar(meta_for_boxcar, {0: 21, 1: 31, 2: 51}, X_new, Y, T_index)

        print("final", X_new.shape, Y_new.shape, T_index.shape)
        return (X_new, Y_new, T_index)

    @staticmethod
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

    def extract_events(self, extract_events_params, meta_df, X, Y):
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
            X_ext, Y_ext, T_ext = self.augment_data(extracted_event_X[:], extracted_event_Y[:], 1,
                                                            target_shrinkage=None, lrand=True, for_python=True,
                                                            meta_for_boxcar=extracted_event_meta)

            print("Saving extracted event data...")
            extract_events.write_h5file(X_ext, Y_ext, f'{extract_events_params["outfile_root"]}.h5', T=T_ext)
            extracted_event_meta.to_csv(f'{extract_events_params["outfile_root"]}.df.csv', index=False)

            return kept_event_X, kept_event_Y, kept_event_meta

    def load_waveform_data(self, h5_file, meta_csv_file, n_samples_in_window=1e6):
        if (type(h5_file) is list):
            self.join_h5_files(h5_file, n_samples_in_window) # put large number in for n_samples_in_window to keep them all
        else:
            print("Reading:", h5_file)
            h5_file = h5py.File(h5_file, 'r')
            X = np.asarray(h5_file['X'])
            Y = np.asarray(h5_file['Y'])
            h5_file.close()

        if meta_csv_file is None:
            meta_df = None
        elif (type(meta_csv_file) is list):
            meta_df = None
            for meta_csv in meta_csv_file:
                if (meta_df is None):
                    meta_df = pd.read_csv(meta_csv)
                else:
                    meta_df = meta_df.append(pd.read_csv(meta_csv))
                print(meta_csv, meta_df.shape)
        else:
            meta_df = pd.read_csv(meta_csv_file)

        return X, Y, meta_df
    
    def reduce_stead_noise_dataset(self, noise_meta_df, X_noise, Y_noise):
        print("Filtering STEAD noise rows...")
        print("Original noise shape:", X_noise.shape)
        noise_rows = np.arange(0, len(X_noise), 3)

        if len(noise_rows) < self.n_signal_waveforms:
            rows_to_sample = np.full(len(X_noise), 1)
            rows_to_sample[noise_rows] = 0
            noise_rows = np.append(noise_rows, np.random.choice(np.arange(0, len(X_noise))[np.where(rows_to_sample)], self.n_signal_waveforms-len(noise_rows)))
            noise_rows = np.sort(noise_rows)

        X_noise = X_noise[noise_rows, :, :]
        Y_noise = Y_noise[noise_rows]
        if noise_meta_df is not None:
            noise_meta_df = noise_meta_df.iloc[noise_rows]

        print("New noise shape:", X_noise.shape)

        return X_noise, Y_noise, noise_meta_df

    def set_n_signal_waveforms(self, n_signal_waveforms):
        self.n_signal_waveforms = n_signal_waveforms

    def process_signal(self, meta_csv_file,    # metadata for the arrivals
                h5_phase_file,    # name of archive with the picked waveforms 
                train_frac = 0.8, # e.g., keep 80 pct of signals for training and 20 pct for validation and testing
                test_frac = 0.5,  # e.g., keep 50 pct of signals for validation and 50 pct of signals for testing
                n_duplicate_train = 3, # randomly repeat waveforms in training dataset but randomize starting location
                target_shrinkage = None,  # make it so that we classifiy at this fraction of central points in the window
                extract_events_params=None):
        """
        Combines the noise and observed phase waveforms then samples them into
        testing and validation test sets.
        """
        print("Loading signal waveforms...")
        X, Y, meta_df = self.load_waveform_data(h5_phase_file, meta_csv_file)

        # Rename columns if metadata is from STEAD
        if np.isin("source_id", meta_df.columns):
            column_names = np.copy(meta_df.columns.values)
            column_names[np.where(column_names == "source_id")[0][0]] = "evid"
            meta_df.columns = column_names

        # Remove certain events from datasets if necessary
        if extract_events_params is not None:
            X, Y, meta_df = self.extract_events(extract_events_params)

        assert meta_df.shape[0] == X.shape[0], 'number of waveforms dont match csv'
        assert meta_df.shape[0] == Y.shape[0], 'number of responses dont match csv'
        self.set_n_signal_waveforms(len(meta_df))

        print("Input data shape:", X.shape, Y.shape)

        print("Sampling signal rows...")
        train_rows, test_rows = self.sample_on_evid(meta_df, train_frac)

        validate_rows = None
        if (test_frac > 0 and test_frac < 1):
            validate_rows, test_rows = train_test_split(test_rows, train_size = test_frac) 
        else:
            if (test_frac == 1):
                test_rows = np.copy(test_rows)
            else:
                validate_rows = np.copy(test_rows)

        print("Creating signal training dataset...")
        train_splits \
            = self.augment_data(X[train_rows, :, :], Y[train_rows], n_duplicate_train,
                        target_shrinkage = target_shrinkage, lrand=True, for_python=True, meta_for_boxcar=meta_df.iloc[train_rows])

        meta_df.iloc[train_rows].to_csv(train_df_name, index=False)

        validate_splits, test_splits = None, None
        if (validate_rows is not None):
            print("Creating signal validation dataset...")
            validate_splits \
                = self.augment_data(X[validate_rows, :, :], Y[validate_rows], 1,
                            target_shrinkage = target_shrinkage, lrand=True, for_python=True, meta_for_boxcar=meta_df.iloc[validate_rows])
            meta_df.iloc[validate_rows].to_csv(validate_df_name, index=False)

        if (test_rows.size > 0):
            print("Creating signal test dataset...")
            test_splits \
                = self.augment_data(X[test_rows, :, :], Y[test_rows], 1, 
                            target_shrinkage = target_shrinkage, lrand=True, for_python=True, meta_for_boxcar=meta_df.iloc[test_rows])
            meta_df.iloc[test_rows].to_csv(test_df_name, index=False)

        return train_splits, test_splits, validate_splits


    def process_noise(self, h5_noise_file, 
                noise_meta_file = None, 
                noise_train_frac = 0.8, # e.g., keep 0.8 pct of noise for training and 0.2 pct for validation, 
                test_frac = 0.5,  # e.g., keep 50 pct of signals for validation and 50 pct of signals for testing,
                reduce_stead_noise=False, 
                target_shrinkage=None):

        print("Loading noise waveforms...")
        n_samples_in_window = self.compute_pad_length()
        X_noise, Y_noise, noise_meta_df = self.load_waveform_data(h5_noise_file, noise_meta_file, n_samples_in_window)

        print("Input noise shape:", X_noise.shape, Y_noise.shape)

        # TODO: Check that this works
        if reduce_stead_noise:
            X_noise, Y_noise, noise_meta_df = self.reduce_stead_noise_dataset(noise_meta_df, X_noise, Y_noise)
        print("Samping noise rows...")
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

        print("Creating noise training dataset...")
        noise_train_splits \
            = self.augment_data(X_noise[noise_train_rows,:,:], Y_noise[noise_train_rows],1,
                        target_shrinkage = target_shrinkage, lrand=False, for_python=True)
        noise_train_splits[-1][:] =-1 # Don't care

        noise_validate_splits, noise_test_splits = None, None
        if (noise_validate_rows.size > 0):
            print("Creating noise validation dataset...")
            noise_validate_splits \
                = self.augment_data(X_noise[noise_validate_rows,:,:], Y_noise[noise_validate_rows], 1,
                            target_shrinkage = target_shrinkage, lrand=False, for_python=True)
            noise_validate_splits[-1][:] =-1 # Don't care

        if (noise_test_rows.size > 0):
            print("Creating noise test dataset...")
            noise_test_splits \
                = self.augment_data(X_noise[noise_test_rows,:,:], Y_noise[noise_test_rows], 1,
                            target_shrinkage = target_shrinkage, lrand=False, for_python=True)
            noise_test_splits[-1][:] =-1 # Don't care

        if noise_meta_file is not None:
            noise_meta_df.iloc[noise_train_rows].to_csv(f'{output_pref}/noise_train.csv')
            if (noise_validate_rows.size > 0):
                noise_meta_df.iloc[noise_validate_rows].to_csv(f'{output_pref}/noise_validate.csv')
            if (noise_test_rows.size > 0):
                noise_meta_df.iloc[noise_test_rows].to_csv(f'{output_pref}/noise_test.csv')

        return noise_train_splits, noise_test_splits, noise_validate_splits

    @staticmethod
    def combine_signal_noise(signal_tuple, noise_tuple):
        combined = []
        for ind in range(3):
            combined.append(np.concatenate((signal_tuple[ind], noise_tuple[ind]), axis=0 ))
        return tuple(combined)

    def write_dataset(self, signal_tuple, noise_tuple):
        