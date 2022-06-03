import pandas as pd
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
import extract_events
import os

class SplitData():

   @staticmethod
   def make_directory(outfile_pref):
       dir = os.path.split(outfile_pref)[0]
       if not os.path.exists(dir):
           print(f"Making output directory {dir}")
           os.mkdir(dir)


   @staticmethod
   def get_matching_event_rows(evids, catalog_df):
      rows = np.zeros(len(catalog_df), dtype='int')
      catalog_evids = catalog_df['evid'].values
      catalog_rows  = catalog_df['qc_rows'].values
      n_found = 0
      for i in range(len(evids)):
         rows_temp = catalog_rows[catalog_evids == evids[i]]
         for row in rows_temp:
               rows[n_found] = row
               n_found = n_found + 1
      rows = rows[0:n_found]
      return rows

   def extract_events(self, input_catalog_df, input_catalog_h5, bounds):
       extracted_df, kept_df = extract_events.separate_events(input_catalog_df, bounds)
       # There are no NGB events for the given time period in blast or historical eq catalog
       X_extracted, y_extracted = extract_events.grab_from_h5files(input_catalog_h5["X"][:, :],
                                                               input_catalog_h5["Y"][:],
                                                               extracted_df)

       X_kept, y_kept= extract_events.grab_from_h5files(input_catalog_h5["X"][:, :],
                                                               input_catalog_h5["Y"][:],
                                                               kept_df)
       print("Kept", X_kept.shape, y_kept.shape)
       print("Extracted", X_extracted.shape, y_extracted.shape)

       return (X_kept, y_kept, kept_df), (X_extracted, y_extracted, extracted_df)

   def split_event_wise(self, catalog_df_in, catalog_h5,
                        train_size = 0.7,
                        validation_size = 0.1,
                        test_size = 0.2,
                        min_training_quality = -1, is_stead=False,
                        extract_event_bounds=None):
        """
        Splits the waveforms event-wise.  This prevents potential target leaking
        by preventing the neural network of gleaning potentially useful source
        information from the training set prior to application to the validation
        and test sets.

        Parameters
        ----------
        catalog_df : pd.dataframe
         Pandas dataframe containing the earthquake catalog metadata.
        catalog_h5 : h5py.File
         HDF5 archive with the waveform data.
        train_size : float
         The proportion of data to map to the training set - e.g., 0.7 is 70 pct.
        validation_size : float
         The proportion of data to map to the validation set which is used to
         determine which epoch to use.
        train_size : float
         The proportion of data to map to the training dataset.
        min_training_quality : float
         Defines the minimum quality to allow into the data and validation sets.
         -1 disables this.

        Returns
        -------
        X_train_h5 : np.ndarray
         The waveform data for the training dataset.
        y_train_h5 : np.array
         The waveform targets for the training.
        train_df : pd.dataframe
         Metadata for the training dataset.
        X_validation_h5 : np.ndarray
         The waveform data for the validation dataset.
        y_validation_h5 : np.array
         The waveform targets for the validation dataset.
        validation_df : pd.dataframe
         Metadata for the validation dataset.
        X_test_h5 : np.ndarray
         The waveform data for the test dataset.
        y_test_h5 : np.array
         The waveform targets for the test dataset.
        test_df : pd.dataframe
         Metadata for the test dataset.
        """

        # And the data

        if extract_event_bounds is not None:
            print("Original shape", catalog_df_in.shape)
            kept_data, extracted_data = self.extract_events(catalog_df_in, catalog_h5, extract_event_bounds)
            X, y, catalog_df_in = kept_data
        else:
            X = catalog_h5['X'][:]
            y = catalog_h5['Y'][:]

        if len(X.shape) == 2:
             X = np.expand_dims(X, axis=2)

        print("Data shape", X.shape, y.shape, catalog_df_in.shape)

        assert X.shape[0] == len(catalog_df_in) and y.shape[0] == len(catalog_df_in), "Data shapes do not match"

        if is_stead:
          cols = catalog_df_in.columns.to_numpy()
          cols[cols=="source_id"] = "evid"
          catalog_df_in.columns = cols
          print("Input df size", catalog_df_in.shape)
          # remove any events in UUSS region
          catalog_df_in = catalog_df_in[~((catalog_df_in['source_latitude'] < 42.5) &
                    (catalog_df_in['source_latitude'] > 36.75) &
                    (catalog_df_in['source_longitude'] < -108.75) &
                    (catalog_df_in['source_longitude'] > -114.25)) &
                  ~((catalog_df_in['source_latitude'] < 45.167) &
                    (catalog_df_in['source_latitude'] > 44) &
                    (catalog_df_in['source_longitude'] < -109.75) &
                    (catalog_df_in['source_longitude'] > -111.333))]
          print("New df size", catalog_df_in.shape)
          # And the data
          print("Original data shape", X.shape, y.shape)
          X = catalog_h5['X'][catalog_df_in.index.values, :, :]
          y = catalog_h5['Y'][catalog_df_in.index.values]
          print("New data shape", X.shape, y.shape)

        assert (train_size + validation_size + test_size - 1) < 1.e-4, 'train, validation test size must sum to 1'
        catalog_df = catalog_df_in.copy(deep = True)
        catalog_df['original_rows'] = np.arange(len(catalog_df))
        catalog_df['qc_rows'] = np.copy(np.arange(len(catalog_df)))
        if (min_training_quality > 0):
         catalog_df = catalog_df[catalog_df['pick_quality'] >= min_training_quality]
         catalog_df['qc_rows'] = np.copy(np.arange(len(catalog_df)))
        events = np.unique(catalog_df['evid'].values)
        train_events, test_events_work = train_test_split(events, train_size = train_size)
        validation_size_work =  validation_size/(validation_size + test_size)
        validation_events, test_events = train_test_split(test_events_work,
                                                         train_size = validation_size_work)
        assert len(validation_events) + len(test_events) + len(train_events) == len(events), 'failed to decompose sets'
        print("Number of events:", len(events))
        print("Number of events in training:", len(train_events))
        print("Number of events in validation:", len(validation_events))
        print("Number of events in testing:", len(test_events))
        train_rows      = self.get_matching_event_rows(train_events, catalog_df)
        validation_rows = self.get_matching_event_rows(validation_events, catalog_df)
        test_rows       = self.get_matching_event_rows(test_events,  catalog_df)
        assert len(train_rows) + len(validation_rows) + len(test_rows) == len(catalog_df), 'failed to find right rows'

        # Extract the rows from the dataframe
        validation_df = None
        test_df = None
        train_df = catalog_df.iloc[train_rows]
        if (len(validation_rows) > 0):
         validation_df = catalog_df.iloc[validation_rows]
        if (len(test_rows) > 0):
         test_df = catalog_df.iloc[test_rows]
        # # And the data
        # X = catalog_h5['X'][:,:,:]
        # y = catalog_h5['Y'][:]
        if (len(train_rows) > 0):
         train_archive_rows = catalog_df['original_rows'].iloc[train_rows]
         X_train_h5      = np.copy(X[train_archive_rows,:,:])
         y_train_h5      = np.copy(y[train_archive_rows])
        X_validation_h5 = None
        y_validation_h5 = None
        if (len(validation_rows) > 0):
         validation_archive_rows = catalog_df['original_rows'].iloc[validation_rows]
         X_validation_h5 = np.copy(X[validation_archive_rows,:,:])
         y_validation_h5 = np.copy(y[validation_archive_rows])
        X_test_h5 = None
        y_test_h5 = None
        n_split_test_rows = len(test_rows)
        if (len(test_rows) > 0):
         # Don't waste data - want to mine all pick residuals during testing
         test_df = catalog_df_in.copy(deep = True)
         test_df['original_rows'] = np.arange(len(test_df))
         test_df['qc_rows'] = np.copy(np.arange(len(test_df)))
         test_rows = self.get_matching_event_rows(test_events, test_df)
         test_df = test_df.iloc[test_rows]
         X_test_h5 = np.copy(X[test_rows,:,:])
         y_test_h5 = np.copy(y[test_rows])
        #print(train_df.describe())
        #print(validation_df.describe())
        #print(test_df.describe())
        assert len(train_df) == len(train_rows), 'failed to get training rows'
        assert len(validation_df) == len(validation_rows), 'failed to get validation rows'
        assert len(test_df) >= len(test_rows), 'failed to get testing rows'
        print("Number of events:", len(events))
        print("Number of training waveforms: %d (%f pct)"%(len(train_df), len(train_df)/len(catalog_df)*100.))
        print("Number of validation waveforms: %d (%f pct)"%(len(validation_df), len(validation_df)/len(catalog_df)*100.))
        print("Nominal number of test waveforms: %d (%f pct)"%(n_split_test_rows, n_split_test_rows/len(catalog_df)*100.))
        print("Actual number of available testing waveforms: %d (%f pct)"%(len(test_df), len(test_df)/len(catalog_df)*100.))

        if extract_event_bounds is not None:
            return X_train_h5, y_train_h5, train_df, \
                   X_validation_h5, y_validation_h5, validation_df, \
                   X_test_h5, y_test_h5, test_df, extracted_data

        return X_train_h5, y_train_h5, train_df, \
            X_validation_h5, y_validation_h5, validation_df, \
            X_test_h5, y_test_h5, test_df

   @staticmethod
   def write_data_and_dataframes(output_file_root,
                                 X_train_h5, y_train_h5, train_df,
                                 X_validation_h5, y_validation_h5, validation_df,
                                 X_test_h5, y_test_h5, test_df):
      """
      Writes the waveforms and dataframes characterizing the waveforms.

      Parameters
      ----------
      output_file_root : string
         The basename for the files to write to disk.  Appropriate identifying
         information will be appended to the file names to denote its intent
         (e.g., train, validation, test) and its file type (e.g., h5 or csv).
      X_train_h5 : np.ndarray
         The waveform data for the training dataset.
      y_train_h5 : np.array
         The waveform targets for the training.
      train_df : pd.dataframe
         Metadata for the training dataset.
      X_validation_h5 : np.ndarray
         The waveform data for the validation dataset.
      y_validation_h5 : np.array
         The waveform targets for the validation dataset.
      validation_df : pd.dataframe
         Metadata for the validation dataset. 
      X_test_h5 : np.ndarray
         The waveform data for the test dataset.
      y_test_h5 : np.array
         The waveform targets for the test dataset.
      test_df : pd.dataframe
         Metadata for the test dataset.
      """
      # Write data
      if (X_train_h5 is not None):
         print("Writing", len(train_df), "training examples")
         assert len(train_df) == X_train_h5.shape[0], 'Training size mismatch'
         assert len(train_df) == len(y_train_h5), 'Training target size mismatch'
         train_df.to_csv(output_file_root + '_train.csv', index=False)
         h5 = h5py.File(output_file_root + '_train.h5', 'w')
         h5['X'] = X_train_h5
         h5['Y'] = y_train_h5
         h5.close()

      if (X_validation_h5 is not None):
         print("Writing", len(validation_df), "validation examples")
         assert len(validation_df) == X_validation_h5.shape[0], 'Validation size mismatch'
         assert len(validation_df) == len(y_validation_h5), 'Validation target size mismatch'
         validation_df.to_csv(output_file_root + '_validation.csv', index=False)
         h5 = h5py.File(output_file_root + '_validation.h5', 'w')
         h5['X'] = X_validation_h5
         h5['Y'] = y_validation_h5
         h5.close()

      if (X_test_h5 is not None):
         print("Writing", len(test_df), "test examples")
         assert len(test_df) == X_test_h5.shape[0], 'Test size mismatch'
         assert len(test_df) == len(y_test_h5), 'Test target size mismatch'
         test_df.to_csv(output_file_root + '_test.csv', index=False)
         h5 = h5py.File(output_file_root + '_test.h5', 'w')
         h5['X'] = X_test_h5
         h5['Y'] = y_test_h5
         h5.close()

   def split_event_wise_and_write(self, catalog_df, catalog_h5,
                                 output_file_root = 'data/uuss',
                                 train_size = 0.7,
                                 validation_size = 0.1,
                                 test_size = 0.2,
                                 min_training_quality = -1,
                                  is_stead=False):
      """
      Splits the waveforms event-wise.  This prevents potential target leaking
      by preventing the neural network of gleaning potentially useful source
      information from the training set prior to application to the validation
      and test sets.

      Parameters
      ----------
      catalog_df : pd.dataframe
         Pandas dataframe containing the earthquake catalog metadata.
      catalog_h5 : h5py.File
         HDF5 archive with the waveform data.
      output_file_root : string
         The basename for the files to write to disk.  Appropriate identifying 
         information will be appended to the file names to denote its intent
         (e.g., train, validation, test) and its file type (e.g., h5 or csv).
      train_size : float
         The proportion of data to map to the training set - e.g., 0.7 is 70 pct.
      validation_size : float
         The proportion of data to map to the validation set which is used to
         determine which epoch to use.
      train_size : float
         The proportion of data to map to the training dataset.
      min_training_quality : float
         Defines the minimum quality to allow into the data and validation sets.
         -1 disables this.
      """
      # Split the data
      X_train_h5, y_train_h5, train_df, \
      X_validation_h5, y_validation_h5, validation_df, \
      X_test_h5, y_test_h5, test_df \
         = self.split_event_wise(catalog_df, catalog_h5,
                              train_size, validation_size, test_size,
                              min_training_quality, is_stead=is_stead)
      # Write the data
      self.write_data_and_dataframes(output_file_root,
                                 X_train_h5, y_train_h5, train_df,
                                 X_validation_h5, y_validation_h5, validation_df,
                                 X_test_h5, y_test_h5, test_df)

class FirstMotionHelper():

   @staticmethod
   def randomly_flip(X, y, df, proportion = 0.5):
    """
    Randomly flips the specified proportion of polarities.
    """
    assert proportion >= 0 and proportion <= 1, 'proportion to flip must be between 0 and 1'
    n = len(X[:,0])
    flippable_rows = np.arange(0, len(y))[ np.where(y != 0)[0] ]
    n_flip = int(proportion*len(flippable_rows)) 
    if (n_flip == 0): 
        return X, y, df
    rows_to_flip = np.random.choice(flippable_rows, size = n_flip, replace = False)
    assert len(np.unique(rows_to_flip)) == len(rows_to_flip), 'cannot flip same row twice'
    rows_to_flip = np.sort(rows_to_flip)
    was_flipped = np.zeros(len(y), dtype = 'int')
    for i in range(len(rows_to_flip)):
        row = rows_to_flip[i]
        X[row,:] = -1*X[row,:]
        y[row] = -1*y[row]
        assert y[row] != 0, 'cant flip 0 row' 
        was_flipped[row] = 1
    df['was_flipped'] = was_flipped
    print("Proportion flipped:", len(rows_to_flip)/np.sum( (y != 0) )*100, "pct" )
    return X, y, df

   @staticmethod
   def upsample_minority_class(X, y, df):
    """
    Upsamples the minority class examples in the training dataset.

    Parameters
    ----------
    X : np.matrix
       The matrix containing waveform training examples.
    y : np.array
       The target classification for each waveform.
    df : pd.DataFrame
       The metadata characterizing the training examples.

    Returns
    -------
    X : np.matrix
       The upsampled waveform examples.
    y : np.array
       The upsampled targets.
    df : pd.DataFrame
       The upsample metadata.
    """
    unique_polarities = np.unique(df['first_motion'].values)
    imax = unique_polarities[0]
    nmax = np.sum(df['first_motion'] == unique_polarities[0])
    print("Polarity: %d has %d examples"%(unique_polarities[0], nmax))
    for i in range(1, len(unique_polarities)):
        nmax_work = np.sum(df['first_motion'] == unique_polarities[i])
        if (nmax_work > nmax):
            imax = unique_polarities[i]
            nmax = nmax_work
        print("Polarity: %d has %d examples"%(unique_polarities[i], nmax_work))
    # upsample minority classes
    rows_resample = np.zeros(0, dtype='int')
    for i in range(len(unique_polarities)):
        row_numbers = np.arange(len(df))
        rows = row_numbers[df['first_motion'] == unique_polarities[i]]
        n_new = nmax - len(rows)
        rows_resample = np.append(rows_resample, rows)
        if (n_new > 0):
            extra_rows = np.random.choice(rows, size = n_new, replace = True)
            rows_resample = np.append(rows_resample, extra_rows)
    # Randomly reorder the rows
    print("Original dataframe length:", len(df))
    np.random.shuffle(rows_resample)
    #temp_df = temp_df.sample(frac=1).reset_index(drop=True)
    print(rows_resample)
    df = df.iloc[rows_resample]
    X = X[rows_resample,:]
    y = y[rows_resample]
    print("New dataframe length:", len(df))
    assert len(X[:,0]) == len(y), 'upsample wrong'
    assert len(df) == len(y), 'upsample wrong'
    return X, y, df

   @staticmethod
   def upsample_down(X, y, df):
      """
      Randomly selects rows from the down first motion class and repeats them
      so that the number of down observations matches the number of up
      observations.  Note, during training we will randomly shift the trace
      start time so we aren't providing the exact training example twice.

      Parameters
      ----------
      X : np.matrix
         Waveforms in original (non-upsampled) dataset
      y : np.array
         Labels in original (non-upsampled) dataset
      df : pd.DataFrame
         Original (non-upsampled) dataframe
   
      Returns
      -------
      X : np.matrix
         Waveforms with additional randomly resampled down waveforms
      y : np.matrix
         Labels with additional randomly resampled down labels
      df : pd.DataFrame
         Dataframe with randomly resampled down labels
      """
      n_up = np.sum( (y == 1)* 1)
      n_down = np.sum( (y ==-1)*1 )
      n_diff = n_up - n_down
      if (n_diff <= 0):
         print("No reason to upsample down class - it isn't the minority class")
         return X, y, df
      down_rows = np.arange(0, len(y))[ np.where(y ==-1)[0] ]
      down_rows = np.random.choice(down_rows, size = n_diff, replace = False)   
      X_down = X[down_rows,:]
      y_down = y[down_rows]
      df_down = df.iloc[down_rows]

      X = np.concatenate([X, X_down])
      y = np.concatenate([y, y_down])
      df = pd.concat([df, df_down])
      return X, y, df