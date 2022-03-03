import sys
sys.path.append('..')
from data_processing import split_data

#np.random.seed(49230)
window_duration = 10.0
n_duplicate_s_train = 1
dt = 0.01  # Sampling period (seconds)
train_frac = 0.8
noise_train_frac = 0.8
test_frac = 0.5

pref = '/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/sDetector'
h5_file_name = pref + '/sSteadDataset.h5'
meta_csv_file = pref + '/sSteadMetadata.csv'
noise_h5_file_name = pref+'/sSteadNoise.h5'
noise_meta_file = f'{pref}/sSteadNoise.csv'

outpref = pref+"/stead_resampled_10s"
train_python_file_name = '%s/trainS.%ds.%ddup.h5' % (outpref, int(window_duration), n_duplicate_s_train)
validate_python_file_name = '%s/validateS.%ds.%ddup.h5' % (outpref, int(window_duration), n_duplicate_s_train)
test_python_file_name = '%s/testS.%ds.%ddup.h5' % (outpref, int(window_duration), n_duplicate_s_train)
validate_df_name = '%s/validateS.%ds.%ddup.df.csv' % (outpref, int(window_duration), n_duplicate_s_train)
test_df_name = '%s/testS.%ds.%ddup.df.csv' % (outpref, int(window_duration), n_duplicate_s_train)
train_df_name = '%s/trainS.%ds.%ddup.df.csv' % (outpref, int(window_duration), n_duplicate_s_train)


split_data.combine(meta_csv_file,
                   h5_file_name,  # name of archive with the picked waveforms
                   noise_h5_file_name,  # name of archive with the noise waveforms
                   train_python_file_name,  # name of the training file
                   validate_python_file_name,
                   test_python_file_name,  # name of the test file
                   validate_df_name,  # name of the csv file with information on the validation rows
                   test_df_name,  # name of the csv file with information on the test rows
                   train_frac=train_frac,  # e.g., keep 80 pct of signals for training and 0.2 pct for validation
                   test_frac=test_frac,  # e.g., keep 100 pct of remaining 20 pct of data for testing as opposed to validation
                   noise_train_frac=noise_train_frac,  # e.g., keep 80 pct of noise for training and 0.2 pct for validation
                   n_duplicate_train=n_duplicate_s_train,
                   # randomly repeat waveforms in training dataset but randomize starting location
                   train_df_name=train_df_name,  # Name of csv file with information on training rows - I added this,
                   window_duration=window_duration,
                   dt=dt,
                   reduce_stead_noise=True,
                   noise_meta_file=noise_meta_file)