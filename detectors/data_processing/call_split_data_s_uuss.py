import sys
sys.path.append('..')
from data_processing import split_data
import os

#np.random.seed(49230)
window_duration = 10.0
n_duplicate_s_train = 2
dt = 0.01  # Sampling period (seconds)
train_frac = 0.8
noise_train_frac = 0.8
test_frac = 0.5

pref = '/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data'
wf_pref = f'{pref}/waveformArchive/sDetector'
h5_file_name = wf_pref + '/uuss_current_earthquake_catalog_3C.h5'
meta_csv_file = wf_pref + '/uuss_current_earthquake_catalog_3C.csv'
noise_h5_file_name = [f'{pref}/yellowstone/allNoiseYellowstoneWaveforms.h5',
                      f'{pref}/magna2020_srl/combined_signal/allnoiseS.10s.2dup.h5']

outpref = f'{wf_pref}/uuss_resampled_10s'
train_python_file_name = '%s/trainS.%ds.%ddup.h5' % (outpref, int(window_duration), n_duplicate_s_train)
validate_python_file_name = '%s/validateS.%ds.%ddup.h5' % (outpref, int(window_duration), n_duplicate_s_train)
test_python_file_name = '%s/testS.%ds.%ddup.h5' % (outpref, int(window_duration), n_duplicate_s_train)
validate_df_name = '%s/validateS.%ds.%ddup.df.csv' % (outpref, int(window_duration), n_duplicate_s_train)
test_df_name = '%s/testS.%ds.%ddup.df.csv' % (outpref, int(window_duration), n_duplicate_s_train)
train_df_name = '%s/trainS.%ds.%ddup.df.csv' % (outpref, int(window_duration), n_duplicate_s_train)

# For NGB events
NGB_outfile_root = '%s/ngbS.%ds.%ddup' % (outpref, int(window_duration), n_duplicate_s_train)
# Remove events within these bounds
lat_min = 44.680
lat_max = 44.870
lon_max = 110.850
lon_min = 110.640
date_min = 130901  # YYMMDD
date_max = 140630  # YYMMDD
################ End Set Params ####################

assert lon_min < lon_max, "lon min must be less than max"
assert lat_min < lat_max, "lat min must be less than max"
assert date_min < date_max, "date min must be less than max"

bounds = {"lat_min": lat_min,
          "lat_max": lat_max,
          "lon_max": lon_max,
          "lon_min": lon_min,
          "date_min": date_min,
          "date_max": date_max}

extract_events_params = {"bounds":bounds, "outfile_root":NGB_outfile_root}

if not os.path.exists(outpref):
    os.makedirs(outpref)

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
                   noise_meta_file=None,
                   extract_events_params=extract_events_params)