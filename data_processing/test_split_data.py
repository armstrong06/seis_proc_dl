from data_processing.split_data_detectors import SplitDetectorData

window_duration = 10.0
n_duplicate_train = 2
dt = 0.01  # Sampling period (seconds)
train_frac = 0.8
noise_train_frac = 0.8
test_frac = 0.5
max_pick_shift = 250

pref = '/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/stead/small_tests'
h5_filename = f'{pref}/pStead_1750_300examples.h5'
meta_file = f'{pref}/pStead_1750_300examples.csv'
noise_h5_filename = f'{pref}/noiseStead_2000_300examples.h5'
noise_meta_file = f'{pref}/noiseStead_2000_300examples.csv'

outpref = f"{pref}/p_resampled_10s/"

# For NGB events - Don't need these for STEAD data
# Remove events within these bounds
# lat_min = 44.680
# lat_max = 44.870
# lon_max = 110.850
# lon_min = 110.640
# date_min = 130901  # YYMMDD
# date_max = 140630  # YYMMDD
# ################ End Set Params ####################
#
# assert lon_min < lon_max, "lon min must be less than max"
# assert lat_min < lat_max, "lat min must be less than max"
# assert date_min < date_max, "date min must be less than max"
#
# bounds = {"lat_min": lat_min,
#           "lat_max": lat_max,
#           "lon_max": lon_max,
#           "lon_min": lon_min,
#           "date_min": date_min,
#           "date_max": date_max}
#
#extract_events_params = {"bounds":bounds, "name":"NGB"}

spliter = SplitDetectorData(window_duration, dt, max_pick_shift, n_duplicate_train, outpref, pick_sample=750)
spliter.load_signal_data(h5_filename, meta_file)
spliter.load_noise_data(noise_h5_filename, noise_meta_file)
spliter.split_signal(train_frac, test_frac, extract_events_params=None)
spliter.process_signal()
spliter.split_noise(noise_train_frac, test_frac, reduce_stead_noise=True)
spliter.process_noise()
spliter.write_combined_datasets()
