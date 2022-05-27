import sys
sys.path.append("/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis-proc-dl/")
from data_processing.split_data_detectors import SplitDetectorData

window_duration = 10.0
n_duplicate_train = 1
dt = 0.01  # Sampling period (seconds)
train_frac = 0.8
noise_train_frac = 0.8
test_frac = 0.5
max_pick_shift = 250
pick_sample = None

pref = '/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive'
h5_filename = f'{pref}/stead_detectors/SStead_2000.h5'
meta_file = f'{pref}/stead_detectors/SStead_2000.csv'
noise_h5_filename = f'{pref}/stead_detectors/noiseStead_2000.h5'
noise_meta_file = f'{pref}/stead_detectors/noiseStead_2000.csv'

outpref = f"{pref}/stead_detectors/s_resampled_{int(window_duration)}s/S."

# Initialize
spliter = SplitDetectorData(window_duration, dt, max_pick_shift, n_duplicate_train, outpref, pick_sample=pick_sample)
# Handle the signal
spliter.load_signal_data(h5_filename, meta_file)
spliter.split_signal(train_frac, test_frac, extract_events_params=None)
spliter.process_signal()
# Handle the noise
spliter.load_noise_data(noise_h5_filename, noise_meta_file)
spliter.split_noise(noise_train_frac, test_frac, reduce_stead_noise=True)
spliter.process_noise()
# Combine and write
spliter.write_combined_datasets()
