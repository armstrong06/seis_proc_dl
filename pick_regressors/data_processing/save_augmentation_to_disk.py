import sys
sys.path.insert(0, "/home/armstrong/Research/git_repos/seis-proc-dl")
from pick_regressors.process_pick_data import randomize_start_times_and_normalize
import h5py
import os
import numpy as np

time_series_len=400
max_dt=0.5
dt=0.01
n_duplicate=2
random_seed=82323
pref = "/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/scsn_stead_pickers"
input_file = f"{pref}/scsn_p_2000_2017_6sec_0.5r_pick_train.hdf5"
output_file = f"{pref}/p_picker_resampled/scsn_train_4s_2dup.h5"

dir = os.path.split(output_file)[0]
if not os.path.exists(dir):
    print(f"Making output directory {dir}")
    os.mkdir(dir)

with h5py.File(input_file, "r") as f:
    print(f.keys())
    X = f['X'][:]

if len(X.shape) == 2:
    X = np.expand_dims(X, axis=2)

X_new, Y = randomize_start_times_and_normalize(X, time_series_len=time_series_len,
                                                               max_dt=max_dt, dt=dt, n_duplicate=n_duplicate,
                                                                random_seed=random_seed)

print("Finished augmenting, writing to:", output_file)
with h5py.File(output_file, "w") as f:
    f.create_dataset(data=X, name="X")
    f.create_dataset(data=Y, name="Y")