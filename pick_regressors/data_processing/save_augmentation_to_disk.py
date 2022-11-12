import sys
sys.path.insert(0, "/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis-proc-dl")
from pick_regressors.process_pick_data import randomize_start_times_and_normalize
import h5py
import os
import numpy as np

time_series_len=600
max_dt=0.7
dt=0.01
n_duplicate=1
random_seed=82323
pref = "/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/sPicker/s_resampled_picker_locationgrid"
split_types = ["LocationGrid"]

for split_type in split_types:
    input_file = f"{pref}/uuss_{split_type}.h5"
    output_file = f"{pref}/augmented/uuss_{split_type}_{int(time_series_len*dt)}s_{n_duplicate}dup.h5"

    print("Starting on split", split_type)
    print(f"time_series_len: {time_series_len}, max_dt: {max_dt}, dt: {dt}, n_duplicate: {n_duplicate}, random_seed: {random_seed}")
    print(f"input file: {input_file}")

    dir = os.path.split(output_file)[0]
    if not os.path.exists(dir):
        print(f"Making output directory {dir}")
        os.mkdir(dir)

    with h5py.File(input_file, "r") as f:
        print(f.keys())
        X = f['X'][:]

    print(X.shape)

    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=2)

    X_new, Y = randomize_start_times_and_normalize(X, time_series_len=time_series_len,
                                                                   max_dt=max_dt, dt=dt, n_duplicate=n_duplicate,
                                                                    random_seed=random_seed)

    print("Finished augmenting, writing to:", output_file)
    print(X_new.shape, Y.shape)
    with h5py.File(output_file, "w") as f:
        f.create_dataset(data=X_new, name="X")
        f.create_dataset(data=Y, name="Y")
        print(f["X"][:].shape, f["Y"][:].shape)
