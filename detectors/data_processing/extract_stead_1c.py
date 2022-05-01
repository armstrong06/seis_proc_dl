import h5py
import glob
import os

pref = "/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/stead"
data_dir = f"{pref}/p_resampled_10s"
out_dir = f"{pref}/p_1c_resampled_10s"

z_ind = 2

for file in glob.glob(f"{data_dir}/*h5"):
    h5f = h5py.File(file, "r")
    X = h5f["X"][:]
    Y = h5f["Y"][:]
    T = h5f["Pick_index"][:]
    h5f.close()
    print(X.shape, Y.shape, T.shape)

    new_file = f"{out_dir}/oneComp.{os.path.split(file)[-1]}"
    h5f = h5py.File(new_file, "w")
    h5f.create_dataset("X", data=X[:, :, z_ind:z_ind+1])
    h5f.create_dataset("Y", data=Y)
    h5f.create_dataset("Pick_index", data=T)
    print(h5f["X"].shape, h5f["Y"].shape, h5f["Pick_index"].shape)
    h5f.close()