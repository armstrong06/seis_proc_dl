import h5py
import glob
import os

pref = "/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive"
data_dir = f"{pref}/noise_ENZ"

z_ind = 2

for file in glob.glob(f"{data_dir}/allNoise*h5"):
    print(file)
    h5f = h5py.File(file, "r")
    X = h5f["X"][:]
    Y = h5f["Y"][:]
    h5f.close()
    print(X.shape, Y.shape)

    new_file = f"{data_dir}/verticalComp.{os.path.split(file)[-1]}"
    print(new_file)
    h5f = h5py.File(new_file, "w")
    h5f.create_dataset("X", data=X[:, :, z_ind:z_ind+1])
    h5f.create_dataset("Y", data=Y)
    print(h5f["X"].shape, h5f["Y"].shape)
    h5f.close()