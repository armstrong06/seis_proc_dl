#%%
import h5py
import numpy as np

pref = "/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive"
# noise_file = f"{pref}/old_data/noise_NEZ/allNoiseMagnaWaveforms.P.10s.h5"
# out_file = f"{pref}/noise_ENZ/allNoiseMagnaWaveformsPermuted.P.10s.h5"
noise_file = f"{pref}/old_data/noise_NEZ/allNoiseYellowstoneWaveforms.h5"
out_file = f"{pref}/noise_ENZ/allNoiseYellowstoneWaveformsPermuted.h5"

with h5py.File(noise_file, "r") as f:
    X = f["X"][:]
    Y = f["Y"][:]

n_examples = X.shape[0]
avg_z_noise = 0
avg_n_noise = 0
avg_e_noise = 0
for i in range(n_examples):
    # Input is (n,e,z)
    n = np.copy(X[i,:,0])
    e = np.copy(X[i,:,1])
    z = np.copy(X[i,:,2])
    # Output is (e, n, z)
    X[i,:,0] = e[:]
    X[i,:,1] = n[:]
    X[i,:,2] = z[:]
    z_noise = np.var(z)
    n_noise = np.var(n)
    e_noise = np.var(e)
    avg_z_noise = z_noise + avg_z_noise
    avg_n_noise = n_noise + avg_n_noise
    avg_e_noise = e_noise + avg_e_noise
print('z,n,e noise:', avg_z_noise/n_examples, avg_n_noise/n_examples, avg_e_noise/n_examples)

fout = h5py.File(out_file, "w")
fout.create_dataset("X", data=X)
fout.create_dataset("Y", data=Y)
fout.close()