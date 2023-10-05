#%%
import h5py
import numpy as np
#%%
boxcar_widths={0: 31, 1: 31, 2: 31}
pref = '/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/oneCompPdetector/' \
       'constant_bc_width/NGB_onecomp_p_resampled_10s/'

#%%
file = h5py.File(f"{pref}/currenteq.NGB.10s.1dup.h5", "r")
X = file["X"][:]
Y = file["Y"][:]
T = file["Pick_index"][:]

file.close()
#%%
Y_new = np.zeros_like(Y)
halfwidth = 15
for i, t in enumerate(T):
    Y_new[i, t-halfwidth:t+halfwidth+1] = 1
    assert np.where(Y_new[i, :]==1)[0].shape[0] == halfwidth*2+1
    assert np.where(Y_new[i, :]==1)[0][halfwidth] == t
#%%
file = h5py.File(f"{pref}/currenteq.NGB.10s.1dup.cb.h5", "w")
file.create_dataset('X', data=X)
file.create_dataset("Pick_index", data=T)
file.create_dataset("Y", data=Y_new)
print(np.where(file["Y"][0, :] == 1)[0].shape)
file.close()