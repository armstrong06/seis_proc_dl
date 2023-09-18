#%%
# Remove some historical earthquakes to make a smaller test dataset
import h5py 
import pandas as pd
import numpy as np
np.random.seed(3175)
#%%
pref = "/home/armstrong/Research/constant_boxcar_widths_NGB/oneCompPDetector/uuss_data"
df = pd.read_csv(f"{pref}/combined.test.10s.1dup.csv")
historical_df = df[(df['event_type'] == 'le') & (df['evid'] < 60000000) ]# %%
qb_df = df[df['event_type'] == 'qb']
len(df) - len(historical_df) - len(qb_df)

assert np.array_equal(historical_df.index.values, range(len(df)-len(historical_df), len(df)))

# %%

percent_test_hist = 0.20
n_hist = int((len(df)-len(historical_df) - len(qb_df))*percent_test_hist)
hist_keep_inds = np.random.choice(historical_df.index, size=n_hist, replace=False)
other_inds = np.arange(len(df)-len(historical_df))
#%%
f = h5py.File(f"{pref}/combined.test.10s.1dup.h5", "r")
X = f["X"][:]
Y = f["Y"][:]
T = f["Pick_index"][:]
f.close()

noise_inds = np.arange(len(df), X.shape[0])
all_inds_new = np.concatenate([other_inds, hist_keep_inds, noise_inds])

X_smaller = np.copy(X[all_inds_new, :])
Y_smaller = np.copy(Y[all_inds_new])
T_smaller = np.copy(T[all_inds_new])

assert np.array_equal(X[:len(df)-len(historical_df)], X_smaller[:len(df)-len(historical_df)])
assert np.array_equal(X[-len(noise_inds):], X_smaller[-len(noise_inds):])

assert np.array_equal(Y[:len(df)-len(historical_df)], Y_smaller[:len(df)-len(historical_df)])
assert np.array_equal(Y[-len(noise_inds):], Y_smaller[-len(noise_inds):])

assert np.array_equal(T[:len(df)-len(historical_df)], T_smaller[:len(df)-len(historical_df)])
assert np.array_equal(T[-len(noise_inds):], T_smaller[-len(noise_inds):])
# %%
df_smaller = df.iloc[all_inds_new[:-len(noise_inds)]]

assert len(df_smaller) == X_smaller.shape[0]-len(noise_inds) 
assert len(df_smaller)== Y_smaller.shape[0]-len(noise_inds)
assert len(df_smaller)== T_smaller.shape[0]-len(noise_inds)

# %%

f = h5py.File(f"{pref}/combined.test.fewerhist.10s.1dup.h5", "w")
f.create_dataset("X", data=X_smaller)
f.create_dataset("Y", data=Y_smaller)
f.create_dataset("Pick_index", data=T_smaller)
f.close()

df_smaller.to_csv(f"{pref}/combined.test.fewerhist.10s.1dup.csv")
# %%