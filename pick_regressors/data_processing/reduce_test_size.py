#%%
# Remove some historical earthquakes to make a smaller test dataset
import h5py 
import pandas as pd
import numpy as np

#%%

df = pd.read_csv("p_picker_resampled/uuss_test.csv")
historical_df = df[(df['event_type'] == 'le') & (df['evid'] < 60000000) ]# %%
qb_df = df[df['event_type'] == 'qb']
len(df) - len(historical_df) - len(qb_df)

assert np.array_equal(historical_df.index.values, range(len(df)-len(historical_df), len(df)))

# %%
# val_df = pd.read_csv("uuss_validation.csv")
# val_heq = val_df[(val_df['event_type'] == 'le') & (val_df['evid'] < 60000000) ]# %%
# val_qb = val_df[(val_df['event_type'] == 'qb')]# %%
# len(val_df) - len(val_heq) - len(val_qb)
# %%

percent_test_hist = 0.20
n_hist = int((len(df)-len(historical_df) - len(qb_df))*percent_test_hist)
hist_keep_inds = np.unique(np.random.choice(historical_df.index, size=n_hist, replace=False))
other_inds = np.arange(len(df)-len(historical_df))
all_inds_new = np.concatenate([other_inds, hist_keep_inds])
#%%
f = h5py.File("p_picker_resampled/uuss_test.h5", "r")
X = f["X"][:]
Y = f["Y"][:]
f.close()

X_smaller = np.copy(X[all_inds_new, :])
Y_smaller = np.copy(Y[all_inds_new])
# %%
df_smaller = df.iloc[all_inds_new]

assert len(df_smaller) == X_smaller.shape[0] and len(df_smaller)== Y_smaller.shape[0]
# %%

f = h5py.File("p_picker_resampled/uuss_test_fewerhist.h5", "w")
f.create_dataset("X", data=X_smaller)
f.create_dataset("Y", data=Y_smaller)
f.close()

df_smaller.to_csv("p_picker_resampled/uuss_test_fewerhist.csv")
# %%