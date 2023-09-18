#%%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time

df = pd.read_csv("/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/stead/PStead_1790.csv")
evids = df.source_id.values
unique_evids = np.unique(evids)
evids_train, evids_test = train_test_split(unique_evids, train_size=0.8)

#%%
start_time = time.time()
rows_train = np.where(np.isin(evids, evids_train))[0]
print(time.time() - start_time)
# 388 s
#%%
start_time = time.time()
evids_in_train = [1 if evid in evids_train else 0 for evid in evids]
print(time.time() - start_time)
#I think around 900 seconds