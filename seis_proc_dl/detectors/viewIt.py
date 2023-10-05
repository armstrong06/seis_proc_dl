#!/usr/bin/env python3
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pref = '/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/sDetector/s_resampled_10s_normalized'
f = h5py.File(f'{pref}/S.train.10s.1dup.h5', 'r')
df = pd.read_csv(f'{pref}/S.train.10s.1dup.csv')#, dtype={'location': object})
for i in range(10000, 10010):
    #plt.plot(np.linspace(-3,3,600), f['X'][251340][:])
    idx = i#(i + 1)*1510 + i
    print(idx)
    """
    network = df['network'][idx] 
    station = df['station'][idx]
    channel = df['channelz'][idx]
    location = df['location'][idx] 
    evid = df['evid'][idx] 
    """

    seism = f['X'][idx]
    y = f["Y"][idx]
    T = f["Pick_index"][idx]
    z = seism[:,2]
    n = seism[:,1]
    e = seism[:,0]

    plt.plot(range(seism.shape[0]), z + 2)
    plt.plot(range(seism.shape[0]), n + 1)
    plt.plot(range(seism.shape[0]), e)
    plt.plot(range(seism.shape[0]), y)
    plt.axvline(T)
    plt.xlabel('Samples')
    plt.grid(True)
    plt.show()

f.close()
