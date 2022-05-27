#!/usr/bin/env python3
# Purpose: Split the STEAD dataset into a training and validation dataset.
import pandas as pd
import numpy as np
import sklearn as sk
import h5py 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys


if __name__ == "__main__":
    make_ec = False
    validation_size = 0.2
    np.random.seed(38883)
    if (make_ec):
        df = pd.read_csv('stead/sSteadMetadata.csv')
        df['original_rows'] = np.arange(0, len(df))
        # Ensure no events are in UT or Yellowstone since USGS or Nevada
        # can contribute events in our authoriative region
        df['source_depth_km'].replace("None", np.nan, inplace = True)
        df['source_depth_km'].fillna(value = np.nan, inplace = True)
        depths = df['source_depth_km'].astype(float)
        df['source_depth_km'] = depths
        avg_depth = int(np.nanmean(df['source_depth_km'].values))
        print("Imputing missing source depths with average depth:", avg_depth)
        df.fillna(avg_depth, inplace=True)
        # Call events at the same `lat/lon' the same and give it an evid.
        print("Computing unique event locations")
        llds = np.zeros([len(df), 3])
        # Quantize to speed this up
        llds[:,0] = df['source_latitude'].to_numpy()
        llds[:,1] = df['source_longitude'].to_numpy()
        llds[:,2] = df['source_depth_km'].to_numpy()
        llds = llds*10
        llds = llds.astype(int) 
        evids = np.zeros(len(df), dtype='int')
        unique_llds = np.unique(llds, axis=0)
        original_rows = df['original_rows'].values
        print("Number of unique event locations:", len(unique_llds))
        print("Assigning event IDs...")
        evid = 1
        for unique_lld in unique_llds:
            lat = unique_lld[0]
            lon = unique_lld[1]
            dep = unique_lld[2]
            keep_me = (llds[:,0] == lat) & (llds[:,1] == lon) & (llds[:,2] == dep)
            rows_local = original_rows[keep_me] #df[ (df['source_latitude'] == lat) & (df['source_longitude'] == lon) & (df['source_depth_km'] == dep)].original_rows
            if (len(rows_local) == 0):
                print(unique_lld)
                sys.exit("Critical error - unique item not found")
            #rows_local = original_rows[ (llds[:,0] == lat) & (llds[:,1] == lon) ]
            evids[rows_local] = evid
            evid = evid + 1 
        print(min(evids), max(evids))
        df['event_counter'] = evids
        df.to_csv('stead/sSteadEC.csv', index=False)
    else:
        print("Will simply load STEAD catalog with ad-hoc event IDs...")
    # Load the dataframe
    df = pd.read_csv('stead/sSteadEC.csv')
    print("Read %d rows"%len(df))
    print("Dropping events in UUSS authorative region...")
    assert min(df.source_longitude) >-180, 'need min lon >-180'
    assert max(df.source_longitude) < 180, 'need max lon < 180'
    df = df[~((df['source_latitude'] < 42.5) &
              (df['source_latitude'] > 36.75) &
              (df['source_longitude'] < -108.75) &
              (df['source_longitude'] > -114.25)) |
              ((df['source_latitude'] < 45.167) &
              (df['source_latitude'] > 44) &
              (df['source_longitude'] < -109.75) &
              (df['source_longitude'] > -111.333)) ]
    print("Will proceed with %d rows"%len(df))
    evids = np.unique(df['event_counter'])
    train_events, validation_events = train_test_split(evids, test_size = validation_size)
    test_df = df[df['event_counter'].isin(train_events)]
    validation_df = df[df['event_counter'].isin(validation_events)]
    rows_train = test_df['original_rows'].values
    rows_validation = validation_df['original_rows'].values
    assert len(rows_train) + len(rows_validation) == len(df), 'missed events' 
    print("Evids", evids)
    f = h5py.File('stead/sSteadDataset.h5', 'r')  

    X = f['X'][:]
    Y = f['Y'][:]

    print("Training rows:", rows_train)
    print("Validation rows:", rows_validation)
    assert len(np.intersect1d(rows_train, rows_validation)) == 0, 'training and validation rows not distinct'

    print("Writing training...")
    of = h5py.File('stead/train_s.h5', 'w')
    of['X'] = X[rows_train,:,:]
    of['Y'] = Y[rows_train]
    of.close()
    test_df.to_csv('stead/train_s.csv', index=False)

    print("Writing validation...")
    of = h5py.File('stead/validation_s.h5', 'w')
    of['X'] = X[rows_validation,:,:]
    of['Y'] = Y[rows_validation]
    of.close()
    test_df.to_csv('stead/validation_s.csv', index=False)
