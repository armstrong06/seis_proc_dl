#!/usr/bin/env python3
# Purpose: Samples Signal Waveforms from the STEAD database for training in a regressor/classifier network.
# Author: Ben Baker - Edits by Alysha Armstrong
# Date: August 2021 - March 2022

import sys
sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/templateMatchingSource/rtseis/notchpeak4_gcc83_build/')
sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/mlmodels/deepLearning/apply/np4_build')
import h5py
import pyuussmlmodels as uuss
import numpy as np
import pandas as pd
import os

if __name__ == "__main__":
    
    stead_root_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/waveformArchive/stead/'
    outfile_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/STEAD_P'
    max_distance = 150 # Probably everything over ~80 will be critically refracted
    n_samples_stead = 6000
    # Make a larger window than we'd use in practice so we can randomly sample
    # from it
    secs_before_pick = -7.5 # Seconds before arrival
    secs_after_pick  =  10 # Seconds after arrival
    dt = 0.01
    # ch1 is a noise directory
    stead_subdirs = ['ch2', 'ch3', 'ch4', 'ch5', 'ch6']
    phase = "P"
    # Data processing used by Cxx implementation
    process = uuss.ThreeComponentPicker.ZRUNet.ProcessData()

    pref = '/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/pDetector3C'
    uuss_meta_df = pd.read_csv(f'{pref}/current_earthquake_catalog_3C.csv')
    uuss_meta_df["comb_code"] = uuss_meta_df["network"] + uuss_meta_df["station"]

    ########################################
    window_length_samples = int((secs_after_pick - secs_before_pick)/dt) + 1
    min_pick_sample = int(-secs_before_pick/dt)
    max_pick_sample = n_samples_stead - int(secs_after_pick/dt) - 1

    df_all = None
    ofl = h5py.File(f'{outfile_dir}/{phase}SteadDataset.h5', "w")
    dset = ofl.create_dataset("X", (0, window_length_samples, 3),
                              maxshape=(None, window_length_samples, 3))
    dset_y = ofl.create_dataset("Y", (0,), maxshape=(None,))
    y = np.zeros([1]) - secs_before_pick
    chunk = np.zeros([1, window_length_samples, 3])
    n_rows = 0

    for stead_subdir in stead_subdirs:
        stead_dir = os.path.join(stead_root_dir, stead_subdir)
        ichunk = stead_subdir.split('ch')[1]
        csv_file = os.path.join(stead_dir, 'chunk' + ichunk + '.csv')
        h5_file = os.path.join(stead_dir, 'chunk' + ichunk + '.hdf5') 
        print("Loading:", csv_file) 
        df = pd.read_csv(csv_file)#, dtype={'s_arrival_sample' : int, 'p_arrival_sample' : int})
        n_rows = n_rows + len(df)
        
        arrival_sample_key = "s_arrival_sample"
        arrival_status_key = "s_status"
        if phase == "P":
            arrival_sample_key = "p_arrival_sample"
            arrival_status_key = "p_status"

        df["comb_code"] = df.network_code + df.receiver_code

        # ~df.source_id.str.contains('uu').values
        df_non_uu = df[(df.trace_category == 'earthquake_local') &
                       (df[arrival_status_key] == 'manual') &
                       (df['source_distance_km'] < max_distance) &
                       (df[arrival_sample_key] > min_pick_sample) &
                       (df[arrival_sample_key] < max_pick_sample) &
                       (np.isin(df.comb_code, uuss_meta_df.comb_code.unique(), invert=True)) &
                       (df.source_magnitude_author) != "UU"]

        # ((np.isin(stead_meta_df["network_code"], uuss_meta_df["network"].unique())) &
        #                           (np.isin(stead_meta_df["receiver_code"], uuss_meta_df["station"].unique()))) |
        #                           (np.isin(stead_meta_df["network_code"],["UU", "WY"]))

        print("Getting data from:", h5_file)
        trace_names = df_non_uu['trace_name'].values
        s_arrival = df_non_uu[arrival_sample_key].values.astype('int')
        hdf = h5py.File(h5_file, 'r')
        keep = np.zeros(len(df_non_uu), dtype='bool')
        #for trace_name in trace_names:
        for i in range(len(trace_names)):
            i0 = s_arrival[i] + int(secs_before_pick/dt)
            i1 = s_arrival[i] + int(secs_after_pick/dt) + 1
            assert i0 > -1, 'i0 out of range'
            assert i1 <= n_samples_stead, 'i1 out of range'
            s_signal = hdf['/data/' + trace_names[i]][i0:i1,:]
            e = s_signal[:,0]
            n = s_signal[:,1]
            z = s_signal[:,2]
            [zproc, nproc, eproc] = process.process_three_component_waveform(z, n, e, dt)
            # Order ENZ
            chunk[0, :, 2] = zproc[:]
            chunk[0, :, 1] = nproc[:]
            chunk[0, :, 0] = eproc[:]

            # check if there are any nan values
            if len(np.where(np.isnan(chunk))[0]) > 0:
                print("Nan values - skipping waveform")
                keep[i] = False
                continue

            orig_index = dset.shape[0]
            dset.resize(dset.shape[0] + chunk.shape[0], axis=0)
            dset[orig_index:, :, :] = chunk

            dset_y.resize(dset_y.shape[0] + y.shape[0], axis=0)
            #print(s_signal.shape)
            #break 
            keep[i] = True
        # Update
        if (df_all is None):
            df_all = df_non_uu[keep]
        else:
            df_all = pd.concat([df_all, df_non_uu[keep]])
        #
        #break
    # Loop on subdirectories
    columns_keep = ['network_code','receiver_code','receiver_type',
                    'receiver_latitude','receiver_longitude','receiver_elevation_m',
                    'p_arrival_sample','p_status','p_weight','p_travel_sec',
                    's_arrival_sample','s_status','s_weight','source_id',
                    'source_origin_time','source_origin_uncertainty_sec',
                    'source_latitude','source_longitude','source_error_sec',
                    'source_gap_deg','source_horizontal_uncertainty_km','source_depth_km',
                    'source_depth_uncertainty_km','source_magnitude','source_magnitude_type',
                    'source_magnitude_author','source_distance_deg','source_distance_km',
                    'back_azimuth_deg','trace_start_time','trace_category','trace_name']

    df_all = df_all[columns_keep]
    print(f"Fraction of all waveforms with {phase} picks:", df_all.shape[0]/float(n_rows))
    df_all.to_csv(f'{outfile_dir}/{phase}SteadMetadata.csv', index=False)
    ofl.close()
