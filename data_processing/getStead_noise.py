#!/usr/bin/env python3
# Purpose: Samples Noise Waveforms from the STEAD database for training in a regressor/classifier network.
# Author: Ben Baker - Edits by Alysha Armstrong
# Date: August 2021 - March 2022

# TODO: Combine this with getStead.py in an organized manner 

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
    outfile_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/sDetector'
    # ch1 is noise
    stead_subdirs = ['ch1']

    max_distance = 150 # Probably everything over ~80 will be critically refracted
    n_samples_stead = 6000
    # Make a larger window than we'd use in practice so we can randomly sample from it
    waveform_length = 20 #seconds
    dt = 0.01

    # Data processing used by Cxx implementation
    process = uuss.ThreeComponentPicker.ZRUNet.ProcessData()

    ##########################################
    waveform_length_samples = waveform_length/dt
    segments_per_trace = int(n_samples_stead//waveform_length_samples) # split the noise window into multiple waveforms

    df_all = None
    ofl = h5py.File(f'{outfile_dir}/SteadNoise.h5', "w")
    dset = ofl.create_dataset("X", (0, waveform_length_samples, 3),
                              maxshape=(None, waveform_length_samples, 3))
    dset_y = ofl.create_dataset("Y", (0,), maxshape=(None,))
    y = np.zeros([1])
    chunk = np.zeros([1, int(waveform_length_samples), 3])
    n_rows = 0
    for stead_subdir in stead_subdirs:
        stead_dir = os.path.join(stead_root_dir, stead_subdir)
        ichunk = stead_subdir.split('ch')[1]
        csv_file = os.path.join(stead_dir, 'chunk' + ichunk + '.csv')
        h5_file = os.path.join(stead_dir, 'chunk' + ichunk + '.hdf5') 
        print("Loading:", csv_file) 
        df = pd.read_csv(csv_file)#, dtype={'s_arrival_sample' : int, 'p_arrival_sample' : int})
        n_rows = n_rows + len(df)
        df_non_uu = df[(df.trace_category == 'noise') &
                       (df['network_code'] != 'UU') &
                       (df['network_code'] != 'WY')]

        print("Getting data from:", h5_file)
        trace_names = df_non_uu['trace_name'].values
        hdf = h5py.File(h5_file, 'r')
        # expand the df to included duplicated noise metadata
        if segments_per_trace > 0:
            df_non_uu = df_non_uu.append([df_non_uu]*int(segments_per_trace-1), ignore_index=False).sort_index()

        keep = np.zeros(len(df_non_uu), dtype='bool')
        #for trace_name in trace_names:
        for i in range(len(trace_names)):
            for waveform_chunk in range(segments_per_trace):
                i0 = int(0 + waveform_chunk*waveform_length_samples)
                i1 = int(i0 + waveform_length_samples)
                assert i0 > -1, 'i0 out of range'
                assert i1 <= n_samples_stead, 'i1 out of range'
                s_signal = hdf['/data/' + trace_names[i]][i0:i1,:]
                e = s_signal[:,0]
                n = s_signal[:,1]
                z = s_signal[:,2]
                [zproc, nproc, eproc] = process.process_three_component_waveform(z, n, e, dt)
                chunk[0, :, 2] = zproc[:]
                chunk[0, :, 1] = nproc[:]
                chunk[0, :, 0] = eproc[:]
                orig_index = dset.shape[0]
                dset.resize(dset.shape[0] + chunk.shape[0], axis=0)
                dset[orig_index:, :, :] = chunk

                dset_y.resize(dset_y.shape[0] + y.shape[0], axis=0)
                #print(s_signal.shape)
                #break
                keep[i+waveform_chunk] = True
        # Update
        print(dset.shape)
        print(len(keep[keep]))
        if (df_all is None):
            df_all = df_non_uu#[keep]
        else:
            df_all = pd.concat([df_all, df_non_uu])#[keep]])
        print(len(df_all.shape))
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
    print("Fraction of all waveforms with Noise?:", df_all.shape[0]/float(n_rows))
    df_all.to_csv(f'{outfile_dir}/SteadNoise.csv', index=False)
    ofl.close()
