#!/usr/bin/env python3
import sys
sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/waveformArchive/gcc_build')
#sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/templateMatchingSource/rtseis/notchpeak4_gcc83_build/')
sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/mlmodels/deepLearning/apply/np4_build')
import h5py
import pyWaveformArchive as pwa
import pyuussmlmodels as uuss
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_data(vertical, north, east,
                 pick_time,
                 trace_cut_start = -8, trace_cut_end = 8):
    """ 
    Applies basic processing and interpolation of input signals.

    Parameters
    ----------
    north : waveform archive object
       The north waveform to filter.
    east : waveform archive object
       The east waveform to filter.
    vertical : waveform archive object
       The vertical archive object 
    pick_time : double
       The pick time in UTC seconds since the epoch.
    trace_cut_start : double
       The seconds before the pick time to begin the cut window.
    trace_cut_end : double
       The seconds after the pick time to end the cut window.

    Returns
    -------
    waveform : waveform archive object
       The filtered waveform.  Note, this can be none if an error
       was encountered or the input signal is too small.
    """
    n_samples_out = int((trace_cut_end - trace_cut_start)/0.01) + 1
    # Ensure sampling rates make sense
    if (abs(north.sampling_rate - east.sampling_rate) > 1.e-4 or
        abs(north.sampling_rate - vertical.sampling_rate) > 1.e-4):
        print("Inconsistent sampling rates")
        return [None, None, None]
    dt = 1./north.sampling_rate
    # Cut data.  Step 1 -> get start times and signal lengths
    e_signal = east.signal
    n_signal = north.signal
    z_signal = vertical.signal
    ne = len(e_signal)
    nn = len(n_signal)
    nz = len(z_signal)
    te0 = east.start_time
    tn0 = north.start_time
    tz0 = vertical.start_time
    te1 = te0 + dt*(ne - 1)
    tn1 = tn0 + dt*(nn - 1)
    tz1 = tz0 + dt*(nz - 1)
    # Get the latest valid start time and earliest valid end time
    t0max = max(max(te0, tn0), tz0) # Last start time
    t1min = min(min(te1, tn1), tz1) # Earliest end time
    n_samples = int((t1min - t0max)/dt + 0.5) + 1
    # Start index
    ie0 = int( (pick_time + trace_cut_start - te0)/dt + 0.5 ) 
    in0 = int( (pick_time + trace_cut_start - tn0)/dt + 0.5 )
    iz0 = int( (pick_time + trace_cut_start - tz0)/dt + 0.5 )
    # End index
    ie1 = ie0 + n_samples_out
    in1 = in0 + n_samples_out
    iz1 = iz0 + n_samples_out
    if (ie0 < 0 or in0 < 0 or iz0 < 0):
        print("Cut starts before trace starts - skipping")
        return [None, None, None]
    if (ie1 > n_samples or in1 > n_samples or iz1 > n_samples):
        print("Cut ends after trace ends - skipping")
        return [None, None, None]
    # Plotting 
    debug = False 
    """
    if (debug):
        plt.plot(e_signal[ie0:ie1])
        plt.plot(n_signal[in0:in1])
        plt.plot(z_signal[iz0:iz1])
        plt.show()
    """
    east_cut     = e_signal[ie0:ie1]
    north_cut    = n_signal[in0:in1]
    vertical_cut = z_signal[iz0:iz1]
    if (debug):
        plt.plot(east_cut)
        plt.plot(north_cut)
        plt.plot(vertical_cut)
        plt.show()
    # Is dead?
    if (max(abs(east_cut))  < 1.e-14 and
        max(abs(north_cut)) < 1.e-14 and
        max(abs(vertical_cut)) < 1.e-14):
        print("Dead channel detected")
        return [None, None, None]
    # Process the waveforms
    process = uuss.ThreeComponentPicker.ZCNN.ProcessData()
    target_dt = process.target_sampling_period
    [zproc, nproc, eproc] = process.process_three_component_waveform(vertical_cut,
                                                                     north_cut,
                                                                     east_cut,
                                                                     dt)
    if (debug):
        plt.plot(eproc)
        plt.plot(nproc)
        plt.plot(zproc)
        plt.show()

    north.signal = nproc
    east.signal = eproc
    vertical.signal = zproc

    north.sampling_rate = 1/target_dt
    east.sampling_rate = 1/target_dt
    vertical.sampling_rate = 1/target_dt

    north.start_time    = tn0 + in0*dt
    east.start_time     = te0 + ie0*dt
    vertical.start_time = tz0 + iz0*dt

    return [vertical, north, east]

def create_s_waveform_list(catalog_3c):
    """
    From the catalog this creates a list of waveforms observed on the
    three-component stations with S picks.

    Parameters
    ----------
    catalog_3c : string
        Name of the three-component catalog.
    """
    phase = 'S'
    catalog_3c_df = pd.read_csv(catalog_3c, dtype={'location': object})
    print("Original length of 3C catalog:", len(catalog_3c_df))
    # Require picks be of the specified phase type
    catalog_3c_df = catalog_3c_df[catalog_3c_df['phase'] == phase]
    print("Length of 3C S catalog:", len(catalog_3c_df))
    # Retain the useful columns like we would for other processing.
    # Typically channel1 will be N and channel2 will be E.
    # though we use 1 and 2 to denote borehole stations which I think
    # have the same reference frame as a conventional seismometer but are
    # arbitrarily rotated as they are forced downhole.
    catalog_df = catalog_3c_df[['evid', 'network', 'station', 'location', 'channel1', 'channel2', 'channelz',
                                'phase', 'arrival_time', 'pick_quality', 'first_motion',
                                'take_off_angle', 'source_receiver_distance', 'source_receiver_azimuth',
                                'travel_time_residual', 'receiver_lat', 'receiver_lon', 'event_lat',
                                'event_lon', 'event_depth', 'origin_time', 'magnitude',
                                'magnitude_type', 'rflag']].copy()
    # Impute blank station codes
    catalog_df['location'].replace(["  "], "", regex=True, inplace=True)
    # Sort catalog
    catalog_df.sort_values(['evid', 'arrival_time'], inplace=True)
    return catalog_df 

def make_archive(catalog_df, archive_manager, output_file_root,
                 s_before =-8, s_after = 8, dt = 0.01):
    if (s_before >= s_after):
        sys.exit("Start cut exceeds end cut")
    if (dt <= 0):
        sys.exit("Sampling period must be positive")
    # Make a larger window than we'd use in practice so we can randomly sample
    # from it
    s_window_samples = int((s_after - s_before)/dt) + 1
    print("Anticipated output length in samples:", s_window_samples)
    #min_s_sample = int(-s_before/dt)
    #max_s_sample = n_samples_stead - int(s_after/dt) - 1
    #max_s_sample = n_samples_stead - int(s_after/dt) - 1
    chunk = np.zeros([1, s_window_samples, 3])

    evids = catalog_df['evid'].values
    stations = catalog_df['station'].values
    networks = catalog_df['network'].values
    channels1 = catalog_df['channel1'].values
    channels2 = catalog_df['channel2'].values
    channelsz = catalog_df['channelz'].values
    locations = catalog_df['location'].values 
    arrival_times = catalog_df['arrival_time'].values
    print(stations, networks, channels1, channels2, channelsz, locations)
    y = np.zeros([1]) - s_before
    lfound = np.zeros(len(evids), dtype='bool')
    # Initialize hdf5
    output_file_h5 = output_file_root + ".h5"
    ofl = h5py.File(output_file_h5, "w")
    dset = ofl.create_dataset("X", (0, s_window_samples, 3), 
                              maxshape=(None, s_window_samples, 3)) 
    dset_y = ofl.create_dataset("Y", (0,), maxshape=(None,))
    for i in range(len(evids)):
        #evids[i] = 60377292
        #networks[i] = 'UU'
        #stations[i] = 'VEC'
        #locations[i] = '01'
        #channels[i] = 'ENZ'

        exists1 = archive_manager.waveform_exists(evids[i], 
                                                  networks[i], stations[i],
                                                  channels1[i], locations[i])
        exists2 = archive_manager.waveform_exists(evids[i], 
                                                  networks[i], stations[i],
                                                  channels2[i], locations[i])
        existsZ = archive_manager.waveform_exists(evids[i], 
                                                  networks[i], stations[i],
                                                  channelsz[i], locations[i])
        if (exists1 and exists2 and existsZ):
            north    = archive_manager.read_waveform(evids[i],
                                                     networks[i], stations[i],
                                                     channels1[i], locations[i])
            east     = archive_manager.read_waveform(evids[i],
                                                     networks[i], stations[i],
                                                     channels2[i], locations[i])
            vertical = archive_manager.read_waveform(evids[i],
                                                     networks[i], stations[i],
                                                     channelsz[i], locations[i])
            # Clean up any junk at end of waveform
            north.remove_trailing_zeros()
            east.remove_trailing_zeros()
            vertical.remove_trailing_zeros()
            [vertical, north, east] = process_data(vertical, north, east,
                                                   arrival_times[i],
                                                   trace_cut_start = s_before,
                                                   trace_cut_end   = s_after) 
            if (north is None):
                print("Insufficient data to process waveform: "
                    + networks[i] + "." + stations[i] + "."
                    + channelsz[i][0:2] + "?." + locations[i])
                continue
            if (len(vertical.signal) != s_window_samples):
                print("Shorted time series - expected it to be longer")
                continue
            # Process data - Output ENZ
            chunk[0, :, 2] = vertical.signal[:]
            chunk[0, :, 1] = north.signal[:]
            chunk[0, :, 0] = east.signal[:]
            orig_index = dset.shape[0]
            dset.resize(dset.shape[0] + chunk.shape[0], axis=0)
            dset[orig_index:, :, :] = chunk

            dset_y.resize(dset_y.shape[0] + y.shape[0], axis=0)
            lfound[i] = True
            #if (len(dset) == 100):
            #    break
        else:
            print("Waveform: "
                + networks[i] + "." + stations[i] + "."
                + channelsz[i][0:2] + "?." + locations[i]
                + " does not exist for event: ", evids[i])
        #if (len(dset) == 100):
        #    break
    k = np.sum(1*lfound)
    print("Read %d waveforms out of %d lines in dataframe (%.2f pct)"%(k, len(evids), float(k)/len(evids)*100))
    #X.resize([k, n_samples])
    #y = np.zeros(k, dtype = 'f4') + (n_samples/2.)*0.01
    output_file_csv = output_file_root + ".csv"
    catalog_df_out = catalog_df[lfound]
    print(len(dset_y))
    print(len(dset))
    print(dset.shape, dset_y.shape, len(catalog_df_out))
    assert len(catalog_df_out) == k, 'dataframe subsample failed'
    assert len(catalog_df_out) == len(dset_y), 'df size != target size'
    assert len(catalog_df_out) == len(dset), 'df size != dataset size'
    catalog_df_out.to_csv(output_file_csv, index=False)

    
if __name__ == "__main__":
    s_before =-4
    s_after  = 4
    dt = 0.01 # Nominal network sampling period
    #print(dir(rtseis.PostProcessing.Waveform))
    archive_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/waveformArchive/archives/'
    catalog_dir = '/uufs/chpc.utah.edu/common/home/koper-group1/bbaker/waveformArchive/data/'

    output_dir = "/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/sPicker"

    h5_archive_files = glob.glob(archive_dir + '/archive_????.h5')
    historical_eq_catalog_3c = os.path.join(catalog_dir, 'historicalEarthquakeArrivalInformation3C.csv')
    current_eq_catalog_3c = os.path.join(catalog_dir, 'currentEarthquakeArrivalInformation3C.csv')
    current_blast_catalog_3c = os.path.join(catalog_dir, 'currentBlastArrivalInformation3C.csv')

    print("Loading current earthquake catalog...")
    current_eq_catalog_df = create_s_waveform_list(current_eq_catalog_3c)
    current_eq_catalog_df['event_type'] = 'le'
    # There's so few blast examples that it is pointless
    print("Loading current blast catalog...")
    current_blast_catalog_df = create_s_waveform_list(current_blast_catalog_3c)
    current_blast_catalog_df['event_type'] = 'qb'
    print("Loading historical earthquake catalog...")
    historical_eq_catalog_df = create_s_waveform_list(historical_eq_catalog_3c)
    historical_eq_catalog_df['event_type'] = 'le'

    print(len(current_eq_catalog_df),  len(historical_eq_catalog_df))

    print("Opening archive files for reading...")
    archive_manager = pwa.ArchiveManager()
    archive_manager.open_files_for_reading(h5_archive_files)
 
    make_archive(current_eq_catalog_df, archive_manager, f'{output_dir}/current_earthquake_catalog_s',
                 s_before, s_after, dt)
    make_archive(current_blast_catalog_df, archive_manager, f'{output_dir}/current_blast_catalog_s',
                 s_before, s_after, dt)
    make_archive(historical_eq_catalog_df, archive_manager, f'{output_dir}/historical_earthquake_catalog_s',
                 s_before, s_after, dt)

    archive_manager.close()
