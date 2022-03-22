from turtle import down
from .base_gatherdata import BaseGatherDataUUSS
import numpy as np
import pandas as pd
from utils.file_manager import Write
from detectors.data_processing import make_yboxcar as boxcar

class OneComponentGatherer(BaseGatherDataUUSS):
    def __init__(self, archive_dir, processing_function):
        super().__init__(archive_dir, processing_function)

    def process_data(self, waveforms, pick_time, trace_cut_start = -3, trace_cut_end = 3):
        """ 
        Overrides method from BaseGatherData
        """
        n_samples_out = int((trace_cut_end - trace_cut_start)/0.01)
        signal = np.copy(waveforms.signal)
        if (len(signal) == 0):
            return None
        processed_signal = self.proc.process_waveform(signal, 1./waveforms.sampling_rate)
        target_dt = self.proc.target_sampling_period
        t0 = waveforms.start_time
        n_samples = len(signal)
        # Extract signal
        trace_start_index = int((pick_time + trace_cut_start - t0)/target_dt + 0.5)
        trace_end_index = trace_start_index + n_samples_out
        if (trace_start_index < 0 or trace_end_index >= n_samples):
            return None 

        waveforms.sampling_rate = 1./target_dt
        waveforms.signal = np.copy(processed_signal[trace_start_index:trace_end_index])
        waveforms.start_time = trace_cut_start
        return waveforms

    def create_waveform_df(self, catalog_file_name):
        """ Overrides method from BaseGatherData. Drop non-vertical channels from catalogs"""
        phase = 'P'
        catalog_df = pd.read_csv(catalog_file_name, dtype={'location': object})

        print("Original length of catalog:", len(catalog_df))
        # Require picks be of the specified phase type
        catalog_df = catalog_df[catalog_df['phase'] == phase]
        print("Length of catalog:", len(catalog_df))
        # Drop non-vertical channels and ensure columns are same so we can 
        # combine the dataframes
        catalog_df = catalog_df[['evid', 'network', 'station', 'location', 'channelz',
                                    'phase', 'arrival_time', 'pick_quality', 'first_motion',
                                    'take_off_angle', 'source_receiver_distance', 'source_receiver_azimuth',
                                    'travel_time_residual', 'receiver_lat', 'receiver_lon', 'event_lat',
                                    'event_lon', 'event_depth', 'origin_time', 'magnitude',
                                    'magnitude_type', 'rflag']]
        
        # Impute blank station codes
        catalog_df['location'].replace(["  "], "", regex=True, inplace=True)
  
        # Sort catalog
        catalog_df.sort_values(['evid', 'arrival_time'], inplace=True)
        return catalog_df 
        
    def make_archive(self, catalog_df, output_file_root):
        """ Overrides method from BaseGatherData."""
        evids = catalog_df['evid'].values
        stations = catalog_df['station'].values
        networks = catalog_df['network'].values
        channels = catalog_df['channelz'].values
        locations = catalog_df['location'].values 
        arrival_times = catalog_df['arrival_time'].values
        fms = catalog_df['first_motion'].values
        
        #print(stations, networks, channels, locations)
        
        lfound = np.zeros(len(evids), dtype='bool')
        X = None
        k = 0
        for i in range(len(evids)):
            exists = self.archive_manager.waveform_exists(evids[i], 
                                                    networks[i], stations[i],
                                                    channels[i], locations[i])
            if (exists):
                waveform = self.archive_manager.read_waveform(evids[i],
                                                        networks[i], stations[i],
                                                        channels[i], locations[i])
                waveform.remove_trailing_zeros() # Clean up any junk at end of waveform
                waveform = self.process_data(waveform, arrival_times[i]) 
                if (waveform is None):
                    print("Insufficient data to process waveform: "
                        + networks[i] + "." + stations[i] + "." + channels[i] + "."
                        + locations[i])
                    continue
                # Process data
                signal = waveform.signal
                n_samples = len(signal)
                if (X is None):
                    X = np.zeros([len(evids), n_samples], dtype='f4')
                X[k, :] = signal[:]
                k = k + 1
                lfound[i] = True
            else:
                print("Waveform: "
                    + networks[i] + "." + stations[i] + "."
                    + channels[i] + "." + locations[i]
                    + " does not exist for event: ", evids[i])
        print("Read %d waveforms out of %d lines in dataframe (%.2f pct)"%(k, len(evids), float(k)/len(evids)*100))
        X.resize([k, n_samples])
        y = np.copy(fms[lfound]) 
        assert len(X[:,0]) == len(y), 'rows in X does not match y'
        output_file_h5 = output_file_root + ".h5"
        Write.h5py_file(["X", "Y"], [X, y], output_file_h5)

        output_file_csv = output_file_root + ".csv"
        catalog_df = catalog_df[lfound]
        assert len(catalog_df) == np.sum(1*lfound), 'dataframe subsample failed'
        catalog_df.to_csv(output_file_csv, index=False)

    def create_combined_catalogs(self, catalog_3c_filename, catalog_1c_filename):
        """
        Join catalogs for 3C and 1C vertical data. 
        Parameters
        ----------
        catalog_3c_filename: string
            path to 3 component csv file
        catalog_1c_filename: string
            path to 1-component csv file

        Returns
        ---------- 
        combined_catalog_df: Pandas DataFrame
            DataFrame of the combined waveform metadata for the csv files - sorted by arrival time       
        """
        catalog_3c_df = self.create_combined_catalogs(catalog_3c_filename)
        catalog_1c_df = self.create_combined_catalogs(catalog_1c_filename)
        combined_catalog_df = pd.concat([catalog_3c_df, catalog_1c_df])
        combined_catalog_df.sort_values(['evid', 'arrival_time'], inplace=True)
        print("Length of combined catalog:", len(combined_catalog_df))
        return combined_catalog_df

    def additional_fm_filtering(catalog_df, drop_down=False):
        """
        Remove the "1" quality picks from First Motion catalog and optionally remove any "down" first motions (for quarry blasts)

        Parameters
        ----------
        catalog_df: Pandas DataFrame
            all waveform metadata to be ordered 
        drop_downs (optional): boolean
            specifies whether to drop the down first motions. Defaults to False.
        Returns
        ---------- 
        catalog_df: Pandas DataFrame
            waveform metadata with "1" quality picks removed and possibly no "down" first motions
        """
        ## Since "1" quality picks (0.75 jiggle weight) tend to be confusing we drop them here.
        catalog_df = catalog_df[(catalog_df['pick_quality'] == 0.5) |
                                (catalog_df['pick_quality'] == 1)]
        if (drop_down):
            print("Dropping", np.sum( (catalog_df['first_motion'] ==-1)*1 ),
                "downward first motions")
            catalog_df = catalog_df[catalog_df['first_motion'] !=-1]
        return catalog_df


    # TODO: Could figure out a better way to separate the functionality for FM and 1C picker
    def process_and_save_waveforms(self, catalog_3c_filename, catalog_1c_filename, output_file_root, event_type, is_first_motion_data=False, drop_down=False):
        """
        Write archived time-series and metadata to disk for specified waveforms

        Parameters
        ----------
        catalog_3c_filename: string
            path to 3 component metadata csv file
        catalog_1c_filename: string
            path to 1-component metadata csv file
        output_file_root: string
            path and name (not including file-type suffix) for the archive time-series and metadata files
        drop_downs (optional): boolean
            specifies whether to drop the down first motions. Defaults to False.
        """
        combined_catalog_df = self.create_combined_catalogs(catalog_3c_filename, catalog_1c_filename)
        if is_first_motion_data:
            combined_catalog_df = self.additional_fm_filtering(combined_catalog_df, drop_down=drop_down)
        combined_catalog_df['event_type'] = event_type
        self.make_archive(combined_catalog_df, output_file_root)

    # def close(self):
    #     """ Close the archive manager used to access the archived time-series data"""
    #     self.archive_manager.close()

class ThreeComponentGatherer(BaseGatherDataUUSS):
    def __init__(self, archive_dir, processing_function):
        super().__init__(archive_dir, processing_function)
        # TODO: I think all 3C will use the same processing functions - fix implementation 

    def process_data(self, waveforms, pick_time, trace_cut_start = -3, trace_cut_end = 3):
        """ 
        Overrides method from BaseGatherData
        """
        n_samples_out = int((trace_cut_end - trace_cut_start)/0.01)
        #proc = uuss.ThreeComponentPicker.ZRUNet.ProcessData()
        signalZ = np.copy(waveforms[0].signal)
        signalN = np.copy(waveforms[1].signal)
        signalE = np.copy(waveforms[2].signal)

        print(waveforms[0].sampling_rate, waveforms[1].sampling_rate, waveforms[2].sampling_rate)
        if len(signalZ) == 0 or len(signalN) == 0 or len(signalE) == 0:
            return [None, None, None]
        if 1. / waveforms[0].sampling_rate != 1. / waveforms[1].sampling_rate != 1. / waveforms[2].sampling_rate:
            print("Sampling rates do not match")
            return [None, None, None]
        if waveforms[0].start_time != waveforms[1].start_time !=  waveforms[2].start_time:
            print("Start times do not match", waveforms[0].start_time, waveforms[1].start_time, waveforms[2].start_time )
            #return [None, None, None]

        assert 1./waveforms[0].sampling_rate == 1./waveforms[1].sampling_rate == 1./waveforms[2].sampling_rate, "Sampling rates do not match"
        # try:
        processed_signal_Z, processed_signal_N, processed_signal_E = self.processing_function.process_three_component_waveform(signalZ, signalN, signalE, 1./waveforms[0].sampling_rate)
        # except:
        #     print("High Corner Exceeds Nyquist")
        #     print("sampling rate", waveforms[0].sampling_rate)
        #     return None

        target_dt = self.processing_function.target_sampling_period
        t0 = waveforms[0].start_time
        n_samples = np.min([len(processed_signal_Z), len(processed_signal_N), len(processed_signal_E)])
        # Extract signal
        trace_start_index = int((pick_time + trace_cut_start - t0)/target_dt + 0.5)
        trace_end_index = trace_start_index + n_samples_out
        if (trace_start_index < 0 or trace_end_index >= n_samples):
            return [None, None, None]

        waveforms[0].sampling_rate = 1./target_dt
        waveforms[1].sampling_rate = 1./target_dt
        waveforms[2].sampling_rate = 1./target_dt
    
        if len(processed_signal_Z) >= n_samples_out and len(processed_signal_N) >= n_samples_out and len(processed_signal_E) >= n_samples_out:
            waveforms[0].signal = np.copy(processed_signal_Z[trace_start_index:trace_end_index])
            waveforms[1].signal = np.copy(processed_signal_N[trace_start_index:trace_end_index])
            waveforms[2].signal = np.copy(processed_signal_E[trace_start_index:trace_end_index])
        else:
            print("Not enough data. Returning None")
            return [None, None, None]

        waveforms[0].start_time = trace_cut_start
        waveforms[1].start_time = trace_cut_start
        waveforms[2].start_time = trace_cut_start

        return waveforms

    def create_waveform_df(self, catalog_file_name, phase_type):
        """ Overrides method from BaseGatherData. Filter by given phase type"""
        catalog_3c_df = pd.read_csv(catalog_file_name, dtype={'location': object})
        print("Original length of 3C catalog:", len(catalog_3c_df))
        # Require picks be of the specified phase type
        catalog_df = catalog_3c_df[catalog_3c_df['phase'] == phase_type]
        print("Length of 3C catalog:", len(catalog_3c_df))

        # Impute blank station codes
        catalog_df['location'].replace(["  "], "", regex=True, inplace=True)
        # Sort catalog
        catalog_df = catalog_df.sort_values(['evid', 'arrival_time'])
        print("Length of", phase_type, "catalog:", len(catalog_df))
        return catalog_df
        
    def make_archive(self, catalog_df, output_file_root, halfwidth, add_boxcar=False):
        """ Overrides method from BaseGatherData."""
        evids = catalog_df['evid'].values
        stations = catalog_df['station'].values
        networks = catalog_df['network'].values
        locations = catalog_df['location'].values
        arrival_times = catalog_df['arrival_time'].values

        # TODO: Check that N&1 go together and E&2
        channelsE = catalog_df["channel2"].values
        channelsN = catalog_df["channel1"].values
        channelsZ = catalog_df["channelz"].values
        print(stations, networks, channelsE, channelsN, channelsZ, locations)
        lfound = np.zeros(len(evids), dtype='bool')
        X = None
        k = 0
        for i in range(len(evids)):
            waveforms_ZNE = []
            for comp in range(3):
                if comp == 0:
                    channel = channelsZ[i]
                    assert channel[-1] == "Z"
                elif comp==1:
                    channel = channelsN[i]
                    assert channel[-1] == "N" or channel[-1] == "1"
                else:
                    channel = channelsE[i]
                    assert channel[-1] == "E" or channel[-1] == "2"

                exists = self.archive_manager.waveform_exists(evids[i],
                                                        networks[i], stations[i],
                                                        channel, locations[i])
                if (exists):
                    waveform = self.archive_manager.read_waveform(evids[i],
                                                            networks[i], stations[i],
                                                            channel, locations[i])
                    waveform.remove_trailing_zeros() # Clean up any junk at end of waveform
                    waveforms_ZNE.append(waveform)
                else:
                    print("Waveform: "
                        + networks[i] + "." + stations[i] + "."
                        + channel + "." + locations[i]
                        + " does not exist for event: ", evids[i])
                    break

            if len(waveforms_ZNE) == 3:
                waveforms = self.process_data(waveforms_ZNE, arrival_times[i], trace_cut_start=-1*halfwidth, trace_cut_end=halfwidth)
                waveform_Z = waveforms[0]
                waveform_N = waveforms[1]
                waveform_E = waveforms[2]
                if waveform_Z is None or waveform_N is None or waveform_E is None:
                    print("Insufficient data to process waveform: "
                        + networks[i] + "." + stations[i] + "." + channel + "."
                        + locations[i])
                    continue
                signal_Z = waveform_Z.signal
                signal_N = waveform_N.signal
                signal_E = waveform_E.signal
                if len(signal_E) != len(signal_N) != len(signal_Z):
                    print("Waveforms are not the same length. skipping...")
                    continue
                #assert len(signal_E) == len(signal_N) == len(signal_Z), "Waveforms are not the same length"
                n_samples = len(signal_Z)

                if (X is None):
                    X = np.zeros([len(evids), n_samples, 3], dtype='f4')

                X_ev = np.zeros([n_samples, 3], dtype='f4')
                X_ev[:, 0] = signal_E[:]
                X_ev[:, 1] = signal_N[:]
                X_ev[:, 2] = signal_Z[:]

                X[k, :, :] = np.copy(X_ev)
                k += 1
                lfound[i] = True
            else:
                print("There are not three waveforms")
                continue

        print("Read %d waveforms out of %d lines in dataframe (%.2f pct)"%(k, len(evids), float(k)/len(evids)*100))

        output_file_csv = output_file_root + ".csv"
        catalog_df = catalog_df[lfound]
        assert len(catalog_df) == np.sum(1*lfound), 'dataframe subsample failed'
        catalog_df.to_csv(output_file_csv, index=False)

        X = np.resize(X, [k, n_samples, 3])
        y = np.zeros(k, dtype = 'f4') + (n_samples/2.)*0.01
        
        if add_boxcar:
            # TODO: Don't hardcode boxcar widths
            y = boxcar.add_boxcar(catalog_df, {0: 21, 1: 31, 2: 51}, X, y, None)

        output_file_h5 = output_file_root + ".h5"
        Write.h5py_file(["X", "Y"], [X, y], output_file_h5)


    def process_and_save_waveforms(self, catalog_filename, phase_type, output_file_root, event_type, add_boxcar=False):
        """
        Write archived time-series and metadata to disk for specified waveforms

        Parameters
        ----------
        catalog_filename: string
            path to 3 component metadata csv file
        output_file_root: string
            path and name (not including file-type suffix) for the archive time-series and metadata files
        """
        catalog_df = self.create_waveform_df(catalog_filename, phase_type)
        catalog_df['event_type'] = event_type
        self.make_archive(catalog_df, output_file_root, add_boxcar=add_boxcar)