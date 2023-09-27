import obspy
from obspy.core.utcdatetime import UTCDateTime as UTC
import numpy as np
import logging
# TODO: Better way to import pyuussmlmodels than adding path?
import sys
sys.path.append("/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/mlmodels/intel_cpu_build")
import pyuussmlmodels

class PhaseDetector():
    def __init__(self, window_length, sliding_interval) -> None:
        self.sliding_interval = sliding_interval
        self.window_length = window_length

    def format_continuous(self, continuous_data, pad_start=False):
        # compute the indices for splitting the continuous data into model inputs
        npts = continuous_data.shape[0]
        # number points between the end of the center and the end of the window
        edge_npts = (self.window_length-self.sliding_interval)//2

        # Assume that if the previous day is included that the very start of the trace will be in the 
        # central window at some point. If not, I think adding one sliding interval at the start will capture it
        start_pad = 0
        if pad_start:
            start_pad = self.sliding_interval

        # Compute the padding at the end of the waveform to be evenly divisible by the sliding window
        end_pad = self.sliding_interval - (npts - self.window_length)%self.sliding_interval
        # If the padding is less than edge_npts, then the last edge of the trace will not be included. This 
        # should be fine when the end is included in the next day but there may not always be a next day.
        if end_pad < edge_npts:
            end_pad += self.window_length//2

        npts_padded = npts + start_pad + end_pad
        n_windows = self.get_n_windows(npts_padded, self.window_length, self.sliding_interval)
        window_start_indices = np.arange(0, npts_padded-2*self.sliding_interval, self.sliding_interval)


        # Extend those windows slightly, when possible to avoid edge effects

        # Process the extend windows

        # Trim the windows back down to the desired size

        # Return array of examples
        pass

    def get_n_windows(self, npts):
        return (npts-self.window_length)//self.sliding_interval + 1

class DataLoader():
    def __init__(self, store_N_seconds=0) -> None:
        self.continuous_data = None
        self.metadata = None
        self.gaps = None
        self.continuous_data_includes_previous = False

        self.previous_continuous_data = None
        self.previous_endtime = None
        # TODO: this should be in seconds, if using before resampling
        self.store_N_seconds = store_N_seconds

    def load_3c_data(self, fileE, fileN, fileZ, min_signal_percent=1):

        self.reset_loader()

        st_E, gaps_E = self.load_channel_data(fileE, 
                                              min_signal_percent=min_signal_percent)
        st_N, gaps_N = self.load_channel_data(fileN, 
                                              min_signal_percent=min_signal_percent)
        st_Z, gaps_Z = self.load_channel_data(fileZ, 
                                              min_signal_percent=min_signal_percent)

        # If one of the channels was skipped, return the entire day as a gap
        #TODO: IF the vertical comp still has enough signal, should I keep it?
        # When an entire day is removed, remove orientation from gap channel info
        if st_E is None:
            gaps_E[0][3] = gaps_E[0][3][0:2] + "?"
            self.gaps = gaps_E
            # If skipping a day, then there is no previous day for the next trace
            self.reset_previous_day()
            return
        elif st_N is None:
            gaps_N[0][3] = gaps_N[0][3][0:2] + "?"
            self.gaps = gaps_N
            self.reset_previous_day()
            return
        elif st_Z is None:
            gaps_Z[0][3] = gaps_Z[0][3][0:2] + "?"
            self.gaps = gaps_Z
            self.reset_previous_day()
            return

        starttimes = [st_E[0].stats.starttime, st_N[0].stats.starttime, st_Z[0].stats.starttime]
        endtimes = [st_E[0].stats.endtime, st_N[0].stats.endtime, st_Z[0].stats.endtime]
        npts = [st_E[0].stats.npts, st_N[0].stats.npts, st_Z[0].stats.npts]
        dt = st_N[0].stats.delta

        # TODO: handle the failures in some way
        assert abs(starttimes[0] - starttimes[1]) < dt
        assert abs(starttimes[1] - starttimes[2]) < dt
        assert abs(endtimes[0] - endtimes[1]) < dt
        assert abs(endtimes[1] - endtimes[2]) < dt
        assert len(np.unique(npts)) == 1

        cont_data = np.zeros((npts[0], 3))
        cont_data[:, 0] = st_E[0].data
        cont_data[:, 1] = st_N[0].data
        cont_data[:, 2] = st_Z[0].data

        gaps = gaps_E + gaps_N + gaps_Z

        self.continuous_data = cont_data
        self.gaps = gaps
        self.save_meta_data(st_E[0].stats)
        
        if self.store_N_seconds > 0:
            self.prepend_previous_data()
            store_N_samples = int(self.store_N_seconds*self.metadata['sampling_rate'])
            self.previous_continuous_data = cont_data[-store_N_samples:, :]
            self.previous_endtime = endtimes[0]

    def load_1c_data(self, file, min_signal_percent=1):

        self.reset_loader()

        st, gaps = self.load_channel_data(file, 
                                              min_signal_percent=min_signal_percent)
        if st is None:
            self.gaps = gaps
            self.reset_previous_day()
            return
        
        cont_data = np.zeros((st[0].stats.npts, 1))
        cont_data[:, 0] = st[0].data

        self.continuous_data = cont_data
        self.gaps = gaps
        self.save_meta_data(st[0].stats)

        if self.store_N_seconds > 0:
            # Update continous data and the metadata to include the end of the previous trace
            self.prepend_previous_data()
            # Save the end of the current trace as the previous trace for next time
            store_N_samples = int(self.store_N_seconds*self.metadata['sampling_rate'])
            self.previous_continuous_data = cont_data[-store_N_samples:, :]
            self.previous_endtime = st[0].stats.endtime

    def prepend_previous_data(self):
        current_starttime = self.metadata['starttime']
        if self.previous_continuous_data is not None:
            # The start of the current trace should be very close to the end of the previous trace
            # Allow a little tolerance, but not sure if that is okay (half a sample)
            # Obspy seems to ignore gaps of 0.0119 s (~1.2 samples)
            if ((current_starttime - self.previous_endtime) < self.metadata['dt']*1.5 and 
                (current_starttime - self.previous_endtime) > 0):
                self.continuous_data = np.concatenate([self.previous_continuous_data, self.continuous_data])
                # TODO: should make this another field in the metadata
                self.metadata['starttime'] = self.previous_endtime
                self.continuous_data_includes_previous = True
            else:
                # TODO: Do something here, like "interpolate" if the traces are close enough
                logging.warning('Cannot concatenate previous days data, data is not continuous')

    def reset_loader(self):
        self.continuous_data = None
        self.metadata = None
        self.gaps = None
        self.continuous_data_includes_previous = False

    def reset_previous_day(self):
        self.previous_continuous_data = None
        self.previous_endtime = None

    def save_meta_data(self, stats, three_channels=True):
        meta_data = {}
        meta_data["sampling_rate"] = stats['sampling_rate']
        meta_data['dt'] = stats['delta']
        meta_data['starttime'] = stats['starttime']
        meta_data['endtime'] = stats['endtime']
        meta_data['npts'] = stats['npts']
        meta_data['network'] = stats['network']
        meta_data['station'] = stats['station']
        meta_data['starttime_epoch'] = stats['starttime'] - UTC("19700101")
        meta_data['endtime_epoch'] = stats['endtime'] - UTC("19700101")
        chan = stats['channel']
        if three_channels:
            chan = f'{chan[0:2]}?'
        meta_data['channel'] = chan

        self.metadata = meta_data

    def load_channel_data(self, file, min_signal_percent=1, expected_file_duration_s=86400):
        """Reads in a miniseed file and check for gaps. Gaps are interpolated. 

        Args:
            file (str): Path to the miniseed file to read in.
            min_signal_percent (int, optional): Minimum percent signal required to process the file. Defaults to 1.
            max_duration_s (int, optional): Expected duration of the miniseed file in seconds. Defaults to 86400.

        Returns:
        tupple: (Obspy Stream, list of gaps)
        """ 

        # use obspy to load the miniseed file(s)
        st = obspy.read(file)

        sampling_rate = round(st[0].stats.sampling_rate)
        # TODO: this only works for days, not hours - fine for me
        # TODO: If the startime happens to be on the day before, it'll mess this up
        starttime = st[0].stats.starttime
        desired_start = UTC(starttime.year, starttime.month, starttime.day)
        desired_end = desired_start + expected_file_duration_s

        # If there is not enough signal in this day, skip the day
        total_npts = np.sum([st[i].stats.npts for i in range(len(st))])
        max_npts = expected_file_duration_s*round(sampling_rate)
        if (total_npts/max_npts)*100 < min_signal_percent:
            # Return the entire file period as a gap
            return None, [self.format_edge_gaps(st, desired_start, desired_end, entire_file=True)]

        # Save gaps so I know to ignore any detections in that region later
        # Don't save gaps if they are smaller than 5 samples for 100 Hz, 2 samples for 40 Hz
        gaps = st.get_gaps(min_gap=0.05)
        # Still need to interpolate the gaps if they are less than min_gap
        if len(st.get_gaps()) > 0:
            # Sometimes the gaps have different sampling rates and delta values, not totally sure why. 
            # The sampling rates do not equal npts/duration for the gap either.
            try:
                st.merge(fill_value='interpolate')
            except:
                delta = round(1/sampling_rate, 3)

                for tr in st:
                    tr.stats.sampling_rate = sampling_rate
                    tr.stats.delta = delta

                st.merge(fill_value='interpolate')

        # Make sure the trace is the desired length (fill in ends if needed) TODO: is it possible that the files are too long?
        # Check for gaps at the start/end of the day and save if they exist
        start_gap, end_gap = self.format_edge_gaps(st, desired_start, desired_end)
        if start_gap is not None:
            gaps.insert(0, start_gap)
            # Fill the start of the trace with the first value, if needed
            st = st.trim(starttime=desired_start, pad=True, 
                         fill_value=st[0].data[0], nearest_sample=False)
        if end_gap is not None:
            gaps += [end_gap]
            # Fill the end of the trace with the last value, if needed
            # set nearest_sample to False or it will extent to the next day 
            # and have 8460001 samples
            st = st.trim(endtime=desired_end, pad=True, 
                         fill_value=st[0].data[-1], nearest_sample=False)

        return st, gaps 

    def preprocess_3c_p(self, n_chunks=24):
        assert self.continuous_data.shape[1] == 3, "Not 3C data"
        preprocessor = pyuussmlmodels.Detectors.UNetThreeComponentP.Preprocessing()
        return self._preprocess_continuous(preprocessor, n_chunks)

    def preprocess_3c_s(self, n_chunks=24):
        assert self.continuous_data.shape[1] == 3, "Not 3C data"
        preprocessor = pyuussmlmodels.Detectors.UNetThreeComponentS.Preprocessing()
        return self._preprocess_continuous(preprocessor, n_chunks)

    def preprocess_1c_p(self, n_chunks=24):
        assert self.continuous_data.shape[1] == 1, "Not 1C data"
        preprocessor = pyuussmlmodels.Detectors.UNetOneComponentP.Preprocessing()
        return self._preprocess_continuous(preprocessor, n_chunks)
    
    def _preprocess_continuous(self, preprocessor, n_chunks):
        npts = self.continuous_data.shape[0]
        n_comps = self.continuous_data.shape[1]
        chunk_npts = npts//n_chunks
        # TODO: This won't work if resampling
        processed_continuous = np.zeros_like(self.continuous_data)
        i0 = 0
        for i in range(n_chunks):
            i1 = np.min([i0+chunk_npts, npts])
            if self.store_N_seconds > 0 and self.continuous_data_includes_previous:
                i1 += int(self.metadata['sampling_rate']*self.store_N_seconds)

            if n_comps == 3:
                E = np.copy(self.continuous_data[i0:i1, 0])
                N = np.copy(self.continuous_data[i0:i1, 1])
                Z = np.copy(self.continuous_data[i0:i1, 2])
                Z_proc, N_proc, E_proc, = preprocessor.process(Z, N, E, sampling_rate=100)
                processed_continuous[i0:i1, 0] = E_proc
                processed_continuous[i0:i1, 1] = N_proc
                processed_continuous[i0:i1, 2] = Z_proc
            else:
                Z = np.copy(self.continuous_data[i0:i1, 0])
                Z_proc = preprocessor.process(Z, sampling_rate=100)
                processed_continuous[i0:i1, :] = Z_proc[:, None]

            i0 += chunk_npts

        return processed_continuous

    @staticmethod
    def format_edge_gaps(st, desired_start, desired_end, entire_file=False):
        """Checks for gaps greater than 1 second at the start and end of an Obspy Stream.
        Only looks at the first trace in the stream. *The desired endtime may be off by ~1 second*
        If a gap exists returns a list in the format of Obspy's get_gaps(), otherwise returns None.

        Args:
            st (Obspy Stream): stream to check for gaps
            max_duration_s (int): expected duration of the file in seconds
            entire_file (boolean, optional): True if the gap is the entire file. Defaults to False.

        Returns:
            tupple or list: tupple of gap info lists for the beginning and end of the trace or a list for the entire file gap. 
        """
        starttime = st[0].stats.starttime
        endtime = st[0].stats.endtime 
        sampling_rate = round(st[0].stats.sampling_rate)

        if entire_file:
            max_duration_s = desired_end - desired_start
            return [st[0].stats.network, st[0].stats.station, st[0].stats.location, st[0].stats.channel, 
                        desired_start, desired_end, max_duration_s, int(max_duration_s*sampling_rate)]

        # Compute duration of gaps at the start and end
        start_delta = starttime - desired_start
        end_delta = desired_end - endtime

        start_gap = None
        if start_delta > 1:
            start_gap = [st[0].stats.network, st[0].stats.station, st[0].stats.location, st[0].stats.channel, 
                        desired_start, starttime, start_delta, int(start_delta*sampling_rate)]

        end_gap = None
        if end_delta > 1:
            end_gap = [st[0].stats.network, st[0].stats.station, st[0].stats.location, st[0].stats.channel,
                        endtime, desired_end, end_delta, int(end_delta*sampling_rate)]

        return start_gap, end_gap
    
    # def preprocess_1c_P(self, n_chunks=24):
    #     preprocessor = pyuussmlmodels.Detectors.UNetOneComponentP.Preprocessing()
    #     npts = self.continuous_data.shape[0]
    #     chunk_npts = npts//n_chunks
    #     processed_continuous = np.zeros_like(self.continuous_data)
    #     i0 = 0
    #     for i in range(n_chunks):
    #         i1 = np.min([i0+chunk_npts, npts])
    #         if self.store_N_seconds > 0 and self.continuous_data_includes_previous:
    #             i1 += int(self.metadata['sampling_rate']*self.store_N_seconds)

    #         Z = np.copy(self.continuous_data[i0:i1, 0])
    #         Z_proc, = preprocessor.process(Z, sampling_rate=100)
    #         processed_continuous[i0:i1, :] = Z_proc
        
    #     return processed_continuous
    