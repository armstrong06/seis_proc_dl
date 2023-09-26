import obspy
from obspy.core.utcdatetime import UTCDateTime as UTC
import numpy as np
import logging

class DataLoader():
    def __init__(self, store_N_samples=0) -> None:
        self.continuous_data = None
        self.metadata = None
        self.gaps = None

        # TODO: needs consideration for missing or removed days
        self.previous_continuous_data = None
        self.previous_endtime = None
        self.store_N_samples = store_N_samples

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
            return
        elif st_N is None:
            gaps_N[0][3] = gaps_N[0][3][0:2] + "?"
            self.gaps = gaps_N
            return
        elif st_Z is None:
            gaps_Z[0][3] = gaps_Z[0][3][0:2] + "?"
            self.gaps = gaps_Z
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

        if self.store_N_samples > 0: 
            self.prepend_data()

        gaps = gaps_E + gaps_N + gaps_Z

        self.continuous_data = cont_data
        self.gaps = gaps
        self.save_meta_data(st_E[0].stats)
        
        if self.store_N_samples > 0:
            self.prepend_previous_data()
            self.previous_continuous_data = cont_data[-self.store_N_samples:, :]
            self.previous_endtime = endtimes[0]

    def load_1c_data(self, file, min_signal_percent=1):

        self.reset_loader()

        st, gaps = self.load_channel_data(file, 
                                              min_signal_percent=min_signal_percent)
        if st is None:
            self.gaps = gaps
            return
        cont_data = np.zeros((st[0].stats.npts, 1))
        cont_data[:, 0] = st[0].data

        self.continuous_data = cont_data
        self.gaps = gaps
        self.save_meta_data(st[0].stats)

        if self.store_N_samples > 0:
            # Update continous data and the metadata to include the end of the previous trace
            self.prepend_previous_data()
            # Save the end of the current trace as the previous trace for nexttime
            self.previous_continuous_data = cont_data[-self.store_N_samples:, :]
            self.previous_endtime = endtimes[0]

    def prepend_data(self):
        new_starttime = self.metadata['starttime']
        if self.previous_data is not None:
            # The start of the current trace should be within one sample after the end of the previous trace
            if ((current_starttime - self.previous_endtime) < self.metadata['dt'] and (current_starttime - self.previous_endtime) > 0):
                self.cont_data = np.join([self.previous_data, cont_data])
                self.metadata['starttime'] = self.previous_endtime
            else:
                logging.warning('Cannot concatenate previous days data, data is not continuous')

    def reset_loader(self):
        self.continuous_data = None
        self.metadata = None
        self.gaps = None

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
            st = st.trim(starttime=desired_start, pad=True, fill_value=st[0].data[0])
        if end_gap is not None:
            gaps += [end_gap]
            # Fill the end of the trace with the last value, if needed
            st = st.trim(endtime=desired_end, pad=True, fill_value=st[0].data[-1])

        return st, gaps 

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

    @staticmethod
    def get_n_windows(npts, window_length, sliding_interval):
        return (npts-window_length)//sliding_interval + 1