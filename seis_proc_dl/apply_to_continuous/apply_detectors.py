import obspy
from obspy.core.utcdatetime import UTCDateTime as UTC
from obspy.core.trace import Stats
import numpy as np
import logging
import torch
import os
# TODO: Better way to import pyuussmlmodels than adding path?
import sys
sys.path.append("/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/mlmodels/intel_cpu_build")
import pyuussmlmodels
from seis_proc_dl.utils.model_helpers import clamp_presigmoid_values
from seis_proc_dl.detectors.models.unet_model import UNetModel

class PhaseDetector():
    def __init__(self, 
                 model_to_load, 
                 num_channels, 
                 min_presigmoid_value=None, 
                 device="cuda:0"):
        #warnings.simplefilter("ignore")
        self.unet = None
        self.device = torch.device(device)
        self.min_presigmoid_value = min_presigmoid_value
        self.__init_model(num_channels, model_to_load)

    def __init_model(self, num_channels, model_to_load):
        self.unet = UNetModel(num_channels=num_channels, num_classes=1).to(self.device)        
        logging.info(f"Initialized {num_channels} comp unet with {self.get_n_params()} params...")
        assert os.path.exists(model_to_load), f"Model {model_to_load} does not exist"
        logging.info("Loading model:", model_to_load)
        check_point = torch.load(model_to_load)
        self.unet.load_state_dict(check_point['model_state_dict'])
        self.unet.eval()

    def apply_model_to_batch(self, X, lsigmoid=True, center_window=None):
        n_samples = X.shape[1] 
        X = torch.from_numpy(X.transpose((0, 2, 1))).float().to(self.device)
        
        with torch.no_grad():
            if (lsigmoid):
                model_output = self.unet.forward(X)
                if self.min_presigmoid_value is not None:
                    model_output = clamp_presigmoid_values(model_output, self.min_presigmoid_value)
                Y_est = torch.sigmoid(model_output)
            else:
                Y_est = self.unet.forward(X)
            
            # Specify axis=1 for case when batch==1
            Y_est = Y_est.squeeze(1)

            if center_window:
                j1 = int(n_samples/2 - center_window)
                j2 = int(n_samples/2 + center_window)
                Y_est = Y_est[:,j1:j2]

        return Y_est.to('cpu').detach().numpy()

    def get_n_params(self):
        return sum(p.numel() for p in self.unet.parameters() if p.requires_grad)

    def apply_to_continuous(self, continuous, batchsize=256, center_window=None):
        n_examples, n_samples, n_channels = continuous.shape
        if center_window is not None:
            n_samples = center_window*2
        # model output is N x S
        post_probs = np.zeros((n_examples, n_samples))
        batch_start = 0
        while batch_start < n_examples:
            batch_end = np.min([batch_start+batchsize, n_examples])
            batch = continuous[batch_start:batch_end, :, :]

            post_probs[batch_start:batch_end, :] = self.apply_model_to_batch(batch, center_window=center_window)

            batch_start += batchsize
        # Not going to flatten the post probs here, in case the center window is not used 
        return post_probs
    
    @staticmethod
    def flatten_post_probs(post_probs):
        return post_probs.flatten()

    @staticmethod
    def save_post_probs(outfile, post_probs, stats):
        assert post_probs.shape[0] == stats['npts'], "posterior probability is the wrong shape"
        st = obspy.Stream()
        tr = obspy.Trace()
        tr.data = (post_probs*100).astype(int)
        tr.stats = Stats(stats)
        st += tr
        st.write(outfile, format="MSEED")

    @staticmethod
    def trim_post_probs(post_probs, start_pad, end_pad, edge_n_samples):
        npts = post_probs.shape[0]
        # Start trim should always be edge_samples
        start_ind = np.max([0, start_pad-edge_n_samples])
        # I don't think end_pad will ever be less than edge_n_samples
        end_ind = np.min([npts, npts-(end_pad-edge_n_samples)])

        return post_probs[start_ind:end_ind]

class DataLoader():
    def __init__(self, store_N_seconds=0) -> None:
        self.continuous_data = None
        self.metadata = None
        self.gaps = None

        self.previous_continuous_data = None
        self.previous_endtime = None
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
                self.metadata['starttime'] = self.previous_endtime
                self.metadata['starttime_epoch'] = self.previous_endtime - UTC("19700101")
                self.metadata['npts'] = self.continuous_data.shape[0]
                self.metadata['previous_appended'] = True
            else:
                # TODO: Do something here, like "interpolate" if the traces are close enough
                logging.warning('Cannot concatenate previous days data, data is not continuous')

    def reset_loader(self):
        self.continuous_data = None
        self.metadata = None
        self.gaps = None

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

        # Indicate that no data has been prepended to the trace
        meta_data['previous_appended'] = False
        meta_data['original_starttime'] = stats['starttime']
        meta_data['original_starttime_epoch'] = stats['starttime'] - UTC("19700101")
        meta_data['original_npts'] = stats['npts']

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
        starttime = st[0].stats.starttime
        desired_start = UTC(starttime.year, starttime.month, starttime.day)
        desired_end = desired_start + expected_file_duration_s
        # TODO: If one of these assertions is thrown, update trimming code to occur
        # If the startime happens to be on the day before, it'll mess up the desired day
        assert starttime >= desired_start, "The stream begins on the previous day of interest"
        assert st[0].stats.endtime <= desired_end, "The end of the trace goes into the next day"

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
                logging.info("Caught obspy Incompatible Traces warning. Fixing the sampling rates...")
                delta = round(1/sampling_rate, 3)

                for tr in st:
                    tr.stats.sampling_rate = sampling_rate
                    tr.stats.delta = delta

                st.merge(fill_value='interpolate')

        # Make sure the trace is the desired length (fill in ends if needed) 
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

    def format_continuous_for_unet(self, unet_window_length, unet_sliding_interval, processing_function=None, normalize=True):
        # compute the indices for splitting the continuous data into model inputs
        npts, n_comps = self.continuous_data.shape
        # Always pad the start to avoid having to update the start time of the post probs
        pad_start = True # (self.store_N_seconds == 0 and self.previous_continuous_data is None)
        total_npts, start_pad_npts, end_pad_npts = self.get_padding(npts, unet_window_length, unet_sliding_interval, pad_start)
        n_windows = self.get_n_windows(total_npts, unet_window_length, unet_sliding_interval)
        window_start_indices = self.get_sliding_window_start_inds(total_npts, unet_window_length, unet_sliding_interval)

        # Extend those windows slightly, when possible to avoid edge effects
        # TODO: Add this in later if needed. It shouldn't be that important because of the relatively small 
        # window + taper size and the sliding windows

        data = self.add_padding(np.copy(self.continuous_data), start_pad_npts, end_pad_npts)

        # Process the extended windows
        formatted = np.zeros((n_windows, unet_window_length, n_comps))
        for w_ind in range(n_windows):
            i0= window_start_indices[w_ind]
            i1 = i0 + unet_window_length
            ex = np.copy(data[i0:i1, :])
            if processing_function is not None:
                ex = processing_function(ex)
            if normalize:
                ex = self.normalize_example(ex)

            formatted[w_ind, :, :] = ex

        return formatted, start_pad_npts, end_pad_npts

    @staticmethod
    def add_padding(data, start_pad_npts, end_pad_npts):
        n_comps = data.shape[1]
        if start_pad_npts > 0:
            start_padding = np.full((start_pad_npts, n_comps), data[0, :])
            data = np.concatenate([start_padding, data])
        if end_pad_npts > 0:
            end_padding = np.full((end_pad_npts, n_comps), data[-1, :])
            data = np.concatenate([data, end_padding])

        return data

    @staticmethod
    def process_1c_P(wf, desired_sampling_rate=100, normalize=True):
        processor = pyuussmlmodels.Detectors.UNetOneComponentP.Preprocessing()
        processed = processor.process(wf[:, 0], sampling_rate=desired_sampling_rate)[:, None]
        return processed
    
    def process_3c_P(self, wfs, desired_sampling_rate=100, normalize=True):
        processor = pyuussmlmodels.Detectors.UNetThreeComponentP.Preprocessing()
        processed = self._process_3c(wfs, processor, desired_sampling_rate)
        return processed
   
    def process_3c_S(self, wfs, desired_sampling_rate=100, normalize=True):
        processor = pyuussmlmodels.Detectors.UNetThreeComponentS.Preprocessing()
        processed = self._process_3c(wfs, processor, desired_sampling_rate)
        return processed
    
    @staticmethod
    def normalize_example(waveform):
        """Normalize one example. Each trace is normalized separately. 
        Args:
            waveform (np.array): waveform of size (# samples, # channels)

        Returns:
            np.array: Normalized waveform
        """
        # normalize the data for the window 
        norm_vals = np.max(abs(waveform), axis=0)
        norm_vals_inv = np.zeros_like(norm_vals, dtype=float)
        for nv_ind in range(len(norm_vals)):
            nv = norm_vals[nv_ind]
            if abs(nv) > 1e-4:
                norm_vals_inv[nv_ind] = 1/nv

        return waveform*norm_vals_inv
    
    @staticmethod
    def _process_3c(wfs, processor, desired_sampling_rate=100):
        east = wfs[:, 0]
        north = wfs[:, 1]
        vert = wfs[:, 2]
        proc_z, proc_n, proc_e = processor.process(vert, north, east, sampling_rate=desired_sampling_rate)
        processed = np.zeros_like(wfs)
        processed[:, 0] = proc_e
        processed[:, 1] = proc_n
        processed[:, 2] = proc_z

        return processed

    @staticmethod
    def get_padding(npts, unet_window_length, unet_sliding_interval, pad_start=True):
        # TODO: This edge_npts calc is not always right (works for my case though)
        # number points between the end of the center and the end of the window
        edge_npts = (unet_window_length-unet_sliding_interval)//2

        # Assume that if the previous day is included that the very start of the trace will be in the 
        # central window at some point. If not, I think adding edge_npts at the start will capture it
        start_pad_npts = 0
        if pad_start:
            start_pad_npts = edge_npts

        npts_padded = npts+start_pad_npts

        # Compute the padding at the end of the waveform to be evenly divisible by the sliding window
        end_pad_npts = unet_sliding_interval - (npts_padded - unet_window_length)%unet_sliding_interval
        # If the padding is less than edge_npts, then the last edge of the trace will not be included. This 
        # should be fine when the end is included in the next day but there may not always be a next day.
        if end_pad_npts < edge_npts:
            end_pad_npts += unet_sliding_interval

        npts_padded += end_pad_npts

        return npts_padded, start_pad_npts, end_pad_npts

    @staticmethod
    def get_sliding_window_start_inds(npts_padded, unet_window_length, unet_sliding_interval):
        return np.arange(0, npts_padded-unet_window_length+unet_sliding_interval, 
                         unet_sliding_interval)

    @staticmethod
    def get_n_windows(npts, window_length, sliding_interval):
        return (npts-window_length)//sliding_interval + 1
    
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
    