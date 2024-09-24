import obspy
from obspy.core.utcdatetime import UTCDateTime as UTC
from obspy.core.trace import Stats
import numpy as np
import logging
import torch
import os
import re
import sys
import json
import datetime
import glob
import openvino as ov

sys.path.append("/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/mlmodels/intel_cpu_build")
# TODO: Better way to import pyuussmlmodels than adding path?
import pyuussmlmodels
from seis_proc_dl.utils.model_helpers import clamp_presigmoid_values
from seis_proc_dl.detectors.models.unet_model import UNetModel
from seis_proc_dl.utils.config_apply_detectors import Config

# Followed this tutorial https://betterstack.com/community/guides/logging/how-to-start-logging-with-python/
# Just a simple logger for now
logger = logging.getLogger("apply_detectors")
stdoutHandler = logging.StreamHandler(stream=sys.stdout)
fmt = logging.Formatter(
    "%(name)s: %(asctime)s | %(levelname)s | %(filename)s:%(lineno)s | %(process)d >>> %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
stdoutHandler.setFormatter(fmt)
logger.addHandler(stdoutHandler)
logger.setLevel(logging.DEBUG)

class ApplyDetector():
    def __init__(self, 
                 ncomps, 
                 config
                 ) -> None:
        """Use DataLoader and PhaseDetector to load and apply detectors to day chunks of seismic data.

        Args:
            ncomps (int): The number of components for the phase detector and data (1 or 3).
            config (object): dictionary defining information needed for the PhaseDetector and DataLoader,
            and path information

        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        self.p_detector = None
        self.s_detector = None
        # There's only a P processing func because need to specify 1 or 3 c
        self.p_proc_func = None
        self.ncomps = ncomps
        config = Config.from_json(config)
        self.dataloader = DataLoader(config.dataloader.store_N_seconds)
        self.data_dir = config.paths.data_dir
        self.outdir = config.paths.output_dir
        self.window_length = config.unet.window_length
        self.sliding_interval = config.unet.sliding_interval
        self.center_window = self.sliding_interval//2
        self.window_edge_npts = (self.window_length-self.sliding_interval)//2
        self.device = config.unet.device
        self.min_torch_threads = config.unet.min_torch_threads
        self.min_presigmoid_value = config.unet.min_presigmoid_value
        self.batchsize = config.unet.batchsize
        self.use_openvino = config.unet.use_openvino
        #self.expected_file_duration_s = config.dataloader.expected_file_duration_s
        self.min_signal_percent = config.dataloader.min_signal_percent

        if ncomps == 1:
            self.p_model_file = config.paths.one_comp_p_model
            self.s_model_file = None
            self.__init_1c()
        elif ncomps == 3:
            self.p_model_file = config.paths.three_comp_p_model
            self.s_model_file = config.paths.three_comp_s_model
            if self.s_model_file is None:
                raise ValueError("S Detector cannot be None for 3C")
            self.__init_3c()
        else:
            raise ValueError("Invalid number of components")

    ### Constructor without config ###
    # def __init__(self, 
    #              ncomps, 
    #              data_dir,
    #              p_model_file, 
    #              s_model_file=None,
    #              window_length=1008,
    #              sliding_interval=500,
    #              device='cpu',
    #              min_torch_threads=2,
    #              min_presigmoid_value=-70
    #              ) -> None:
        
    #     self.dataloader = DataLoader()
    #     self.p_detector = None
    #     self.s_detector = None
    #     self.p_proc_func = None

    #     self.data_dir = data_dir
    #     self.p_model_file = p_model_file
    #     self.s_model_file = s_model_file
    #     self.window_length = window_length
    #     self.sliding_interval = sliding_interval
    #     self.center_window = sliding_interval//2
    #     self.window_edge_npts = (window_length-sliding_interval)//2
    #     self.device = device
    #     self.min_torch_threads = min_torch_threads
    #     self.min_presigmoid_value = min_presigmoid_value

    #     if ncomps == 1:
    #         self.__init_1c()
    #     elif ncomps == 3:
    #         if s_model_file is None:
    #             raise ValueError("S Detector cannot be None for 3C")
    #         self.__init_3c()
    #     else:
    #         raise ValueError("Invalid number of components")

    def __init_1c(self):
        """Initialize the phase detector for 1 component P picker
        """
        self.p_detector = PhaseDetector(self.p_model_file,
                            1,
                            "P",
                            min_presigmoid_value=self.min_presigmoid_value,
                            device=self.device,
                            num_torch_threads=self.min_torch_threads)
        if self.use_openvino:
            self.p_detector.compile_openvino_model(self.window_length)
        self.p_proc_func = self.dataloader.process_1c_P

    def __init_3c(self):
        """Initialize the phase detectors for 3C P and S detectors
        """
        self.p_detector = PhaseDetector(self.p_model_file,
                            3,
                            "P",
                            min_presigmoid_value=self.min_presigmoid_value,
                            device=self.device,
                            num_torch_threads=self.min_torch_threads)
        if self.use_openvino:
            self.p_detector.compile_openvino_model(self.window_length)
        self.s_detector = PhaseDetector(self.s_model_file,
                            3,
                            "S",
                            min_presigmoid_value=self.min_presigmoid_value,
                            device=self.device,
                            num_torch_threads=self.min_torch_threads)
        if self.use_openvino:
            self.s_detector.compile_openvino_model(self.window_length)
        self.p_proc_func = self.dataloader.process_3c_P

    def apply_to_multiple_days(self, stat, chan, year, month, day, n_days, debug_N_examples=-1):
        """Apply the phase detector to multiple days of data for a single station/channel. Assumes data
        is stored in YYYY/MM/DD folder in the data directory specified in the config file. Writes output
        to YYYY/MM/DD folder in the output directory specified in the config file.

        Args:
            stat (str): Station name
            chan (str): Channel name, for 3C use "?" for the orientation or omit (e.g. HH? or HH)
            year (int): Year of start date
            month (int): month of start date
            day (int): date of start date
            n_days (int): number of days to iterate over
            debug_N_examples (int, optional): Number of waveform segments to pass to the phase detector, use 
            all inputs if -1. Defaults to -1.
        """
        
        assert year >= 2002 and year <= 2022, "Year is invalid"
        assert month > 0 and month < 13, "Month is invalid"
        assert day > 0 and day <= 31, "Day is invalid"
        assert n_days > 0, "Number of days is invalid"

        ### Set the start date and time increment of the files ###
        date = datetime.date(year, month, day)
        delta = datetime.timedelta(days=1)

        ### Iterate over the specified number of days ###
        for _ in range(n_days):
            ### The data files are organized Y/m/d, get the appropriate date/station files ###
            date_str = date.strftime("%Y/%m/%d")
            files = sorted(glob.glob(os.path.join(self.data_dir, date_str, f'*{stat}*{chan}*')))

            ### Make the output dirs have the same structure as the data dirs ###
            date_outdir = os.path.join(self.outdir, date_str)

            ### If there are no files for that station/day, move to the next day ###
            if len(files) == 0:
                logger.info(f'No data for {date_str} {stat} {chan}')
                continue
            elif (self.ncomps == 1 and len(files) != 1) or (self.ncomps ==3 and len(files) != 3):
                logger.warning(f"Incorrect number of files found for {date_str} {stat} {chan}")
                continue 

            self.apply_to_one_file(files, date_outdir, debug_N_examples=debug_N_examples)

            date += delta

    def apply_to_one_file(self, files, outdir=None, debug_N_examples=-1):
        """Process one miniseed file and write posterior probability and data information to disk. Runs both 
        the P and S detector for 3C data and just the P detector for 1C data. The output file names will contain the
        E/1 channel name for 3C data and the Z channel name for 1C.

        Args:
            files (list): The miniseed files to read in. Should contain 1 string for 1C and 3 for 3C - order E, N, Z. 
            outdir (str/path, optional): Directory to write to. If None, uses the output directory specified 
            in the config file. Defaults to None.
            debug_N_examples (int, optional): Number of waveform segments to pass to the phase detector, use 
            all inputs if -1. Defaults to -1.
        """
        if outdir is None:
            outdir = self.outdir
            
        if self.ncomps == 1:
            self.dataloader.load_1c_data(files[0], 
                                         min_signal_percent=self.min_signal_percent,)
                                         #expected_file_duration_s=self.expected_file_duration_s)
        else:
            self.dataloader.load_3c_data(files[0], files[1], files[2], 
                                         min_signal_percent=self.min_signal_percent,)
                                         #expected_file_duration_s=self.expected_file_duration_s)
        self.__apply_to_one_phase(self.p_detector, self.p_proc_func, 
                                  files[0], outdir, debug_N_examples=debug_N_examples)

        if self.ncomps == 3:
            self.__apply_to_one_phase(self.s_detector, self.dataloader.process_3c_S, 
                                      files[0], outdir, debug_N_examples=debug_N_examples)

        ### Save the station meta info (including gaps) to a file in the same dir as the post probs. ###
        ### Only need one file per station/day pair ###
        meta_outfile_name =  self.dataloader.make_outfile_name(files[0], outdir)
        self.dataloader.write_data_info(meta_outfile_name)

    def __apply_to_one_phase(self, detector, proc_func, file_for_name, outdir, debug_N_examples=-1):
        """Format continuous data and apply phase detector. Write posterior probabilities to disk.

        Args:
            detector (object): Initialized PhaseDetector with the appropriate model loaded.
            proc_func (object): The processing function to use when processing each waveform segment.
            file_for_name (str): Input file name used to derive the posterior probability output file name.
            outdir (str): Specify the directory to save the posterior probabilities to.
            debug_N_examples (int, optional):  Number of waveform segments to pass to the phase detector, use 
            all inputs if -1. Defaults to -1.
        """
        data, start_pad_npts, end_pad_npts = self.dataloader.format_continuous_for_unet(self.window_length,
                                                            self.sliding_interval,
                                                            proc_func,
                                                            normalize=True
                                                            )

        probs_outfile_name = detector.make_outfile_name(file_for_name, outdir)

        if debug_N_examples > 0:
            logger.debug("Reducing data to %d examples", debug_N_examples)
            data = data[0:debug_N_examples, :, :]
            # Update npts so an error does not get thrown in save_post_probs
            self.dataloader.metadata['npts'] = int(debug_N_examples*self.center_window*2)
            # Ensure don't trim the post probs 
            start_pad_npts, end_pad_npts = self.window_edge_npts, self.window_edge_npts

        cont_post_probs = detector.get_continuous_post_probs(data, 
                                                            self.center_window,
                                                            self.window_edge_npts,
                                                            batchsize = self.batchsize,
                                                            start_pad_npts=start_pad_npts, 
                                                            end_pad_npts=end_pad_npts)
        
        detector.save_post_probs(probs_outfile_name, cont_post_probs, self.dataloader.metadata)

class PhaseDetector():
    def __init__(self, 
                 model_to_load, 
                 num_channels, 
                 phase_type,
                 min_presigmoid_value=None, 
                 device="cpu",
                 num_torch_threads=2):
        """Load and apply UNet phase detectors. Save the posterior probabilities to disk.

        Args:
            model_to_load (str): File path to model weights to load. 
            num_channels (int): The number of input channels for the model (1 or 3).
            phase_type (str): The intended phase type of the model (P or S).
            min_presigmoid_value (int, optional): Value to clamp the output of the model prior to the sigmoid function.
             Goal is to help stabilize the model output. Defaults to None.
            device (str, optional): Name of the device for Pytorch to use. Defaults to "cpu".
            num_torch_threads (int, optional): Number of threads for Pytorch to use. Defaults to 2.
        """
        #warnings.simplefilter("ignore")
        self.phase_type = phase_type
        self.num_threads = num_torch_threads
        if num_torch_threads > 0:
            torch.set_num_threads(num_torch_threads)
        self.unet = None
        self.torch_device = torch.device(device)
        self.openvino_device = device.upper()
        self.min_presigmoid_value = min_presigmoid_value
        self.openvino_compiled = False
        self.__init_model(num_channels, model_to_load)
        self.num_channels = num_channels

    def __init_model(self, num_channels, model_to_load):
        """Load the model weights and put in eval mode.

        Args:
            num_channels (int): Number of input channels for the model.
            model_to_load (str): Path to the model weights.
        """
        # Load the torch model
        self.unet = UNetModel(num_channels=num_channels, num_classes=1, apply_last_sigmoid=True).to(self.torch_device)        
        logger.info(f"Initialized {num_channels} comp {self.phase_type} unet with {self.get_n_params()} params...")
        assert os.path.exists(model_to_load), f"Model {model_to_load} does not exist"
        logger.info(f"Loading model: {model_to_load}")
        check_point = torch.load(model_to_load, map_location=self.torch_device)
        self.unet.load_state_dict(check_point['model_state_dict'])
        self.unet.eval()

    def apply_model_to_batch(self, X, center_window=None):
        """Apply the UNet to one batch of data

        Args:
            X (np.array): Input to model, shape should be B x S x C.
            Defaults to True.
            center_window (int, optional): Number of samples (W) on either side of the center of the model output to 
            return. Used with sliding windows. If None, returns all S samples. Defaults to None.

        Returns:
            np.array: Posterior probabilities in shape (B x S) or (B x W).
        """
        n_samples = X.shape[1] 
        X = torch.from_numpy(X.transpose((0, 2, 1))).float().to(self.torch_device)
        
        with torch.no_grad():
            Y_est = self.unet.forward(X)
            
            # Specify axis=1 for case when batch==1
            Y_est = Y_est.squeeze(1)

            if center_window:
                j1 = int(n_samples/2 - center_window)
                j2 = int(n_samples/2 + center_window)
                Y_est = Y_est[:,j1:j2]

        # TODO: I think I can change this, don't need to send to cpu if already on cpu
        return Y_est.to('cpu').detach().numpy()

    def compile_openvino_model(self, window_length):
            # Example input shape
            input = torch.zeros((1, self.num_channels, window_length)).float()

            # Convert the model to openvino
            ov_model = ov.convert_model(self.unet, example_input=(input,))

            # Compile the model for the appropriate device
            core = ov.Core()
            n_threads = self.num_threads
            # Pytorch uses -1 to indicate using all threads while openvino uses 0
            if n_threads < 0:
                n_threads = 0
            ov_compiled_model = core.compile_model(ov_model, 
                                                   self.openvino_device,
                                                   config={ov.properties.inference_num_threads(): n_threads})
            self.unet = ov_compiled_model
            self.openvino_compiled = True

    def apply_openvino_model_to_batch(self, X, center_window=None):
            """Apply the UNet to one batch of data using openvino model

            Args:
                X (np.array): Input to model, shape should be B x S x C.
                Defaults to True.
                center_window (int, optional): Number of samples (W) on either side of the center of the model output to 
                return. Used with sliding windows. If None, returns all S samples. Defaults to None.

            Returns:
                np.array: Posterior probabilities in shape (B x S) or (B x W).
            """
            n_samples = X.shape[1] 
            # Model input expects float32, input X is float64
            X = ov.runtime.Tensor(X.transpose((0, 2, 1)).astype("float32"))

            Y_est = self.unet({0: X})[0]
            # Specify axis=1 for case when batch==1
            Y_est = np.squeeze(Y_est, axis=1)

            if center_window:
                j1 = int(n_samples/2 - center_window)
                j2 = int(n_samples/2 + center_window)
                Y_est = Y_est[:,j1:j2]

            return Y_est

    def get_n_params(self):
        """ Get the number of trainable parameters in the model"""
        
        if self.openvino_compiled:
            logger.warning('Can only return number of params for Torch model, not OpenVino')
            return None
        
        return sum(p.numel() for p in self.unet.parameters() if p.requires_grad)

    def apply(self, input, batchsize=256, center_window=None):
        """Apply the UNet to batches of data.

        Args:
            input (np.array): Data to apply model to. In format N x S x C. 
            batchsize (int, optional): Batch size to use. Defaults to 256.
            center_window (int, optional): Number of samples (W) on either side of the center of the model output to 
            return. Used with sliding windows. If None, returns all S samples. Defaults to None.

        Returns:
            np.array: Posterior probabilities in shape (N x S) or (N x W).
        """
        n_examples, n_samples, n_channels = input.shape
        if center_window is not None:
            n_samples = center_window*2
        # model output is N x S
        post_probs = np.zeros((n_examples, n_samples))
        batch_start = 0
        while batch_start < n_examples:
            batch_end = np.min([batch_start+batchsize, n_examples])
            batch = input[batch_start:batch_end, :, :]

            if not self.openvino_compiled:
                post_probs[batch_start:batch_end, :] = self.apply_model_to_batch(batch, 
                                                                                 center_window=center_window)
            else:
                post_probs[batch_start:batch_end, :] = self.apply_openvino_model_to_batch(batch, 
                                                                                          center_window=center_window)

            batch_start += batchsize

        # Not going to flatten the post probs here, in case the center window is not used 
        return post_probs
    
    def get_continuous_post_probs(self, input, center_window, window_edge_npts, 
                                  batchsize=256, start_pad_npts=0, end_pad_npts=0):
        """Wrapper function to get the continuous posterior probabilities when using a sliding window  

        Args:
            input (np.array): Data to apply model to. In format N x S x C.
            center_window (int): Number of samples (W) on either side of the center of the model output to 
            return. Used with sliding windows. If None, returns all S samples.
            window_edge_npts (int): Number of points on either end of the center window. 
            batchsize (int, optional):  Batch size to use. Defaults to 256.
            start_pad_npts (int, optional): Number of samples added to the beginning of the input, 
            which should not be returned. Defaults to 0.
            end_pad_npts (int, optional): Number of samples added to the end of the input, 
            which should not be returned. Defaults to 0.

        Returns:
            np.array: Continuous posterior probabilities in shape (X, ). Where X is the number of points in the
            continuous data.
        """
        unet_output = self.apply(input, center_window=center_window, batchsize=batchsize)
        cont_post_probs = self.flatten_model_output(unet_output)
        cont_post_probs = self.trim_post_probs(cont_post_probs, 
                                                start_pad_npts, 
                                                end_pad_npts,
                                                window_edge_npts)

        return cont_post_probs

    @staticmethod
    def flatten_model_output(post_probs):
        """Flatten the output of the model so the posterior probabilities are continuous. 

        Args:
            post_probs (np.array): Output of the model, shape (N x W)

        Returns:
            np.array: Flattened posterior probabilities in shape (X, ). X=NxW
        """
        return post_probs.flatten()

    @staticmethod
    def save_post_probs(outfile, post_probs, stats):
        """Write the continuous posterior probabilities to disk in miniseed format with 2 digits of precision.
        Posterior probabilities will be between 0 and 99.

        Args:
            outfile (str): Path and name of the output file.
            post_probs (np.array): Flattened (1D) posterior probabilities
            stats (object): Dictionary object containing the station information, start time, and number of points.
        """
        assert post_probs.shape[0] == stats['npts'], "posterior probability is the wrong shape"
        logger.debug("writing %s", outfile)
        st = obspy.Stream()
        tr = obspy.Trace()
        tr.data = (post_probs*100).astype(np.int16)
        tr.stats = Stats(stats)
        st += tr
        st.write(outfile, format="MSEED")
        
    @staticmethod
    def trim_post_probs(post_probs, start_pad, end_pad, edge_n_samples):
        """Remove ends of the flattened posterior probabilities that correspond to padding in the input data.
        Not totally sure if this function is needed in my code. 

        Args:
            post_probs (np.array): Flattened posterior probabilities in shape (X, )
            start_pad (int): Number of samples added to the start of the input data.
            end_pad (int): Number of samples added to the end of the input data.
            edge_n_samples (int): Number of samples on either end of the center window. 

        Returns:
            np.array: Trimmed posterior probabilities - should be the same length as the unpadded input.
        """
        assert len(post_probs.shape) == 1, "Post probs need to be flattened before trimming"
        
        # TODO: Remove this eventually => checking if I will ever actually need to call this fn
        if start_pad != 254 or end_pad != 254:
            logger.debug("Padding does not equal 254")

        npts = post_probs.shape[0]
        # Start trim should always be edge_samples
        start_ind = np.max([0, start_pad-edge_n_samples])
        # I don't think end_pad will ever be less than edge_n_samples
        end_ind = np.min([npts, npts-(end_pad-edge_n_samples)])

        return post_probs[start_ind:end_ind]

    def make_outfile_name(self, wf_filename, dir):
        """Given an input file name, create the output path for the posterior probabilities.
        Makes the output directory if it does not exist.

        Args:
            wf_filename (str): Input file name in format NET.STAT.LOC.CHAN__START__END.mseed
            dir (str): Path to write the output file to. 

        Returns:
            str: Output path and filename. Filename in format probs.PHASE__NET.STAT.LOC.CHAN__START__END.mseed.
        """
        # split = re.split(r"__|T", os.path.basename(wf_filename))
        # chan = "Z"
        # if self.num_channels > 1:
        #     chan = ""
        # post_prob_name = f"probs.{self.phase_type}__{split[0][0:-1]}{chan}__{split[1]}__{split[3]}.mseed"

        post_prob_name = f"probs.{self.phase_type}__{os.path.basename(wf_filename)}"

        if not os.path.exists(dir):
            logger.debug(f"Making directory {dir}")
            os.makedirs(dir)

        outfile = os.path.join(dir, post_prob_name)

        return outfile

class DataLoader():
    def __init__(self, store_N_seconds=0) -> None:
        """Load day long miniseed files. Format and process for the PhaseDetector. Intended to be used for one
        station/channel over multiple days.

        Args:
            store_N_seconds (int, optional): Number of seconds at the end of the current data to store for appending to 
            the start of the next days data. Defaults to 0.
        """
        self.continuous_data = None
        self.metadata = None
        self.gaps = None

        self.previous_continuous_data = None
        self.previous_endtime = None
        self.store_N_seconds = store_N_seconds

    def load_3c_data(self, fileE, fileN, fileZ, min_signal_percent=1, expected_file_duration_s=86400):
        """Load miniseed files for a 3C station.

        Args:
            fileE (str): Path to the E/1 component file. 
            fileN (str): Path to the N/2 component file. 
            fileZ (str): Path to the Z component file. 
            min_signal_percent (int, optional): Minimum percentage of signal that must be present in all channels.
              Defaults to 1.
            expected_file_duration_s (int, optional): The number of expected seconds in file. 
            Only implemented for 86400. Defaults to 86400.
        """

        self.reset_loader()

        # assert np.isin(re.split( "[.|__]", os.path.basename(fileE))[3], ["EHE", "EH1", "BHE", "BH1", "HHE"]), "E file is incorrect"
        # assert np.isin(re.split( "[.|__]", os.path.basename(fileN))[3], ["EHN", "EH2", "BHN", "BH2", "HHN"]), "N file is incorrect"
        # assert np.isin(re.split( "[.|__]", os.path.basename(fileZ))[3], ["EHZ", "BHZ", "HHZ"]), "Z file is incorrect"

        st_E, gaps_E = self.load_channel_data(fileE, 
                                              min_signal_percent=min_signal_percent,
                                              expected_file_duration_s=expected_file_duration_s)
        st_N, gaps_N = self.load_channel_data(fileN, 
                                              min_signal_percent=min_signal_percent,
                                              expected_file_duration_s=expected_file_duration_s)
        st_Z, gaps_Z = self.load_channel_data(fileZ, 
                                              min_signal_percent=min_signal_percent,
                                              expected_file_duration_s=expected_file_duration_s)

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

        assert np.isin(st_E[0].stats.channel, ["EHE", "EH1", "BHE", "BH1", "HHE"]), "E file is incorrect"
        assert np.isin(st_N[0].stats.channel, ["EHN", "EH2", "BHN", "BH2", "HHN"]), "N file is incorrect"
        assert np.isin(st_Z[0].stats.channel, ["EHZ", "BHZ", "HHZ"]), "Z file is incorrect"

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
        self.store_meta_data(st_E[0].stats)
        
        if self.store_N_seconds > 0:
            self.prepend_previous_data()

    def load_1c_data(self, file, min_signal_percent=1, expected_file_duration_s=86400):
        """Load miniseed file for a 1C station.

        Args:
            file (str): Path to the Z component file. 
            min_signal_percent (int, optional): Minimum percentage of signal that must be present in all channels.
              Defaults to 1.
            expected_file_duration_s (int, optional): The number of expected seconds in file. 
            Only implemented for 86400. Defaults to 86400.
        """

        self.reset_loader()

        st, gaps = self.load_channel_data(file, 
                                              min_signal_percent=min_signal_percent,
                                              expected_file_duration_s=expected_file_duration_s)
        if st is None:
            self.gaps = gaps
            self.reset_previous_day()
            return
        
        cont_data = np.zeros((st[0].stats.npts, 1))
        cont_data[:, 0] = st[0].data

        self.continuous_data = cont_data
        self.gaps = gaps
        self.store_meta_data(st[0].stats, three_channels=False)

        if self.store_N_seconds > 0:
            # Update continous data and the metadata to include the end of the previous trace
            # TODO: don't prepend previous data if the end of the previous day has been filled in because
            # of a gap
            self.prepend_previous_data()
   
    def format_continuous_for_unet(self, unet_window_length, unet_sliding_interval, processing_function=None, normalize=True):
        """Formats and processes the continuous data for input into a phase detector.

        Args:
            unet_window_length (int): The duration of the input to the model in samples.
            unet_sliding_interval (int): The length of the sliding interval in samples.
            processing_function (object, optional): The processing function to apply to each input of the model 
            separately. Defaults to None.
            normalize (bool, optional): Whether or not to normalize each input of the model. Defaults to True.

        Returns:
            tupple: (np.ndarray - Inputs for the phase detector (N x S x C), int - number of samples added to the start of the
            data, int - number of samples added to the end of the data)
        """
        # compute the indices for splitting the continuous data into model inputs
        npts, n_comps = self.continuous_data.shape
        # Always pad the start to avoid having to update the start time of the post probs
        pad_start = True # (self.store_N_seconds == 0 and self.previous_continuous_data is None)
        total_npts, start_pad_npts, end_pad_npts = self.get_padding(npts, unet_window_length, unet_sliding_interval, pad_start)
        n_windows = self.get_n_windows(total_npts, unet_window_length, unet_sliding_interval)
        window_start_indices = self.get_sliding_window_start_inds(total_npts, unet_window_length, unet_sliding_interval)
        
        assert n_windows == window_start_indices.shape[0]
        
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
    def process_1c_P(wf, desired_sampling_rate=100):
        """Wrapper around pyuussmlmodels 1C UNet preprocessing function for use in format_continuous_for_unet.

        Args:
            wf (np.array): 1, 1C waveform (S, 1)
            desired_sampling_rate (int, optional): Desired sampling rate for wf. Defaults to 100.

        Returns:
            np.array: Processed wf (S, 1)
        """
        assert wf.shape[1] == 1, "Incorrect number of channels"
        processor = pyuussmlmodels.Detectors.UNetOneComponentP.Preprocessing()
        processed = processor.process(wf[:, 0], sampling_rate=desired_sampling_rate)[:, None]
        return processed
    
    def process_3c_P(self, wfs, desired_sampling_rate=100):
        """Wrapper around pyuussmlmodels 3C P UNet preprocessing function for use in format_continuous_for_unet.

        Args:
            wf (np.array): 1, 3C waveform (S, 3)
            desired_sampling_rate (int, optional): Desired sampling rate for wf. Defaults to 100.

        Returns:
            np.array: Processed wf (S, 3)
        """
        assert wfs.shape[1] == 3, "Incorrect number of channels"
        processor = pyuussmlmodels.Detectors.UNetThreeComponentP.Preprocessing()
        processed = self._process_3c(wfs, processor, desired_sampling_rate)
        return processed
   
    def process_3c_S(self, wfs, desired_sampling_rate=100):
        """Wrapper around pyuussmlmodels 3C S UNet preprocessing function for use in format_continuous_for_unet.

        Args:
            wf (np.array): 1, 3C waveform (S, 3)
            desired_sampling_rate (int, optional): Desired sampling rate for wf. Defaults to 100.

        Returns:
            np.array: Processed wf (S, 3)
        """
        assert wfs.shape[1] == 3, "Incorrect number of channels"
        processor = pyuussmlmodels.Detectors.UNetThreeComponentS.Preprocessing()
        processed = self._process_3c(wfs, processor, desired_sampling_rate)
        return processed
    
    def prepend_previous_data(self):
        """Add previous days data to the start of the current days data and update the metadata.
        """
        current_starttime = self.metadata['starttime']
        if self.previous_continuous_data is not None:
            # The start of the current trace should be very close to the end of the previous trace
            # Allow a little tolerance, but not sure if that is okay (half a sample)
            # Obspy seems to ignore gaps of 0.0119 s (~1.2 samples)
            if ((current_starttime - self.previous_endtime) < self.metadata['dt']*1.5 and 
                (current_starttime - self.previous_endtime) > 0):
                self.continuous_data = np.concatenate([self.previous_continuous_data, self.continuous_data])
                self.metadata['starttime'] = self.metadata['starttime'] - self.store_N_seconds
                self.metadata['starttime_epoch'] = self.metadata['starttime'] - UTC("19700101")
                self.metadata['npts'] = self.continuous_data.shape[0]
                self.metadata['previous_appended'] = True
            else:
                # TODO: Do something here, like "interpolate" if the traces are close enough
                logger.warning('Cannot concatenate previous days data, data is not continuous')

    def reset_loader(self):
        """Reset the current continuous data and it's metadata and update the stored previous data.
        """
        if self.store_N_seconds > 0 and self.continuous_data is not None:
            # Update previous day
            store_N_samples = int(self.store_N_seconds*self.metadata['sampling_rate'])
            self.previous_continuous_data = self.continuous_data[-store_N_samples:, :]
            self.previous_endtime = self.metadata['original_endtime']
        self.continuous_data = None
        self.metadata = None
        self.gaps = None

    def reset_previous_day(self):
        """Clear the data stored for the previous day.
        """
        self.previous_continuous_data = None
        self.previous_endtime = None

    def store_meta_data(self, stats, three_channels=True):
        """Store the relevant miniseed metadata for use later. The orientation of the 3C channel 
        information is replaced with '?'.

        Args:
            stats (object): The Obspy Stats associated with the miniseed file
            three_channels (bool, optional): True for 3 channels stations, False for 1 channel. Defaults to True.
        """
        meta_data = {}
        meta_data['network'] = stats['network']
        meta_data['station'] = stats['station']
        chan = stats['channel']
        if three_channels:
            chan = f'{chan[0:2]}?'
        meta_data['channel'] = chan
        meta_data["sampling_rate"] = stats['sampling_rate']
        meta_data['dt'] = stats['delta']
        meta_data['starttime'] = stats['starttime']
        # Don't store 'endtime' because readonly field in obspy 
        #meta_data['endtime'] = stats['endtime']
        meta_data['npts'] = stats['npts']

        # Keep epoch time for convenience later, backup in case isoformat str format
        #  can't be read in by other application 
        meta_data['starttime_epoch'] = stats['starttime'] - UTC("19700101")
        meta_data['endtime_epoch'] = stats['endtime'] - UTC("19700101")

        # Indicate that no data has been prepended to the trace
        # Store the original information - original_starttime and orignal_npts since starttime
        # and npts will change if previous data is prepended. Original_endtime so I can make 
        # sure the endtime in the final post probs file matches, since endtime will be updated 
        # by obspy based on the starttime and npts.
        meta_data['previous_appended'] = False
        meta_data['original_starttime'] = stats['starttime']
        meta_data['original_endtime'] = stats['endtime']
        meta_data['original_starttime_epoch'] = stats['starttime'] - UTC("19700101")
        meta_data['original_endtime_epoch'] = stats['endtime'] - UTC("19700101")
        meta_data['original_npts'] = stats['npts']

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

        logger.debug("loading %s", file)
        # use obspy to load the miniseed file(s)
        st = obspy.read(file)

        sampling_rate = round(st[0].stats.sampling_rate)
        # TODO: this only works for days, not hours - fine for me
        if expected_file_duration_s != 24*60*60:
            raise NotImplementedError("DataLoader only works for day long files at the moment...")
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
            logger.warning(f"{os.path.basename(file)} does not have enough data, skipping")
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
                logger.info("Caught obspy Incompatible Traces warning. Fixing the sampling rates...")
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

    def write_data_info(self, outpath):
        """Write the trace metadata and gap information to a json file. For each gap, 
        there is a list containing the channel of the gap, the gap starttime and endtime in epoch format, the duration of 
        the gap in samples, and the sample corresponding to the start and end of the gap in 
        the posterior probability file.  

        Args:
            filename (str): The name of the file
            dir (str): The directory to store the file
        """
        all_gap_info = []
        for gap in self.gaps:
            gap_info = []
            start_ind = int((gap[4]-self.metadata['starttime'])*self.metadata['sampling_rate'])
            end_ind = start_ind + gap[-1]
            gap_info += [gap[3],
                         gap[4] - UTC('1970-01-01'), 
                         gap[5] - UTC('1970-01-01'), 
                         gap[-1], 
                         start_ind, 
                         end_ind]
            all_gap_info.append(gap_info)
        
        # Can't use json as is because of UTCDateTime object - maybe make them strings?
        # Like json because it is supported by other languages, not just python
        gap_dict = self.metadata.copy()
        for key in gap_dict.keys():
            if "time" in key and 'epoch' not in key:
                gap_dict[key] = gap_dict[key].isoformat()

        gap_dict['gaps'] = all_gap_info
        #outpath = os.path.join(dir, filename)
        logger.debug("writing %s", outpath)
        with open(outpath, 'w') as fp:
            json.dump(gap_dict, fp, sort_keys=True, 
                      indent=4, ensure_ascii=False)

    @staticmethod
    def add_padding(data, start_pad_npts, end_pad_npts):
        """Add padding to the ends of the continuous data. The first value is used to pad the front of the 
        data and the last value is used to pad the end of the data.

        Args:
            data (np.array): Continuous data.
            start_pad_npts (int): Number of samples to add to the start of the data.
            end_pad_npts (int): Number of samples to add to the end of the data.

        Returns:
            np.array: Padded data.
        """
        n_comps = data.shape[1]
        if start_pad_npts > 0:
            start_padding = np.full((start_pad_npts, n_comps), data[0, :])
            data = np.concatenate([start_padding, data])
        if end_pad_npts > 0:
            end_padding = np.full((end_pad_npts, n_comps), data[-1, :])
            data = np.concatenate([data, end_padding])

        return data

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
        """Process 3C data using pyuussmlmodels 3C UNet Preprocessing

        Args:
            wfs (np.array): Waveform to process (S, 3)
            processor (object): The pyuussmlmodels function to use. 
            desired_sampling_rate (int, optional): Desired sampling rate of the waveform. Defaults to 100.

        Returns:
            np.array: Processed waveform (S, 3)
        """
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
        """Compute how much padding to add to either end of the data when using a sliding window so every window 
        will be full and every sample of data will be in the central part of the window at some point.

        Args:
            npts (int): The original number of samples
            unet_window_length (int): The length of the unet input in samples
            unet_sliding_interval (int): the length of the sliding window to use
            pad_start (bool, optional): Whether to pad the start or not. Defaults to True.

        Returns:
            tupple: (int - the total number of points after padding, int - the number of points to add to the start, 
            int - the number of points to add to the end)
        """
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
        """Get the start indicies of sliding windows.

        Args:
            npts_padded (int): The total number of points after padding.
            unet_window_length (int): The length of the unet input in samples
            unet_sliding_interval (int): the length of the sliding window to use

        Returns:
            np.array: np array of the starting indicies
        """
        return np.arange(0, npts_padded-unet_window_length+unet_sliding_interval, 
                         unet_sliding_interval)

    @staticmethod
    def get_n_windows(npts, window_length, sliding_interval):
        """Count the number of sliding windows 

        Args:
            npts (_type_):  The total number of points after padding.
            window_length (int): The length of the unet input in samples
            sliding_interval (int): the length of the sliding window to use

        Returns:
            int: The number of sliding windows.
        """
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
    
    @staticmethod
    def make_outfile_name(wf_filename, dir):
        """Given an input file name, create the output path for the waveform metadata and gaps.
        Makes the output directory if it does not exist.

        Args:
            wf_filename (str): Input file name in format NET.STAT.LOC.CHAN__START__END.mseed
            dir (str): Path to write the output file to. 

        Returns:
            str: Output path and filename. Filename in format NET.STAT.LOC.CHAN__START__END.json.
        """
        post_prob_name = f'{os.path.basename(wf_filename).split(".mseed")[0]}.json'

        # if num_channels > 1:
        #     split = re.split(r"__|T", os.path.basename(wf_filename))
        #     post_prob_name = f"{split[0][0:-1]}__{split[1]}__{split[3]}.json"

        if not os.path.exists(dir):
            logger.debug(f"Making directory {dir}")
            os.makedirs(dir)

        outfile = os.path.join(dir, post_prob_name)

        return outfile