import obspy
from obspy.core.utcdatetime import UTCDateTime as UTC
from obspy.core.trace import Stats
from pathlib import Path
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
import openvino.properties.hint as hints
import openvino.properties as props
import time

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

        if self.use_openvino:
            self.use_async = config.unet.use_async

        self.post_probs_file_type = config.unet.post_probs_file_type
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

    def __init_1c(self):
        """Initialize the phase detector for 1 component P picker
        """
        self.p_detector = PhaseDetector(self.p_model_file,
                            1,
                            "P",
                            min_presigmoid_value=self.min_presigmoid_value,
                            device=self.device,
                            num_torch_threads=self.min_torch_threads,
                            post_probs_file_type=self.post_probs_file_type)
        if self.use_openvino:
            self.p_detector.compile_openvino_model(self.window_length, 
                                                   self.batchsize, 
                                                   self.use_async)
        self.p_proc_func = self.dataloader.process_1c_P

    def __init_3c(self):
        """Initialize the phase detectors for 3C P and S detectors
        """
        self.p_detector = PhaseDetector(self.p_model_file,
                            3,
                            "P",
                            min_presigmoid_value=self.min_presigmoid_value,
                            device=self.device,
                            num_torch_threads=self.min_torch_threads,
                            post_probs_file_type=self.post_probs_file_type)
        if self.use_openvino:
            self.p_detector.compile_openvino_model(self.window_length, 
                                                   self.batchsize, 
                                                   self.use_async)
        self.s_detector = PhaseDetector(self.s_model_file,
                            3,
                            "S",
                            min_presigmoid_value=self.min_presigmoid_value,
                            device=self.device,
                            num_torch_threads=self.min_torch_threads,
                            post_probs_file_type=self.post_probs_file_type)
        
        if self.use_openvino:
            self.s_detector.compile_openvino_model(self.window_length, 
                                                   self.batchsize, 
                                                   self.use_async)
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
        
        assert year >= 2002 and year <= 2023, "Year is invalid"
        assert month > 0 and month < 13, "Month is invalid"
        assert day > 0 and day <= 31, "Day is invalid"
        assert n_days > 0, "Number of days is invalid"

        ### Set the start date and time increment of the files ###
        date = datetime.datetime(year, month, day)
        delta = datetime.timedelta(days=1)

        # If using 1 comp, make sure Z is specified in the channel
        if self.ncomps == 1 and ((len(chan) == 2) or ("?" in chan)):
            chan = chan[0:2] + "Z"

        stat_startdate, stat_enddate = self.get_station_dates(year, stat, chan)
        date, n_days = self.validate_date_range(stat_startdate, 
                                                stat_enddate,
                                                date, 
                                                n_days)
        if (date is None) or (n_days == 0):
            raise ValueError(f"Valid date range for station {stat}.{chan} is {stat_startdate} - {stat_enddate}. Exiting.")

        missing_dates = []
        file_error_dates = []
        insufficient_data_dates = []
        ### Iterate over the specified number of days ###
        for _ in range(n_days):
            ### If starting in a new year, reload metadata
            if date.year > year:
                year = date.year
                stat_startdate, stat_enddate = self.get_station_dates(year, stat, chan)
                logger.info(f"Starting in new year ({year})")

            # Should not need this anymore because of validate_date_range
            ### Make sure that the station is operational for this date
            # if not self.validate_run_date(date, stat_startdate, stat_enddate):
            #     # logger.warning(f"Valid date range for station {stat}.{chan} is {stat_startdate} - {stat_enddate}. Exiting.")
            #     # return
            #     continue

            ### The data files are organized Y/m/d, get the appropriate date/station files ###
            date_str = date.strftime("%Y/%m/%d")
            files = sorted(glob.glob(os.path.join(self.data_dir, date_str, f'*{stat}*{chan}*')))

            ### Make the output dirs have the same structure as the data dirs ###
            date_outdir = os.path.join(self.outdir, date_str)

            ### If there are no files for that station/day, move to the next day ###
            if len(files) == 0:
                logger.info(f'No data for {date_str} {stat} {chan}')
                missing_dates.append(date_str)
                # Reset dataloader
                self.dataloader.error_in_loading()
                continue
            elif (self.ncomps == 1 and len(files) != 1) or (self.ncomps ==3 and len(files) != 3):
                logger.warning(f"Incorrect number of files found for {date_str} {stat} {chan}")
                file_error_dates.append(date_str)
                # Reset dataloader
                self.dataloader.error_in_loading()
                continue 

            applied_successfully = self.apply_to_one_file(files, date_outdir, debug_N_examples=debug_N_examples)
            if not applied_successfully:
                insufficient_data_dates.append(date_str)

            date += delta

        if len(missing_dates) > 0:
            self.write_dates_to_file(self.outdir, "missing", stat, chan, missing_dates)
        if len(file_error_dates) > 0:
            self.write_dates_to_file(self.outdir, "file_error", stat, chan, file_error_dates)
        if len(insufficient_data_dates) > 0:
            self.write_dates_to_file(self.outdir, "insufficient_data", stat, chan, insufficient_data_dates)

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

        Returns:
            bool: True if data is loaded successfully, otherwise False
        """
        if outdir is None:
            outdir = self.outdir

        meta_outfile_name =  self.dataloader.make_outfile_name(files[0], outdir)

        start_total = time.time()
        if self.ncomps == 1:
            load_succeeded = self.dataloader.load_1c_data(files[0], 
                                         min_signal_percent=self.min_signal_percent,)
                                         #expected_file_duration_s=self.expected_file_duration_s)
        else:
            load_succeeded = self.dataloader.load_3c_data(files[0], files[1], files[2], 
                                         min_signal_percent=self.min_signal_percent,)
                                         #expected_file_duration_s=self.expected_file_duration_s)

        if not load_succeeded:
            self.dataloader.error_in_loading(outfile=meta_outfile_name)
            return False
        
        logger.debug(f"Time to load data: {time.time() - start_total:0.2f} s")
        start_P = time.time()
        self.__apply_to_one_phase(self.p_detector, self.p_proc_func, 
                                  files[0], outdir, debug_N_examples=debug_N_examples)
        logger.debug(f"Total time to apply P model: {time.time() - start_P:0.2f} s")
        if self.ncomps == 3:
            start_S = time.time()
            self.__apply_to_one_phase(self.s_detector, self.dataloader.process_3c_S, 
                                      files[0], outdir, debug_N_examples=debug_N_examples)
            logger.debug(f"Total time to apply S model: {time.time() - start_S:0.2f} s")

        ### Save the station meta info (including gaps) to a file in the same dir as the post probs. ###
        ### Only need one file per station/day pair ###
        self.dataloader.write_data_info(meta_outfile_name)
        logger.debug(f"Total run time for day: {time.time() - start_total:0.2f} s")
        return True

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
        starttime = time.time()
        data, start_pad_npts, end_pad_npts = self.dataloader.format_continuous_for_unet(self.window_length,
                                                            self.sliding_interval,
                                                            proc_func,
                                                            normalize=True
                                                            )
        logger.debug(f"Time to format continuous data for UNET: {time.time() - starttime:0.2f} s")
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

    def get_station_dates(self, year, stat, chan):
        """Read in station xml files and get the start and end dates for the appropriate channels.

        Args:
            year (str): year of the data being processed
            stat (str): station name
            chan (str): chan code

        Returns:
            tuple: (start Datetime, end DateTime)
        """
        file_name = os.path.join(self.data_dir, str(year), f"stations/*{stat}.xml")
        files = glob.glob(file_name)
        
        if len(files) != 1:
            # raise ValueError(f"Station xml file {file_name} does not exist. Check station and channel values.")
            logger.warning(f"Station xml file {file_name} does not exist...")
            return None, None
        
        inv = obspy.read_inventory(files[0])
        
        # If a 3C station, only look at E/N or 1/2 components
        # in case a 3C was changed to 1C or vice versa
        if self.ncomps == 3:
            if len(chan) == 3:
                chan = chan[:-1]
            chan += "[!Z]"

        inv = inv.select(channel=chan)

        if len(inv) == 0:
            logger.warning(f"No metadata for channel {chan} in {file_name}")
            return None, None

        stat_info = inv.select(channel=chan)[0][0]
        start_date = np.min([stat_info[i].start_date for i in range(len(stat_info))])
        ends = [stat_info[i].end_date for i in range(len(stat_info))]
        end_date = (None if None in ends else np.max(ends))

        if start_date is not None:
            start_date = start_date.datetime
        if end_date is not None:
            end_date = end_date.datetime

        return start_date, end_date

    @staticmethod
    def validate_run_date(current_date, start_date, end_date):
        """Check if the date of interest is within the station/channel operating period. 

        Args:
            current_date (datetime): date of interest
            start_date (datetime): start date of the station
            end_date (datetime): end date of the station

        Returns:
            bool: True if valid date, False if not. 
        """
        # Likley no metadata read in
        if start_date is None:
            return False
        
        # Station is on going 
        if current_date >= start_date and end_date is None:
            return True
        
        if current_date >= start_date and current_date <= end_date:
            return True
        
        return False
    
    @staticmethod
    def validate_date_range(stat_startdate, stat_enddate, run_startdate, n_days):
        """Validates and corrects a date range given the station operation period. The date range is
        invalid if no station start date is available or the date range is entirely outside of station operation. 
        If valid, the start date and number of days for will be updated to agree with the station start and end 
        dates, as needed.

        Args:
            stat_startdate (_type_): _description_
            stat_enddate (_type_): _description_
            run_startdate (_type_): _description_
            n_days (_type_): _description_

        Returns:
            _type_: _description_
        """
        run_enddate = run_startdate + n_days*datetime.timedelta(days=1)

        # Date range of interest is entirely before the station being up or
        # or no start date was found in the metadata 
        if (stat_startdate is None) or (run_enddate < stat_startdate):
            return None, 0
        
        # Date range of interest is entirely after the station being removed.
        # It is okay for the station enddate to be none.
        if (stat_enddate is not None) and (run_startdate > stat_enddate):
            return None, 0
        
        # The date range of interest is partially before the station being up
        if run_startdate < stat_startdate:
            #run_startdate = stat_startdate
            run_startdate = datetime.datetime(year=stat_startdate.year, 
                                              month=stat_startdate.month, 
                                              day=stat_startdate.day)

        # The date range of interest is partially after the station being removed.
        # It is okay for the station enddate to be none. 
        if (stat_enddate is not None) and (run_enddate > stat_enddate):
            run_enddate = stat_enddate

        n_days = (run_enddate - run_startdate).days

        return run_startdate, n_days

    def write_dates_to_file(self, basedir, error_type, stat, chan, dates):
        """Append the list of dates to a text file for that station/channel. Will make a DataIssues dir 
        and a subdirectory based on the error_type, if needed. 

        Args:
            basedir (str): Path that will contain the DataIssues dir
            error_type (str): Type of error the dates relate to. "missing", "file_error", or "insufficient_data"
            stat (str): Station name for the file name
            chan (str): Channel code for the file name
            dates (list): List of date strings (Y/m/d) to write to the file

        Raises:
            ValueError: If the error_type is not supported
        """
        if error_type not in ["missing", "file_error", "insufficient_data"]:
            raise ValueError("Unexpected date_type to write to file")
        
        if self.ncomps == 3 and len(chan) == 3:
            chan = chan[0:2]
        #     chan += "?"

        outdir = os.path.join(basedir, "DataIssues", error_type)
        if not os.path.exists(outdir):
            logger.debug(f"Making directory {outdir}")
            try:
                os.makedirs(outdir)
            except:
                logger.info(f"{outdir} likely created by another job...")

        filename = f"{stat}.{chan}.txt"
        with open(os.path.join(outdir, filename), "a") as f:
            for missing_date in dates:
                f.write(f"{missing_date}\n")

class PhaseDetector():
    def __init__(self, 
                 model_to_load, 
                 num_channels, 
                 phase_type,
                 min_presigmoid_value=None, 
                 device="cpu",
                 num_torch_threads=2,
                 post_probs_file_type="MSEED"):
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
        self.use_openvino_async = False
        
        self.__init_model(num_channels, model_to_load)
        self.num_channels = num_channels
        self.post_probs_file_type = post_probs_file_type    
        
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

    def compile_openvino_model(self, window_length, batchsize, use_async):
            # Convert the model to openvino
            ov_model = ov.convert_model(self.unet, 
                                        input=([batchsize, self.num_channels, window_length],
                                                ov.Type.f32))

            # Compile the model for the appropriate device
            core = ov.Core()
            n_threads = self.num_threads
            # Pytorch uses -1 to indicate using all threads while openvino uses 0
            if n_threads < 0:
                n_threads = 0

            perf_hint = hints.PerformanceMode.LATENCY
            if use_async:
                perf_hint = hints.PerformanceMode.THROUGHPUT

            ov_compiled_model = core.compile_model(ov_model, 
                                                   self.openvino_device,
                                                   config={ov.properties.inference_num_threads(): n_threads,
                                                           hints.performance_mode: perf_hint,#})
                                                           "AFFINITY": "NUMA"})
            self.unet = ov_compiled_model
            self.openvino_compiled = True
            self.use_openvino_async = use_async

    def apply_sync_openvino_model_to_batch(self, X, center_window=None):
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

    def apply_async_openvino(self, input, batchsize=2, center_window=None):
        n_examples_org, n_samples, n_channels = input.shape
        input = input.transpose((0, 2, 1)).astype("float32").copy()

        j1 = 0
        j2 = n_samples
        if center_window is not None:
            j1 = int(n_samples/2 - center_window)
            j2 = int(n_samples/2 + center_window)
        
        n_examples = n_examples_org
        if n_examples_org % batchsize != 0:
            expansion_size = (batchsize - n_examples%batchsize)
            n_examples += expansion_size
            input = np.append(input, 
                              np.zeros((expansion_size, n_channels, n_samples), dtype="float32"), 
                              axis=0) 

        post_probs = np.zeros((n_examples, 1, n_samples), dtype="float32")

        # Set up inference request
        num_requests = self.unet.get_property(props.optimal_number_of_infer_requests)
        # print("Optimal number of requests: ", num_requests)
        infer_queue = ov.AsyncInferQueue(self.unet, num_requests)

        # Function for storing the results
        def callback(infer_request, userdata):
            results = infer_request.get_output_tensor().data
            s = results.shape[0]
            post_probs[userdata:userdata+s, :, :] = results
        infer_queue.set_callback(callback)

        sind = 0
        while sind < n_examples:
            eind = np.min([n_examples, sind+batchsize*num_requests])

            for i in range(sind, eind, batchsize):
                shared_tensor = ov.Tensor(input[i:i+batchsize, :, :], shared_memory=True)
                infer_queue.start_async({0: shared_tensor}, userdata=i)

            sind += num_requests*batchsize
        infer_queue.wait_all()

        post_probs = np.squeeze(post_probs[:n_examples_org, :, j1:j2], axis=1)
        # Not going to flatten the post probs here, in case the center window is not used 
        return post_probs


    def apply_sync(self, input, batchsize=256, center_window=None):
        """Apply the UNet to batches of data in a syncronous manner.

        Args:
            input (np.array): Data to apply model to. In format N x S x C. 
            batchsize (int, optional): Batch size to use. Defaults to 256.
            center_window (int, optional): Number of samples (W) on either side of the center of the model output to 
            return. Used with sliding windows. If None, returns all S samples. Defaults to None.

        Returns:
            np.array: Posterior probabilities in shape (N x S) or (N x W).
        """
        n_examples_org, n_samples, n_channels = input.shape

        n_examples = n_examples_org
        if n_examples_org % batchsize != 0:
            expansion_size = (batchsize - n_examples%batchsize)
            n_examples += expansion_size
            input = np.append(input, 
                            np.zeros((expansion_size, n_samples, n_channels), dtype="float32"), 
                            axis=0) 
            
        if center_window is not None:
            n_samples = center_window*2

        # model output is N x S
        post_probs = np.zeros((n_examples, n_samples), dtype="float32")
        batch_start = 0
        while batch_start < n_examples:
            batch_end = np.min([batch_start+batchsize, n_examples])
            batch = input[batch_start:batch_end, :, :]

            if not self.openvino_compiled:
                post_probs[batch_start:batch_end, :] = self.apply_model_to_batch(batch, 
                                                                                 center_window=center_window)
            else:
                post_probs[batch_start:batch_end, :] = self.apply_sync_openvino_model_to_batch(batch, 
                                                                                          center_window=center_window)

            batch_start += batchsize

        # Not going to flatten the post probs here, in case the center window is not used 
        return post_probs[:n_examples_org, :]
    
    def get_n_params(self):
        """ Get the number of trainable parameters in the model"""
        
        if self.openvino_compiled:
            logger.warning('Can only return number of params for Torch model, not OpenVino')
            return None
        
        return sum(p.numel() for p in self.unet.parameters() if p.requires_grad)

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
        if self.use_openvino_async:
            unet_output = self.apply_async_openvino(input, center_window=center_window, batchsize=batchsize)
        else:
            unet_output = self.apply_sync(input, center_window=center_window, batchsize=batchsize)
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

    def save_post_probs(self, outfile, post_probs, stats):
        if self.post_probs_file_type == "MSEED":
            self.save_post_probs_miniseed(outfile, post_probs, stats)
        elif self.post_probs_file_type == "NP":
            self.save_post_probs_numpy(outfile, post_probs)
        else:
            raise ValueError("post_probs_file_type must be NP or MSEED")

    @staticmethod
    def save_post_probs_miniseed(outfile, post_probs, stats):
        """Write the continuous posterior probabilities to disk in miniseed format with 2 digits of precision.
        Posterior probabilities will be between 0 and 99.

        Args:
            outfile (str): Path and name of the output file.
            post_probs (np.array): Flattened (1D) posterior probabilities
            stats (object): Dictionary object containing the station information, start time, and number of points.
        """
        assert post_probs.shape[0] == stats['npts'], "posterior probability is the wrong shape"
        outfile = outfile + ".mseed"
        logger.debug("writing %s", outfile)
        st = obspy.Stream()
        tr = obspy.Trace()
        # I have no idea why, but making the data int32 and not specifying the miniseed encoding makes the  
        # smallest output files (5.3 MB) compared to saving as using int32 encoding (34 MB) or int16 (17 MB).
        # Must be some extra compression?
        tr.data = (post_probs*100).astype(np.int32)
        tr.stats = Stats(stats)
        st += tr
        st.write(outfile, format="MSEED") #, encoding="INT16")

    @staticmethod
    def save_post_probs_numpy(outfile, post_probs):
        """Write the continuous posterior probabilities to disk in npz format with 2 digits of precision.
        Posterior probabilities will be between 0 and 99.

        Args:
            outfile (str): Path and name of the output file.
            post_probs (np.array): Flattened (1D) posterior probabilities
        """
        data = (post_probs*100).astype(np.uint8)
        np.savez_compressed(outfile, probs=data)
        
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
        Makes the output directory if it does not exist. Does not include the file format.

        Args:
            wf_filename (str): Input file name in format NET.STAT.LOC.CHAN__START__END.mseed
            dir (str): Path to write the output file to. 

        Returns:
            str: Output path and filename. Filename in format probs.PHASE__NET.STAT.LOC.CHAN__START__END.
        """
        # split = re.split(r"__|T", os.path.basename(wf_filename))
        # chan = "Z"
        # if self.num_channels > 1:
        #     chan = ""
        # post_prob_name = f"probs.{self.phase_type}__{split[0][0:-1]}{chan}__{split[1]}__{split[3]}.mseed"

        post_prob_name = f"probs.{self.phase_type}__{Path(wf_filename).stem}"
        # {os.path.basename(wf_filename).split(".mseed")[0]}
        # I think I need to move this lock, maybe by the imports?
        if not os.path.exists(dir):
            logger.debug(f"Making directory {dir}")
            try:
                os.makedirs(dir)
            except:
                logger.info(f"{dir} likely created by another job...")

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

    def error_in_loading(self, outfile=None):
        # Save outfile info (only works if the data was able to be loaded in)
        if outfile is not None:
            self.write_data_info(outfile)

        self.reset_loader()
        # If skipping a day, then there is no previous day for the next trace
        self.reset_previous_day()

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
        
        Returns:
            bool: True if there is sufficient data on all channels, False otherwise
        """

        self.reset_loader()

        # assert np.isin(re.split( "[.|__]", os.path.basename(fileE))[3], ["EHE", "EH1", "BHE", "BH1", "HHE"]), "E file is incorrect"
        # assert np.isin(re.split( "[.|__]", os.path.basename(fileN))[3], ["EHN", "EH2", "BHN", "BH2", "HHN"]), "N file is incorrect"
        # assert np.isin(re.split( "[.|__]", os.path.basename(fileZ))[3], ["EHZ", "BHZ", "HHZ"]), "Z file is incorrect"

        load_succeeded_E, st_E, gaps_E = self.load_channel_data(fileE, 
                                              min_signal_percent=min_signal_percent,
                                              expected_file_duration_s=expected_file_duration_s)
        load_succeeded_N, st_N, gaps_N = self.load_channel_data(fileN, 
                                              min_signal_percent=min_signal_percent,
                                              expected_file_duration_s=expected_file_duration_s)
        load_succeeded_Z, st_Z, gaps_Z = self.load_channel_data(fileZ, 
                                              min_signal_percent=min_signal_percent,
                                              expected_file_duration_s=expected_file_duration_s)

        assert np.isin(st_E[0].stats.channel, ["EHE", "EH1", "BHE", "BH1", "HHE", "HH1"]), "E file is incorrect"
        assert np.isin(st_N[0].stats.channel, ["EHN", "EH2", "BHN", "BH2", "HHN", "HH2"]), "N file is incorrect"
        assert np.isin(st_Z[0].stats.channel, ["EHZ", "BHZ", "HHZ"]), "Z file is incorrect"

        # # When an entire day is removed, remove orientation from gap channel info
        # if not load_succeeded_Z:
        #     gaps_Z[0][3] = gaps_Z[0][3][0:2] + "?"
        #     self.gaps = gaps_Z
        #     # self.reset_previous_day()
        #     return False
        # elif not load_succeeded_N:
        #     gaps_N[0][3] = gaps_N[0][3][0:2] + "?"
        #     self.gaps = gaps_N
        #     # self.reset_previous_day()
        #     return False
        # elif not load_succeeded_E:
        #     gaps_E[0][3] = gaps_E[0][3][0:2] + "?"
        #     self.gaps = gaps_E
        #     # If skipping a day, then there is no previous day for the next trace
        #     # self.reset_previous_day()
        #     return False

        # Keep all gap information and store metadata - regardless if load failed
        gaps = gaps_E + gaps_N + gaps_Z
        self.gaps = gaps
        self.store_meta_data(st_E[0].stats)
        # If any loads failed, exit
        if (not load_succeeded_Z) or (not load_succeeded_E) or (not load_succeeded_N):
            return False

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

        self.continuous_data = cont_data
        
        if self.store_N_seconds > 0:
            self.prepend_previous_data()

        return True

    def load_1c_data(self, file, min_signal_percent=1, expected_file_duration_s=86400):
        """Load miniseed file for a 1C station.

        Args:
            file (str): Path to the Z component file. 
            min_signal_percent (int, optional): Minimum percentage of signal that must be present in all channels.
              Defaults to 1.
            expected_file_duration_s (int, optional): The number of expected seconds in file. 
            Only implemented for 86400. Defaults to 86400.

        Returns:
            bool: True if there is sufficient data, False otherwise
        """

        self.reset_loader()

        load_succeeded, st, gaps = self.load_channel_data(file, 
                                              min_signal_percent=min_signal_percent,
                                              expected_file_duration_s=expected_file_duration_s)
        
        self.gaps = gaps
        self.store_meta_data(st[0].stats, three_channels=False)

        if not load_succeeded:
            return False
        
        cont_data = np.zeros((st[0].stats.npts, 1))
        cont_data[:, 0] = st[0].data

        self.continuous_data = cont_data

        if self.store_N_seconds > 0:
            # Update continous data and the metadata to include the end of the previous trace
            # TODO: don't prepend previous data if the end of the previous day has been filled in because
            # of a gap
            self.prepend_previous_data()

        return True
   
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
        elif self.store_N_seconds > 0 and self.continuous_data is None:
            # If no continous data was loaded in, reset the previous day information
            self.reset_previous_day()
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
        """Reads in a miniseed file and check for gaps. Gaps are interpolated, if there is sufficient data. 

        Args:
            file (str): Path to the miniseed file to read in.
            min_signal_percent (int, optional): Minimum percent signal required to process the file. Defaults to 1.
            max_duration_s (int, optional): Expected duration of the miniseed file in seconds. Defaults to 86400.

        Returns:
        tupple: (bool: Sufficient data or not, Obspy Stream, list of gaps)
        """ 

        logger.debug("loading %s", file)
        # use obspy to load the miniseed file(s)
        st = obspy.read(file)

        sampling_rate = round(st[0].stats.sampling_rate)

        # Safer to just not allow other sampling rates for now
        if (abs(sampling_rate - 100) > 1e-3):
            raise NotImplementedError("Only data sampled at 100 Hz has been tested")
        
        # TODO: this only works for days, not hours - fine for me
        if expected_file_duration_s != 24*60*60:
            raise NotImplementedError("DataLoader only works for day long files at the moment...")
        
        starttime = st[0].stats.starttime
        desired_start = UTC(starttime.year, starttime.month, starttime.day)
        desired_end = desired_start + expected_file_duration_s #-0.01)
        # TODO: If the startime happens to be on the day before, it'll mess up the desired day
        assert starttime >= desired_start, "The stream begins on the previous day of interest"
        
        # If there is not enough signal in this day, skip the day
        total_npts = np.sum([st[i].stats.npts for i in range(len(st))])
        max_npts = expected_file_duration_s*round(sampling_rate)
        sufficient_data = True
        if (total_npts/max_npts)*100 < min_signal_percent:
            logger.warning(f"{os.path.basename(file)} does not have enough data, skipping")
            sufficient_data = False
            # Return the entire file period as a gap
            #return False, st, [self.format_edge_gaps(st, desired_start, desired_end, entire_file=True)]

        # Save gaps so I know to ignore any detections in that region later
        # Don't save gaps if they are smaller than 5 samples for 100 Hz, 2 samples for 40 Hz
        gaps = st.get_gaps(min_gap=0.05)

        # Get start/end gaps for days without sufficient data and return
        if not sufficient_data:
            start_gap, end_gap = self.format_edge_gaps(st, desired_start, desired_end, interpolated=False)
            if (start_gap != None) and (start_gap[5] - start_gap[4] >= 0.05):
                gaps.insert(0, start_gap)
            if (end_gap != None) and (end_gap[5] - end_gap[4] >= 0.05):
                gaps += [end_gap]    
            return False, st, gaps

        # Still need to interpolate the gaps if they are less than min_gap
        if len(st.get_gaps()) > 0:
            # Sometimes the gaps have different sampling rates and delta values, not totally sure why. 
            # The sampling rates do not equal npts/duration for the gap either.
            try:
                st.merge(fill_value='interpolate')
            except Exception as e:
                logger.info(f"Caught obspy Incompatible Traces warning: {e}. Fixing the sampling rates...")
                delta = round(1/sampling_rate, 3)

                for tr in st:
                    tr.stats.sampling_rate = sampling_rate
                    tr.stats.delta = delta

                st.merge(fill_value='interpolate')

        # Make sure the trace is the desired length (fill in ends if needed) 
        # Check for gaps at the start/end of the day and save if they exist
        start_gap, end_gap = self.format_edge_gaps(st, desired_start, desired_end)

        if start_gap is not None:
            # Don't save gaps less than 0.05 s (5 samples)
            if start_gap[5] - start_gap[4] >= 0.05:
                gaps.insert(0, start_gap)
            # Fill the start of the trace with the first value, if needed
            st = st.trim(starttime=desired_start, pad=True, 
                         fill_value=st[0].data[0], nearest_sample=False)
        if end_gap is not None:
            # Don't save gaps less than 0.05 s (5 samples)
            if end_gap[5] - end_gap[4] >= 0.05:
                gaps += [end_gap]
            # Fill the end of the trace with the last value, if needed
            # set nearest_sample to False or it will extent to the next day 
            # and have 8460001 samples
            st = st.trim(endtime=desired_end, pad=True, 
                         fill_value=st[0].data[-1], nearest_sample=False)

        # Trim the waveform to the desired length if it is too long
        # Do this after filling the start gap so I can trim it using the number of 
        # samples instead of the trim function - which might result in +- 1 sample difference
        # Obspy handles endtime update after reducing the data 
        if st[0].stats.endtime > desired_end:
            st[0].data = st[0].data[0:int(expected_file_duration_s*sampling_rate)]
        assert st[0].stats.endtime <= desired_end, "The end of the trace goes into the next day"

        return True, st, gaps 

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
        # TODO: I'm pretty sure this won't work if the desired_sampling_rate != current sampling rate
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
    def format_edge_gaps(st, desired_start, desired_end, entire_file=False, interpolated=True):
        """Checks for gaps greater than 1 sample at the start and end of an Obspy Stream.
        Only looks at the first trace in the stream. *The desired endtime may be off by ~1 second*
        If a gap exists returns a list in the format of Obspy's get_gaps(), otherwise returns None.

        Args:
            st (Obspy Stream): stream to check for gaps
            max_duration_s (int): expected duration of the file in seconds
            entire_file (boolean, optional): True if the gap is the entire file. Defaults to False.

        Returns:
            tupple or list: tupple of gap info lists for the beginning and end of the trace or a list for the entire file gap. 
        """
        end_gap_st_ind = 0
        if not interpolated:
            end_gap_st_ind = -1

        starttime = st[0].stats.starttime
        endtime = st[end_gap_st_ind].stats.endtime 
        sampling_rate = round(st[0].stats.sampling_rate)

        if entire_file:
            max_duration_s = desired_end - desired_start
            return [st[0].stats.network, st[0].stats.station, st[0].stats.location, st[0].stats.channel, 
                        desired_start, desired_end, max_duration_s, int(max_duration_s*sampling_rate)]

        # Compute duration of gaps at the start and end
        start_delta = starttime - desired_start
        end_delta = desired_end - endtime

        dt = st[0].stats.delta
        start_gap = None
        if start_delta > dt:
            start_gap = [st[0].stats.network, st[0].stats.station, st[0].stats.location, st[0].stats.channel, 
                        desired_start, starttime, start_delta, int(start_delta*sampling_rate)]

        end_gap = None
        if end_delta > dt:
            end_gap = [st[end_gap_st_ind].stats.network, st[end_gap_st_ind].stats.station, 
                       st[end_gap_st_ind].stats.location, st[end_gap_st_ind].stats.channel,
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
            try:
                os.makedirs(dir)
            except:
                logger.info(f"{dir} likely created by another job...")

        outfile = os.path.join(dir, post_prob_name)

        return outfile