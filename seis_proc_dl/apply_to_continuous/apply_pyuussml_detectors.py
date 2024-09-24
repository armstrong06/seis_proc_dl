import obspy
from obspy.core.utcdatetime import UTCDateTime as UTC
from obspy.core.trace import Stats
import numpy as np
import logging
import torch
import os
import re
# TODO: Better way to import pyuussmlmodels than adding path?
import sys
import json
import datetime
import glob

from seis_proc_dl.apply_to_continuous.apply_detectors import DataLoader

sys.path.append("/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/mlmodels/intel_cpu_build")
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

class ApplyDetectorPyuussml():
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
        self.device = config.unet.device
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
        self.p_detector = pyuussmlmodels.Detectors.UNetOneComponentP.Inference()
        self.p_detector.load(self.p_model_file,
                             pyuussmlmodels.Detectors.UNetOneComponentP.ModelFormat.ONNX) # Update?
        assert self.p_detector.is_initialized
        self.p_proc_func = self.dataloader.process_1c_P

    def __init_3c(self):
        """Initialize the phase detectors for 3C P and S detectors
        """
        self.p_detector = pyuussmlmodels.Detectors.UNetThreeComponentP.Inference()
        self.p_detector.load(self.p_model_file,
                             pyuussmlmodels.Detectors.UNetThreeComponentP.ModelFormat.ONNX) # Update?
        assert self.p_detector.is_initialized
        
        self.s_detector = pyuussmlmodels.Detectors.UNetThreeComponentS.Inference()
        self.s_detector.load(self.p_model_file,
                             pyuussmlmodels.Detectors.UNetThreeComponentS.ModelFormat.ONNX) # Update?
        assert self.s_detector.is_initialized

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
                                  files[0], outdir, 
                                  phase_type="P",
                                  debug_N_examples=debug_N_examples)

        if self.ncomps == 3:
            self.__apply_to_one_phase(self.s_detector, self.dataloader.process_3c_S, 
                                      files[0], outdir, 
                                      phase_type="S", 
                                      debug_N_examples=debug_N_examples)

        ### Save the station meta info (including gaps) to a file in the same dir as the post probs. ###
        ### Only need one file per station/day pair ###
        meta_outfile_name =  self.dataloader.make_outfile_name(files[0], outdir)
        self.dataloader.write_data_info(meta_outfile_name)

    def __apply_to_one_phase(self, 
                             detector, 
                             proc_func, 
                             file_for_name, 
                             outdir,
                             phase_type,
                             debug_N_examples=-1):
        """Format continuous data and apply phase detector. Write posterior probabilities to disk.

        Args:
            detector (object): Initialized PhaseDetector with the appropriate model loaded.
            proc_func (object): The processing function to use when processing each waveform segment.
            file_for_name (str): Input file name used to derive the posterior probability output file name.
            outdir (str): Specify the directory to save the posterior probabilities to.
            debug_N_examples (int, optional):  Number of waveform segments to pass to the phase detector, use 
            all inputs if -1. Defaults to -1.
        """
        probs_outfile_name = self.make_outfile_name(phase_type, file_for_name, outdir)

        data_day = self.dataloader.continuous_data
        if debug_N_examples > 0:
            central_window_start_end_ind = detector.central_window_start_end_index
            central_window_n_samples = central_window_start_end_ind[1] - central_window_start_end_ind[0]
            logger.debug("Reducing data to %d examples", debug_N_examples)
            data_day = data_day[0:int(debug_N_examples*detector.expected_signal_length), :] 
            # Update npts so an error does not get thrown in save_post_probs
            # self.dataloader.metadata['npts'] = int(debug_N_examples*central_window_n_samples*2)
            # Ben's code keeps the output the same size - TODO: check what he does for edge cases
            self.dataloader.metadata['npts'] = data_day.shape[0]
        n_samples_total = data_day.shape[0]

        all_cont_post_probs = []
        sampling_rate = detector.sampling_rate
        n_samples_hour = int(60*60*sampling_rate)
        buffer = int(60*2*sampling_rate) # Add one minute buffer on either end
        start_sample = 0 # Begin at 0
        start_pad = 0   # No buffer at the beginning of the day
        end_pad = buffer    # Buffer at the beginning but not the end

        while (start_sample + 2*buffer) < n_samples_total:
            end_sample = int(start_sample + n_samples_hour + buffer)
            if start_pad != 0:
                end_sample += buffer
            if end_sample >= n_samples_total:
                end_sample = n_samples_total
                end_pad = 0

            data = data_day[start_sample:end_sample]
            data = proc_func(data)

            if data.shape[1] == 3:
                cont_post_probs = detector.predict_probability(data[:, 2], 
                                                            data[:, 1], 
                                                            data[:, 0], 
                                                            use_sliding_window=True)
            else:
                cont_post_probs = detector.predict_probability(data[:, 0], 
                                                            use_sliding_window=True)
            last_sample = int(cont_post_probs.shape[0]-end_pad)
            #print(start_pad, last_sample, last_sample-start_pad)
            cont_post_probs = cont_post_probs[start_pad:last_sample]
            # assert cont_post_probs.shape[0] == n_samples_hour
            all_cont_post_probs.append(cont_post_probs)
            start_sample = int(end_sample - 2*buffer)
            start_pad = buffer

        all_cont_post_probs = np.concatenate(all_cont_post_probs)
        assert all_cont_post_probs.shape[0] == n_samples_total
        self.save_post_probs(probs_outfile_name, all_cont_post_probs, self.dataloader.metadata)

    def make_outfile_name(self, phase_type, wf_filename, dir):
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

        post_prob_name = f"probs.{phase_type}__{os.path.basename(wf_filename)}"

        if not os.path.exists(dir):
            logger.debug(f"Making directory {dir}")
            os.makedirs(dir)

        outfile = os.path.join(dir, post_prob_name)

        return outfile
    
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