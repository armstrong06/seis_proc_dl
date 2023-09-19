#!/usr/bin/env python3
from locale import normalize
import numpy as np
from numpy.lib.nanfunctions import nancumsum
import pandas as pd
import warnings
import os
import torch
import torch.utils.data
from sklearn.metrics import confusion_matrix
import time
import sys
import obspy
import glob
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt
import h5py
from scipy.stats import norm 

from utils.model_helpers import clamp_presigmoid_values
from detectors.models.unet import UNetModel
from pick_regressors.cnn_picker import CNNNet
from fm_classifiers.train_fm import FMNet
from utils.config_apply_models import Config

from joblib import load

sys.path.append("..atucus/swag_modified")
from swag import seismic_data as swag_seismic_data
from swag import losses as swag_losses
from swag import models as swag_models
from swag import utils as swag_utils
from swag.posteriors import SWAG


class Dset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        print(data.shape)
        self.data = torch.from_numpy(data.transpose((0, 2, 1))).float()
        
    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)

class swag_picker():
    def __init__(self, model_name, checkpoint_file, seed, cov_mat=True, K=20, device="cuda:0"):
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        eps = 1e-12
        model_cfg = getattr(swag_models, model_name)
        self.cov_mat = cov_mat
        self.seed = seed
        # self.loaders = swag_seismic_data.loaders(
        #     train_filename,
        #     None,
        #     train_data_path,
        #     batch_size,
        #     num_workers,
        #     shuffle_train=False)

        self.model = SWAG(
            model_cfg.base,
            no_cov_mat=not cov_mat,
            max_num_models=K,
            *model_cfg.args,
            **model_cfg.kwargs)
        
        #self.model.cuda()
        self.device = device
        self.model.to(self.device)

        print("Loading model %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint["state_dict"])
        #self.batchsize = batchsize

    def apply_model(self, dset_loader, N, train_loader, scale=0.5):
        # dset = Dset(data)
        # dset_loader = torch.utils.data.DataLoader(
        #     dset,
        #     batch_size=self.batchsize,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        #     pin_memory=True,
        # )
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        predictions = np.zeros((len(dset_loader.dataset), N))
        for i in range(N):
            self.model.sample(scale=scale, cov=self.cov_mat)
            swag_utils.bn_update(train_loader, self.model)
            self.model.eval()
            k = 0
            for input in dset_loader:
                #input = input.cuda(non_blocking=True)
                input = input.to(self.device, non_blocking=True)
                torch.manual_seed(i)
                output = self.model(input)
                with torch.no_grad():
                    predictions[k : k + input.size()[0], i:i+1] = output.cpu().numpy()
                k += input.size()[0]

        return predictions

class detector():
    def __init__(self, phase, model_to_test, num_channels=3, min_presigmoid_value=None,
                 device="cuda:0"):
        self.phase = phase
        warnings.simplefilter("ignore")
        self.device = torch.device(device)
        self.min_presigmoid_value = min_presigmoid_value

        print(f"Initializing {num_channels} comp {phase} unet...")
        self.unet = UNetModel(num_channels=num_channels, num_classes=1).to(self.device)
        print("Number of parameters in my model:", self.get_n_params())
        assert os.path.exists(model_to_test), f"Model {model_to_test} does not exist"
        print("Loading model:", model_to_test)
        check_point = torch.load(model_to_test)
        self.unet.load_state_dict(check_point['model_state_dict'])
        self.unet.eval()

    def apply_model_to_batch(self, X, lsigmoid=True, center_window=None):
        n_samples = X.shape[1] 
        X = torch.from_numpy(X.transpose((0, 2, 1))).float().to(self.device)
        
        if (lsigmoid):
            model_output = self.unet.forward(X)
            if self.min_presigmoid_value is not None:
                model_output = clamp_presigmoid_values(model_output, self.min_presigmoid_value)
            Y_est = torch.sigmoid(model_output)
        else:
            Y_est = self.unet.forward(X)
        
        Y_est = Y_est.squeeze()

        if center_window:
            j1 = int(n_samples/2 - center_window)
            j2 = int(n_samples/2 + center_window)
            Y_est = Y_est[:,j1:j2]

        if self.phase == "S":
            return Y_est.to('cpu').detach().numpy(), model_output.to("cpu").detach().numpy().squeeze()
        else:
            return Y_est.to('cpu').detach().numpy()

    def get_n_params(self):
        return sum(p.numel() for p in self.unet.parameters() if p.requires_grad)


    @staticmethod
    def sliding_window_phase_arrival_estimator_updated(Y, window_size=100, thresh=0.1, end_thresh_diff=0.05):
        """

        Find potential phase arrivals in the probability time-series output from the UNet given a minimum threshold value.
        Chooses window to look for detection based on when the proba goes above thresh and then below again
        :param Y: Probability time series
        :param window_size: step size for sliding window
        :param thresh: minimum probability threshold to record picks
        :param min_boxcar_width: approx. minimum width of a boxcar detection allowed
        :return: list of the samples in Y with an expected phase arrival (given thresh)
        """
        i1 = 0
        picks = []
        proba_values = []
        widths = []
        end_thresh = thresh-end_thresh_diff
        while i1 < Y.shape[0]:
            i2 = i1 + np.min([(Y.shape[0] - i1), window_size])
            if np.any(Y[i1:i2] >= thresh):
                # find first ind in window above thresh (start looking for max proba)
                start_win = i1 + np.where(Y[i1:i2] >= thresh)[0][0]
                # find end ind where proba has gone below thresh - if start and end inds are too close together, find a new end index
                search_win_size = 100
                possible_win_lengths = np.where(Y[start_win:start_win+search_win_size] < end_thresh)[0]
                while len(possible_win_lengths) < 1:
                    search_win_size += 10
                    possible_win_lengths = np.where(Y[start_win:start_win+search_win_size] < thresh)[0]
                    if start_win + search_win_size > Y.shape[0]:
                        break
                if len(possible_win_lengths) == 0:
                    end_win = Y.shape[0]
                else:
                    end_win = start_win + possible_win_lengths[0]
                proba = np.max(Y[start_win:end_win])
                pick = start_win + np.where(Y[start_win:end_win] == proba)[0][0]
                widths.append(end_win-start_win)
                picks.append(pick)
                proba_values.append(proba)
                i1 = end_win
            else:
                i1 += window_size

        return np.array(proba_values), np.array(picks), np.array(widths)
        
class phase_picker():
    def __init__(self, phase, model_to_test, max_dt_nn=0.75, device=torch.device("cuda:0")):
        self.device = device
        self.phase = phase
        print(f"Initializing {phase} picking network...")
        
        num_channels = 1
        if phase == "S":
            num_channels = 3

        self.cnnnet = CNNNet(num_channels=num_channels, min_lag = -max_dt_nn, max_lag = +max_dt_nn).to(device)

        print("Loading model: ", model_to_test)
        check_point = torch.load(model_to_test)
        self.model_mean = check_point['validation_mean_of']
        self.cnnnet.load_state_dict(check_point['model_state_dict'])
        self.cnnnet.eval()

    def apply_model_to_batch(self, X):
        # if self.phase == "P":
        #     X = X.reshape([X.shape[0], 1, X.shape[1]])
        # else:
        X = X.transpose((0, 2, 1))
        X = torch.tensor(X, dtype=torch.float).to(self.device)
        y_est = self.cnnnet.forward(X)
        y_est  = y_est.detach().cpu().numpy() + self.model_mean
        return y_est

class fm_picker():
    def __init__(self, model_in,polarity=[1,-1, 0], device=torch.device("cuda:0")):
        self.device = device
        self.polarity = np.array(polarity)
        print("Initializing FM network...")
        self.fmnet = FMNet().to(device)
        print("Loading model: ", model_in)
        check_point = torch.load(model_in)
        self.fmnet.load_state_dict(check_point['model_state_dict'])
        self.fmnet.eval()

    def apply_model_to_batch(self, X):
        X = X.reshape([X.shape[0], 1, X.shape[1]])
        X_hat = torch.from_numpy(X).float().to(self.device)
        p = self.fmnet.forward(X_hat)
        pcpu = p.cpu().data.numpy()
        #y_pred = self.polarity[np.argmax(pcpu[:, :], axis=1)]

        return pcpu

    def get_polarity(self):
        return self.polarity

class apply_models():
    def __init__(self, CFG):
        # from finetune_config import CFG

        config = Config.from_json(CFG)

        self.batch_size = config.model_params.batch_size
        self.center_window = config.model_params.center_window
        self.sliding_interval = config.model_params.sliding_interval
        self.unet_window_length = config.model_params.unet_window_length
        self.pcnn_window_length = config.model_params.pcnn_window_length
        self.scnn_window_length = config.model_params.scnn_window_length
        self.batch_size=config.model_params.batch_size

        self.data_dir = config.paths.data_dir
        model_dir = config.paths.model_dir
        self.output_dir = config.paths.output_dir

        if not os.path.exists(self.output_dir):
            print("Making director", self.output_dir)
            os.makedirs(self.output_dir)

        self.catalog_file = config.output_file_names.catalog_file
        self.output_probability_file = config.output_file_names.output_probability_file

        self.device = config.options.device
        self.single_stat_string = config.options.single_stat_string
        self.debug_s_detector = config.options.debug_s_detector
        self.debug_inds_file = config.options.debug_inds_file
        self.p_detection_thresh = config.options.p_detection_thresh
        self.s_detection_thresh = config.options.s_detection_thresh

        # Initialize models 
        models = config.models
        min_presigmoid_value = config.model_params.min_presigmoid_value
        self.p_detector3c = detector("P", f'{model_dir}/{models.pDetector3c}', num_channels=3, min_presigmoid_value=min_presigmoid_value, device=self.device)
        self.s_detector = detector("S", f'{model_dir}/{models.sDetector}', num_channels=3, min_presigmoid_value=min_presigmoid_value, device=self.device)
        self.p_detector1c = detector("P", f'{model_dir}/{models.pDetector1c}', num_channels=1, min_presigmoid_value=min_presigmoid_value, device=self.device)

        if config.swag_info is None:
            self.use_swag = False
            self.ppicker = phase_picker("P", f'{model_dir}/{models.pPicker}', device=self.device)
            self.spicker = phase_picker("S", f'{model_dir}/{models.sPicker}', max_dt_nn=0.85, device=self.device)
        else:
            # self, batch_size, model_name, checkpoint_file, seed, train_filename, train_data_path, 
            # cov_mat=True, num_workers=4, K=20
            self.use_swag = True
            seeds = config.swag_info.seeds
            k = config.swag_info.K
            self.num_workers = config.swag_info.num_workers
            cov_mat = config.swag_info.cov_mat
            swag_ptrain_filename = config.swag_info.p_train_file
            swag_pdata_path = config.swag_info.p_data_path
            swag_strain_filename = config.swag_info.s_train_file
            swag_sdata_path = config.swag_info.s_data_path      
            self.picker_summary_method = config.swag_info.picker_summary_method

            self.swag_p_loaders = swag_seismic_data.loaders(
                            swag_ptrain_filename,
                            None,
                            swag_pdata_path,
                            self.batch_size,
                            self.num_workers,
                            shuffle_train=False)

            self.swag_s_loaders = swag_seismic_data.loaders(
                            swag_strain_filename,
                            None,
                            swag_sdata_path,
                            self.batch_size,
                            self.num_workers,
                            shuffle_train=False)

            self.swag_ppickers = [swag_picker("PPicker", f'{model_dir}/{models.swagPPicker[0]}', seeds[0], cov_mat=cov_mat, K=k, device=self.device),
                                swag_picker("PPicker", f'{model_dir}/{models.swagPPicker[1]}', seeds[1], cov_mat=cov_mat, K=k, device=self.device),
                                swag_picker("PPicker", f'{model_dir}/{models.swagPPicker[2]}', seeds[2], cov_mat=cov_mat, K=k, device=self.device)]

            self.swag_spickers = [swag_picker("SPicker", f'{model_dir}/{models.swagSPicker[0]}', seeds[0], cov_mat=cov_mat, K=k, device=self.device),
                                swag_picker("SPicker", f'{model_dir}/{models.swagSPicker[1]}', seeds[1], cov_mat=cov_mat, K=k, device=self.device),
                                swag_picker("SPicker", f'{model_dir}/{models.swagSPicker[2]}', seeds[2], cov_mat=cov_mat, K=k, device=self.device)]
            
            if config.swag_info.calibration_path is None:
                p_calibration_file = f"{model_dir}/{config.swag_info.P_calibration_file}"
                s_calibration_file = f"{model_dir}/{config.swag_info.S_calibration_file}"
            else:
                p_calibration_file = f"{config.swag_info.calibration_path}/{config.swag_info.P_calibration_file}"
                s_calibration_file = f"{config.swag_info.calibration_path}/{config.swag_info.S_calibration_file}"
            self.swag_p_lb, self.swag_p_ub = self.get_calibrated_pick_bounds(p_calibration_file, config.swag_info.CI_lower_bound, config.swag_info.CI_upper_bound)
            self.swag_s_lb, self.swag_s_ub = self.get_calibrated_pick_bounds(s_calibration_file, config.swag_info.CI_lower_bound, config.swag_info.CI_upper_bound)
            self.N_S = config.swag_info.N_S
            self.N_P = config.swag_info.N_P

        self.fmpicker = fm_picker(f'{model_dir}/{models.fmPicker}')
        self.results = []

    @staticmethod
    def get_calibrated_pick_bounds(iso_reg_inv_file, lb, ub):
        # Transform the lower and upper bounds to be calibrated
        iso_reg_inv = load(iso_reg_inv_file)
        lb_transform = iso_reg_inv.transform([lb])[0]
        ub_transform = iso_reg_inv.transform([ub])[0]
        return lb_transform, ub_transform

    def load_continuous(self, file_name, num_channels=3):
        ## Read in the data 
        st = obspy.read(file_name)
        print(f"Read in {st}")

        if len(st) > num_channels:
            print(f"Filling {len(st.get_gaps())} gaps in data with interpolate values")
            st.merge(fill_value="interpolate")
            print("Corrected Stream", st)

        # Check for gaps
        assert len(st.get_gaps()) == 0, "Gaps in data"

        if num_channels == 3:
            # Check number of points and sampling rates 
            if st[0].stats.npts != st[1].stats.npts or st[1].stats.npts != st[2].stats.npts:
                print("Trying to correct the number of points...")
                correct_npts = 24*60*60*st[0].stats.sampling_rate + 1
                good_trace = -1
                for i in range(len(st)):
                    if st[i].stats.npts == correct_npts or st[i].stats.npts == correct_npts-1:
                        good_trace = i
                if good_trace > -1:
                    for tr in st:
                        tr.trim(st[good_trace].stats.starttime, st[good_trace].stats.endtime, pad=True, fill_value=0)
                    print("Corrected", st)
                else:
                    # TODO: return an error message or something 
                    print("ERROR: NPTS do not match")
                    return None, 0
                
            if st[0].stats.delta != st[1].stats.delta or st[1].stats.delta != st[2].stats.delta:
                print("ERROR: Sampling rates do not match")
                return None, 0

        if st[0].stats.sampling_rate != 100:
            print(f"Resampled from {st[0].stats.sampling_rate} to 100 Hz...")
            st.resample(100)

        npts = st[0].stats.npts

        # Check number of points and sampling rates 
        if num_channels == 3:
            assert st[0].stats.npts == st[1].stats.npts == st[2].stats.npts
            assert st[0].stats.delta == st[1].stats.delta == st[2].stats.delta
        assert st[0].stats.delta == 0.01

        # start time of the hour long trace in UNIX time
        starttime_epoch = st[0].stats.starttime - UTCDateTime("19700101")
        for tr in st:
            tr.stats.starttime_epoch = starttime_epoch

        # how much to extend the last window to be evenly divisible by the sliding window
        padding_seconds = (self.sliding_interval - (npts-self.unet_window_length)%self.sliding_interval)*st[0].stats.delta
        new_endtime = st[0].stats.endtime+padding_seconds
        for tr in st:
            tr.trim(st[0].stats.starttime, new_endtime, pad=True, fill_value=tr.data[-1])
        
        if num_channels == 3:
            assert st[0].stats.npts == st[1].stats.npts == st[2].stats.npts

        npts=st[0].stats.npts
        # # Use this for debugging
        # npts = 60*60*100
        # number of times to move the window 
        n_intervals = (npts-self.unet_window_length)//self.sliding_interval + 1

        return st, n_intervals

    def stream_to_tensor_3c(self, st_preproc):
        npts=st_preproc[0].stats.npts
        # Start index of each sliding window
        # Use this for debugginh
        # npts = 60*60*100
        cont_data = np.zeros((npts, 3))
        order = {"E":0, "1":0, "N":1, "2":1, "Z":2}
        tr0_ind = order[st_preproc[0].stats.channel[-1]]
        tr1_ind = order[st_preproc[1].stats.channel[-1]]
        tr2_ind = order[st_preproc[2].stats.channel[-1]]

        # assert "Z" in st_preproc[2].stats.channel
        # assert "N" in st_preproc[1].stats.channel or "2" in st_preproc[1].stats.channel
        # assert "E" in st_preproc[0].stats.channel or "1" in st_preproc[0].stats.channel

        # For UNet, data is ordered E, N, Z
        cont_data[:, tr0_ind] = st_preproc[0].data[:npts]
        cont_data[:, tr1_ind] = st_preproc[1].data[:npts]
        cont_data[:, tr2_ind] = st_preproc[2].data[:npts]
        # print("Reducing data for testing", cont_data.shape)
        return cont_data

    def stream_to_tensor_1c(self, st_preproc):
        npts=st_preproc[0].stats.npts
        cont_data = np.zeros((npts, 1))
        cont_data[:, 0] = st_preproc[0].data

        return cont_data

    def apply_detectors_nonedgecases(self, cont_data, n_intervals, batch_size, start_indicies, debug_s=False, num_channels=3, debug_inds=None):
        # loop over batches - excluding the first and last window 
        if debug_s and num_channels == 1:
            debug_s = False
            print("Cannot debug S for 1C station, turning debug mode off")
        
        if not debug_s:
            p_posterior_probs = np.zeros((n_intervals-2, 2*self.center_window))

        if num_channels == 3:
            s_posterior_probs = np.zeros((n_intervals-2, 2*self.center_window))
            pre_sigmoid_output = np.zeros((n_intervals-2, self.unet_window_length))

        processed_data = np.zeros((n_intervals, self.unet_window_length, num_channels))
        batch_start = 1
        while batch_start < n_intervals-1:
            batch_end = np.min([batch_start+batch_size, n_intervals-1])
            batch = np.zeros((batch_end-batch_start, self.unet_window_length, num_channels))
            # loop over the examples in the batch to pull the windows from the continous data 
            for example_ind in np.arange(batch_start, batch_end):
                ex_start = start_indicies[example_ind]
                example = cont_data[ex_start:ex_start+self.unet_window_length].copy()

                example = self.process_waveform(example)
                batch[example_ind-batch_start, :, :] = example

            if num_channels == 3:
                if not debug_s:
                    p_posterior_probs[batch_start-1:batch_end-1, :] = self.p_detector3c.apply_model_to_batch(batch, center_window=self.center_window)
                s_posterior_probs[batch_start-1:batch_end-1, :], pre_sigmoid_output[batch_start-1:batch_end-1] = self.s_detector.apply_model_to_batch(batch, center_window=self.center_window)
            else:
                p_posterior_probs[batch_start-1:batch_end-1, :] = self.p_detector1c.apply_model_to_batch(batch, center_window=self.center_window)
            
            batch_start += batch_size

        if debug_s:
            bad_cont_data = []
            surrounding_cont_data = []
            bad_post_probs = []
            bad_presigmoid = []

        if num_channels == 3:
            # Handle nan values - probably will be handled with the next one
            nan_inds = np.unique(np.where(np.isnan(s_posterior_probs))[0])
            for bad_s_ind in nan_inds:
                print(f"{len(np.where(np.isnan(s_posterior_probs[bad_s_ind]))[0])} nan values in S window {bad_s_ind}")
                print("Setting all probabilites in this window to 0...")
                if debug_s:
                    # bad_cont_data.append(cont_data[start_indicies[bad_s_ind+1]:start_indicies[bad_s_ind+1]+self.unet_window_length])
                    # surrounding_cont_data.append(cont_data[start_indicies[bad_s_ind]:start_indicies[bad_s_ind]+self.unet_window_length])
                    # surrounding_cont_data.append(cont_data[start_indicies[bad_s_ind+2]:start_indicies[bad_s_ind+2]+self.unet_window_length])
                    bad_cont_data.append(processed_data[bad_s_ind, :, :])
                    surrounding_cont_data.append(processed_data[bad_s_ind-1,:,:])
                    surrounding_cont_data.append(processed_data[bad_s_ind+1, :, :])
                    bad_post_probs.append(np.concatenate([np.zeros(254), s_posterior_probs[bad_s_ind], np.zeros(254)]))
                    bad_presigmoid.append(pre_sigmoid_output[bad_s_ind].squeeze())

                s_posterior_probs[bad_s_ind, :] = np.zeros(s_posterior_probs.shape[1])

            assert ~np.any(np.isnan(s_posterior_probs))

        # Find probabilities that have too many 1.0 values - something wonky is going on and probabilities are garbage
            ones_inds, ones_cnts = np.unique(np.where(s_posterior_probs == 1.0)[0], return_counts=True)
            bad_ones_inds = ones_inds[ones_cnts > 5]
            if debug_s:
                for bad_s_ind in bad_ones_inds:
                    # bad_cont_data.append(cont_data[start_indicies[bad_s_ind+1]:start_indicies[bad_s_ind+1]+self.unet_window_length])
                    # surrounding_cont_data.append(cont_data[start_indicies[bad_s_ind]:start_indicies[bad_s_ind]+self.unet_window_length])
                    # surrounding_cont_data.append(cont_data[start_indicies[bad_s_ind+2]:start_indicies[bad_s_ind+2]+self.unet_window_length])
                    bad_cont_data.append(processed_data[bad_s_ind, :, :])
                    surrounding_cont_data.append(processed_data[bad_s_ind-1,:,:])
                    surrounding_cont_data.append(processed_data[bad_s_ind+1, :, :])
                    bad_post_probs.append(np.concatenate([np.zeros(254), s_posterior_probs[bad_s_ind], np.zeros(254)]))
                    bad_presigmoid.append(pre_sigmoid_output[bad_s_ind].squeeze())

            # Zero out the probabilities for those inds 
            s_posterior_probs[ones_inds[ones_cnts > 5]] = np.zeros(s_posterior_probs.shape[1])
            print(f"Zeroing {len(ones_inds[ones_cnts > 5])} S probabilites suspected to be bad...")
            print(f"{len(ones_inds[ones_cnts < 6])} other S probabilities that might be bad, but weren't removed")
            assert s_posterior_probs.shape[0] == n_intervals-2

            if debug_s:
                return None, (np.array(bad_cont_data), np.array(bad_post_probs), np.array(bad_presigmoid), np.array(nan_inds), 
                                np.array(bad_ones_inds), np.array(surrounding_cont_data))
            
            if debug_inds is not None:
                debug_cont = []
                debug_surround = []
                debug_post_probs = []
                debug_presigmoid = []
                for debug_ind in debug_inds:
                    debug_cont.append(processed_data[int(debug_ind), :, :])
                    debug_surround.append(processed_data[debug_ind-1,:,:])
                    debug_surround.append(processed_data[debug_ind+1, :, :])
                    debug_post_probs.append(np.concatenate([np.zeros(254), s_posterior_probs[debug_ind], np.zeros(254)]))
                    debug_presigmoid.append(pre_sigmoid_output[debug_ind].squeeze())

                return None, (np.array(debug_cont), np.array(debug_post_probs), np.array(debug_presigmoid), np.array(debug_surround), debug_inds)

            return p_posterior_probs, s_posterior_probs

        return p_posterior_probs, None
            
    def apply_detectors_edgecases(self, cont_data, n_intervals, start_indicies, num_channels=3):
        """
        Doc string 
        """
        edge_cases = np.zeros((2, self.unet_window_length, num_channels))
        cnt = 0
        for example_ind in [0, n_intervals-1]:
            ex_start = start_indicies[example_ind]
            example = cont_data[ex_start:ex_start+self.unet_window_length].copy()
            example = self.process_waveform(example)
            # example = example/np.max(abs(example), axis=0)
            edge_cases[cnt, :, :] = example
            cnt += 1
        if num_channels == 3:
            p_edge_case_probs = self.p_detector3c.apply_model_to_batch(edge_cases, center_window=None)
            s_edge_case_probs, _ = self.s_detector.apply_model_to_batch(edge_cases, center_window=None)
            return p_edge_case_probs, s_edge_case_probs

        p_edge_case_probs = self.p_detector1c.apply_model_to_batch(edge_cases, center_window=None)
        return p_edge_case_probs, None

    def get_detector_picks(self, cont_data, n_intervals, batch_size=None, num_channels=3, save_probs=None, debug_s=False, debug_inds=None):
        if batch_size is None:
            batch_size = self.batch_size

        print("Running detectors...")
        npts = cont_data.shape[0]
        start_indicies = np.arange(0, npts-2*self.sliding_interval, self.sliding_interval)
        assert len(start_indicies) == n_intervals

        inner_p_posterior_probs, inner_s_posterior_probs = self.apply_detectors_nonedgecases(cont_data, n_intervals, batch_size, start_indicies, 
                                                                    num_channels=num_channels, debug_s=debug_s, debug_inds=debug_inds)
        
        if debug_s or debug_inds is not None:
            return inner_p_posterior_probs, inner_s_posterior_probs

        p_edge_case_probs, s_edge_case_probs = self.apply_detectors_edgecases(cont_data, n_intervals, start_indicies, num_channels=num_channels)
        
        all_p_posterior_probs = self.concat_probs(p_edge_case_probs, inner_p_posterior_probs, npts)
        # pick_widths are forced to be atleast 20, so probably not a good indicator of quality, unless they are large (not in updated)
        p_pick_probs, p_pick_inds, p_pick_widths = self.p_detector3c.sliding_window_phase_arrival_estimator_updated(all_p_posterior_probs, 100, thresh=self.p_detection_thresh)
        ppick_dict = self.detector_results_to_dict(p_pick_probs, p_pick_inds, p_pick_widths)
        print(f'Found {len(p_pick_inds)} P arrivals above {self.p_detection_thresh} threshold')

        if num_channels == 3:
            all_s_posterior_probs = self.concat_probs(s_edge_case_probs, inner_s_posterior_probs, npts)
            s_pick_probs, s_pick_inds, s_pick_widths = self.s_detector.sliding_window_phase_arrival_estimator_updated(all_s_posterior_probs, 100, thresh=self.s_detection_thresh)
            spick_dict = self.detector_results_to_dict(s_pick_probs, s_pick_inds, s_pick_widths)
            print(f'Found {len(s_pick_inds)} S arrivals above {self.s_detection_thresh} threshold')
            if save_probs is not None:
                file = h5py.File(os.path.join(self.output_dir, save_probs), "w")
                file.create_dataset("p_proba", data=all_p_posterior_probs)
                file.create_dataset("s_proba", data=all_s_posterior_probs)
                file.close()

            return ppick_dict, spick_dict

        if save_probs is not None:
                file = h5py.File(os.path.join(self.output_dir, save_probs), "w")
                file.create_dataset("p_proba", data=all_p_posterior_probs)
                file.close()

        return ppick_dict, None

    @staticmethod
    def detector_results_to_dict(pick_probs, pick_inds, pick_widths):
        results_dict = {
            "pick_probs": pick_probs, 
            "pick_inds": pick_inds, 
            "pick_widths":pick_widths
        }

        return results_dict

    def concat_probs(self, edge_probs, inner_probs, npts):
        edge_case_width = (self.unet_window_length - 2*self.center_window)//2 + 2*self.center_window
        all_posterior_probs = np.concatenate([edge_probs[0, 0:edge_case_width], inner_probs.flatten(), edge_probs[1, -edge_case_width:]])
        #Remove this line for debugging
        assert len(all_posterior_probs) == npts
        return all_posterior_probs

    def apply_cnn(self, model, phase, cont_data, pick_inds, batch_size, n_classes=1):
        if phase == "P":
            window_length=self.pcnn_window_length
            chan_ind = [cont_data.shape[1]-1]
            num_channels = 1
        else:
            window_length = self.scnn_window_length
            chan_ind = range(3)
            num_channels = 3

        n_picks = len(pick_inds)
        model_outputs = np.zeros((n_picks, n_classes))
        waveform_halfwidth = (window_length//2)
        batch_start = 0
        while batch_start < n_picks:
            batch_end = np.min([batch_start+batch_size, n_picks])
            batch = np.zeros((batch_end-batch_start, window_length, num_channels))
            # loop over the examples in the batch to pull the windows from the continous data 
            for example_ind in np.arange(batch_start, batch_end):
                processing_start = int(pick_inds[example_ind]) - waveform_halfwidth - 100
                processing_end = processing_start + window_length + 200
                if processing_start < 0:
                    example_preproc = cont_data[0:processing_end, chan_ind].copy()
                elif processing_end > len(cont_data):
                    example_preproc = cont_data[processing_start:, chan_ind].copy()
                else:
                    example_preproc = cont_data[processing_start:processing_end, chan_ind].copy()                

                # Don't normalize here - normalize after trimming to the appropriate length 
                example_proc = self.process_waveform(example_preproc, normalize=False)

                ex_start = int(pick_inds[example_ind]) - waveform_halfwidth
                # If the pick is close to the beginning, pad the start of the waveform with zeros
                # TODO: I don't know why I padded with ones before but it seems wrong, switching to zeros
                if processing_start < 0: #ex_start < 0:
                    #pad_zeros = np.zeros((-ex_start, num_channels))
                    if ex_start < 0:
                        pad = np.zeros((-ex_start, num_channels)) * example_proc[0, :]
                        example_proc = np.concatenate([pad, example_proc[0:-100]])
                    else:
                        # processing start is negative
                        start_ind = 100 + (processing_start)
                        example_proc = example_proc[start_ind:-100]
                elif processing_end > len(cont_data): #ex_start + window_length > len(cont_data):
                    # try pad = np.ones(correct_size)*cont_data[-1, :]
                    if ex_start + window_length > len(cont_data):
                        pad = np.zeros((ex_start + window_length - len(cont_data), num_channels)) * example_proc[-1, :]
                        example_proc= np.concatenate([example_proc[100:], pad])
                    else:
                        end_ind = 100 - (processing_end - len(cont_data))
                        example_proc = example_proc[100:-end_ind]
                else:
                    example_proc = example_proc[100:-100]

                example_norm = self.normalize(example_proc)

                batch[example_ind-batch_start, :] = example_norm

            model_outputs[batch_start:batch_end] = model.apply_model_to_batch(batch)
            batch_start += batch_size

        return model_outputs

    def apply_swag(self, models, phase, cont_data, pick_inds, N):
        if phase == "P":
            window_length=self.pcnn_window_length
            chan_ind = [cont_data.shape[1]-1]
            num_channels = 1
            train_loader = self.swag_p_loaders["train"]
        else:
            window_length = self.scnn_window_length
            chan_ind = range(3)
            num_channels = 3
            train_loader = self.swag_s_loaders["train"]

        n_picks = len(pick_inds)
        processed_input = np.zeros((n_picks, window_length, num_channels))
        waveform_halfwidth = (window_length//2)
        for example_ind in np.arange(n_picks):
            processing_start = int(pick_inds[example_ind]) - waveform_halfwidth - 100
            processing_end = processing_start + window_length + 200
            if processing_start < 0:
                example_preproc = cont_data[0:processing_end, chan_ind].copy()
            elif processing_end > len(cont_data):
                example_preproc = cont_data[processing_start:, chan_ind].copy()
            else:
                example_preproc = cont_data[processing_start:processing_end, chan_ind].copy()                

            # Don't normalize here - normalize after trimming to the appropriate length 
            example_proc = self.process_waveform(example_preproc, normalize=False)

            ex_start = int(pick_inds[example_ind]) - waveform_halfwidth
            # If the pick is close to the beginning, pad the start of the waveform with zeros
            # TODO: I don't know why I padded with ones before but it seems wrong, switching to zeros
            if processing_start < 0: #ex_start < 0:
                #pad_zeros = np.zeros((-ex_start, num_channels))
                if ex_start < 0:
                    pad = np.zeros((-ex_start, num_channels)) * example_proc[0, :]
                    example_proc = np.concatenate([pad, example_proc[0:-100]])
                else:
                    # processing start is negative
                    start_ind = 100 + (processing_start)
                    example_proc = example_proc[start_ind:-100]
            elif processing_end > len(cont_data): #ex_start + window_length > len(cont_data):
                # try pad = np.ones(correct_size)*cont_data[-1, :]
                if ex_start + window_length > len(cont_data):
                    pad = np.zeros((ex_start + window_length - len(cont_data), num_channels)) * example_proc[-1, :]
                    example_proc= np.concatenate([example_proc[100:], pad])
                else:
                    end_ind = 100 - (processing_end - len(cont_data))
                    example_proc = example_proc[100:-end_ind]
            else:
                example_proc = example_proc[100:-100]

            example_norm = self.normalize(example_proc)
            processed_input[example_ind] = example_norm

        # Pass all data to SWAG model instead of one batch at a time
        # Iterate over the various SWAG models in the ensemble 
        dset = Dset(processed_input)
        dset_loader = torch.utils.data.DataLoader(
            dset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        ensemble_outputs = np.zeros((processed_input.shape[0], N*len(models)))
        for i, model in enumerate(models):
            ensemble_outputs[:, i*N:i*N+N] = model.apply_model(dset_loader, N, train_loader)

        return ensemble_outputs

    def process_waveform(self, waveform_in, normalize=True):

        num_channels = waveform_in.shape[-1]
        obs_ex = obspy.Stream()
        for chan in range(num_channels):
            tr = obspy.Trace(waveform_in[:, chan])
            tr.stats.sampling_rate = 100.0
            obs_ex += tr
            
        obs_ex.detrend('linear')
        # obs_ex.detrend('demean')
        obs_ex.taper(0.01, type="cosine")
        obs_ex.filter("bandpass", freqmin=1.0, freqmax=17.0, corners=2)

        waveform_out = np.zeros_like(waveform_in)
        for chan in range(num_channels):
            waveform_out[:, chan] = obs_ex[chan].data
        
        ## When just looking at adding BP filter here, had only this line + other processing in self.preprocess_data ###
        ## example = self.obspy_bandpass(example, freqmin=1.0, freqmax=17.0, corners=2)
        
        if normalize:
            # normalize the data for the window 
            waveform_out = self.normalize(waveform_out)

        return waveform_out
    
    @staticmethod
    def normalize(waveform):
        # normalize the data for the window 
        norm_vals = np.max(abs(waveform), axis=0)
        norm_vals_inv = np.zeros_like(norm_vals)
        for nv_ind in range(len(norm_vals)):
            nv = norm_vals[nv_ind]
            if abs(nv) > 1e-4:
                norm_vals_inv[nv_ind] = 1/nv

        return waveform*norm_vals_inv

    def get_p_pick_corrections(self, cont_data, pick_inds, batch_size=None, N=None):
        if batch_size is None:
            batch_size = self.batch_size
        print("Calculating P pick corrections...")
        if N is None:
            pick_corrections = self.apply_cnn(self.ppicker, "P", cont_data, pick_inds, batch_size, n_classes=1).squeeze()
        else:
            pick_corrections = self.apply_swag(self.swag_ppickers, "P", cont_data, pick_inds, N)
        return pick_corrections

    def get_s_pick_corrections(self, cont_data, pick_inds, batch_size=None, N=None):
        if batch_size is None:
            batch_size = self.batch_size
        print("Calculating S pick corrections...")
        if N is None:
            pick_corrections = self.apply_cnn(self.spicker, "S", cont_data, pick_inds, batch_size, n_classes=1).squeeze()
        else:
            pick_corrections = self.apply_swag(self.swag_spickers, "S", cont_data, pick_inds, N)            
        return pick_corrections

    def get_fm_information(self, cont_data, pick_inds, pick_corrections, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        print("Calculating first motions...")
        updated_pick_inds = np.round(pick_inds + pick_corrections*100).astype("int")
        fm_probs = self.apply_cnn(self.fmpicker, "P", cont_data, updated_pick_inds, batch_size, n_classes=3)
        polarity = self.fmpicker.get_polarity()
        fm_predictions = polarity[np.argmax(fm_probs[:, :], axis=1)]

        return fm_predictions, fm_probs

    @staticmethod
    def calculate_absolute_arrival_times(starttime_epoch, pick_inds, pick_corrections=None):
        if pick_corrections is None:
            arrival_times = starttime_epoch + pick_inds*0.01 
        else:
            arrival_times = starttime_epoch + (pick_inds*0.01 + pick_corrections)

        return arrival_times

    def update_results(self, tr_stats, pick_dict, phase_type, pick_ids, pick_corrections=None, fm_predictions=None, fm_probs=None, num_channels=3, swag_info=None):
        arrival_times = pick_dict["arrival_times"]
        pick_probs = pick_dict["pick_probs"]
        pick_widths = pick_dict["pick_widths"]
        if num_channels == 3:
            channel = f'{tr_stats["channel"][:-1]}'
        else:
            channel = tr_stats["channel"]

        n_picks = len(arrival_times)
        for pick_ind in range(n_picks):
            pick_summary = {
                "network": tr_stats["network"], 
                "station":tr_stats["station"],
                "location":tr_stats["location"],
                "channel": channel,
                "phase_type":phase_type,
                "arrival_time": arrival_times.item(pick_ind),
                "detection_probability": pick_probs.item(pick_ind),
                "approximate_width": pick_widths.item(pick_ind),
                "pick_id": pick_ids.item(pick_ind)
            }
            if pick_corrections is not None:
                pick_summary["applied_pick_correction"] = pick_corrections.item(pick_ind)

            if swag_info is not None:
                pick_summary["pick_lb"] = swag_info["lb"].item(pick_ind)
                pick_summary["pick_ub"] = swag_info["ub"].item(pick_ind)
                pick_summary["pick_std"] = swag_info["std"].item(pick_ind)

            if fm_predictions is not None:
                # polarity=[1,-1,0]
                pick_summary["fm_prediction"] = int(fm_predictions[pick_ind])
                pick_summary["fm_probability_up"] = fm_probs[pick_ind, 0]
                pick_summary["fm_probability_down"] = fm_probs[pick_ind, 1]
                pick_summary["fm_probability_unknown"] = fm_probs[pick_ind, 2]

                #"fm_prob_ind":np.argmax(fm_probs[pick_ind, :])
            
            self.results.append(pick_summary)
        
    def save_results_to_csv(self):
        outname = os.path.join(self.output_dir, f"{self.catalog_file}.csv")
        print(f"Saving results to {outname}")
        results_df = pd.DataFrame(self.results)
        results_df = results_df.sort_values("arrival_time")
        results_df.to_csv(outname, index=False, float_format="%.7f")

    def calibrate_swag_predictions(self, predictions, lb_transform, ub_transform):
        if self.picker_summary_method == "mean":
            y_pred = np.mean(predictions, axis=1)
        elif self.picker_summary_method == "median":
            y_pred = np.median(predictions, axis=1)
        pred_std = np.std(predictions, axis=1)
        df_data = {"y_pred":y_pred, "std":pred_std}
        df = pd.DataFrame(data=df_data)
        y_lb = df.apply(lambda x: norm.ppf(lb_transform, x["y_pred"], x["std"]), axis=1).values
        y_ub = df.apply(lambda x: norm.ppf(ub_transform, x["y_pred"], x["std"]), axis=1).values

        summary = {"y_pred":y_pred, "std": pred_std, "lb": y_lb, "ub": y_ub}

        return summary

    def save_swag_predictions(self, predictions, station, date, phase, pick_ids):
        pred_outdir = os.path.join(self.output_dir, "swag_predictions")
        if not os.path.exists(pred_outdir):
            print("Making director", pred_outdir)
            os.makedirs(pred_outdir)
        outfile = f"{pred_outdir}/{station.split('*')[0]}.{station.split('*')[1]}.{date}.{phase}.predictions"
        np.savez(outfile, predictions=predictions, pick_ids=pick_ids)

    def apply_to_data(self):
        if self.debug_inds_file is not None and self.debug_s_detector:
            raise ValueError("Cannot debug s detector and output indicies of interest, change one to False/None")

        if self.debug_inds_file is not None:
            debug_inds = np.loadtxt(self.debug_inds_file, dtype="int")
        else:
            debug_inds = None

        # Get the unique starting dates of files
        dates = set()
        for file in glob.glob(f"{self.data_dir}/*mseed"):
            date = file.split("__")
            dates.add(date[1])
        dates = np.sort(list(dates))
    
        #dates = dates[-1:] 
        dates = [dates[5]] # TODO: REMOVE THIS
        print(dates)
        # Iterate over the dates
        for date in dates:
            print(f"Starting on date from {date}...")
            # Get the station names and a count from data_dir file names for the given date.
            # If only intersted in one station, don't do all that counting
            station_names = {}
            stat_name_ind = 1
            if self.data_dir[0] == ".":
                stat_name_ind = 2
            for file in glob.glob(f"{self.data_dir}/*__{date}__*mseed"):
                stat_name = file.split(".")[stat_name_ind]
                chan = file.split(".")[stat_name_ind+2].split("__")[0][:-1]
                name = f"{stat_name}*{chan}"
                if name in station_names.keys():
                    station_names[name] += 1
                else:
                    station_names[name] = 1

            print(f"Found {len(station_names.keys())} station/channel pairs")
            station_name_list = station_names.keys()
            if self.single_stat_string is not None:
                station_name_list = [self.single_stat_string]

            print(station_name_list)
            for stat in station_name_list:
                # Check the number of channels for the given station
                if station_names[stat] == 1:
                    num_channels = 1
                elif station_names[stat] == 3:
                    num_channels = 3
                else:
                    print(f"Wrong number of channels, skipping {stat_name}...")
                    continue
                #assert station_names[stat] == 3, f"Must have 3 channels, not {station_names[stat]}"

                #### Go through all processing for 1, 3C station ####
                station_file_name = f'{self.data_dir}/*{stat}*__{date}__*.mseed'
                st_preproc, n_intervals = self.load_continuous(station_file_name, num_channels=num_channels)
                if st_preproc is None:
                    print("Skipping data from this station for this day...")
                    continue
                
                if num_channels == 3:
                    data_tensor = self.stream_to_tensor_3c(st_preproc)
                    pdetector_results, sdetector_results = self.get_detector_picks(data_tensor, n_intervals, num_channels=num_channels, 
                    save_probs=self.output_probability_file, debug_s=self.debug_s_detector, debug_inds=debug_inds)                    
                    if not self.debug_s_detector:
                        spick_ids = np.arange(len(self.results)+len(pdetector_results["pick_inds"]), len(self.results)+len(sdetector_results["pick_inds"]))

                    if self.debug_s_detector:
                        print(len(self.results))
                        print(len(sdetector_results[0]))

                        spick_ids = np.arange(len(self.results), len(self.results)+len(sdetector_results[0]))
                        f = h5py.File(os.path.join(self.output_dir, f'{self.catalog_file}.{date}.h5'), "w")
                        f.create_dataset("continuous_data", data=sdetector_results[0])
                        f.create_dataset("posterior_probs", data=sdetector_results[1])
                        f.create_dataset("presigmoid_output", data=sdetector_results[2])
                        f.create_dataset("nan_inds", data=sdetector_results[3])
                        f.create_dataset("five_one_inds", data=sdetector_results[4])
                        f.create_dataset("surrounding_cont_data", data=sdetector_results[5])
                        f.close()
                        continue
                    elif debug_inds is not None:
                        f = h5py.File(os.path.join(self.output_dir, f'{self.catalog_file}.{date}.h5'), "w")
                        f.create_dataset("continuous_data", data=sdetector_results[0])
                        f.create_dataset("posterior_probs", data=sdetector_results[1])
                        f.create_dataset("presigmoid_output", data=sdetector_results[2])
                        f.create_dataset("surrounding_cont_data", data=sdetector_results[3])
                        f.create_dataset("indicies", data=sdetector_results[4])
                        f.close()
                        continue
                else:
                    data_tensor = self.stream_to_tensor_1c(st_preproc)
                    pdetector_results, _ = self.get_detector_picks(data_tensor, n_intervals, num_channels=num_channels, save_probs=self.output_probability_file)
                
                ppick_ids = np.arange(len(self.results), len(self.results)+len(pdetector_results["pick_inds"]))

                if not self.use_swag:
                    ppick_corrections = self.get_p_pick_corrections(data_tensor, pdetector_results["pick_inds"])
                    fm_predictions, fm_probs = self.get_fm_information(data_tensor, pdetector_results["pick_inds"], ppick_corrections)
                    pdetector_results["arrival_times"] = self.calculate_absolute_arrival_times(st_preproc[0].stats.starttime_epoch, 
                                                                                                    pdetector_results["pick_inds"], 
                                                                                                    ppick_corrections)
                    self.update_results(st_preproc[0].stats, pdetector_results, "P", ppick_ids, ppick_corrections, fm_predictions, fm_probs, num_channels=num_channels)
                else:
                    ensemble_ppick_corrections = self.get_p_pick_corrections(data_tensor, pdetector_results["pick_inds"], N=self.N_P)
                    ppick_stats = self.calibrate_swag_predictions(ensemble_ppick_corrections, self.swag_p_lb, self.swag_p_ub)
                    fm_predictions, fm_probs = self.get_fm_information(data_tensor, pdetector_results["pick_inds"], ppick_stats["y_pred"])
                    pdetector_results["arrival_times"] = self.calculate_absolute_arrival_times(st_preproc[0].stats.starttime_epoch, 
                                                                                                    pdetector_results["pick_inds"], 
                                                                                                    ppick_stats["y_pred"])
                
                    self.update_results(st_preproc[0].stats, pdetector_results, "P", ppick_ids, ppick_stats["y_pred"], fm_predictions, fm_probs, num_channels=num_channels,
                                            swag_info=ppick_stats)
                    self.save_swag_predictions(ensemble_ppick_corrections, stat, date, "P", ppick_ids)

                if num_channels == 3:
                    if not self.use_swag:
                        spick_corrections = self.get_s_pick_corrections(data_tensor, sdetector_results["pick_inds"])
                        sdetector_results["arrival_times"] = self.calculate_absolute_arrival_times(st_preproc[0].stats.starttime_epoch, 
                                                                                                    sdetector_results["pick_inds"], 
                                                                                                    spick_corrections)
                        self.update_results(st_preproc[0].stats, sdetector_results, "S", spick_ids, spick_corrections)
                    else:
                        ensemble_spick_corrections = self.get_s_pick_corrections(data_tensor, sdetector_results["pick_inds"], N=self.N_S)
                        spick_stats = self.calibrate_swag_predictions(ensemble_spick_corrections, self.swag_s_lb, self.swag_s_ub)
                        sdetector_results["arrival_times"] = self.calculate_absolute_arrival_times(st_preproc[0].stats.starttime_epoch, 
                                                                                                    sdetector_results["pick_inds"], 
                                                                                                    spick_stats["y_pred"])

                        self.update_results(st_preproc[0].stats, sdetector_results, "S", spick_ids, spick_stats["y_pred"], num_channels=num_channels,swag_info=spick_stats)
                        self.save_swag_predictions(ensemble_spick_corrections, stat, date, "S", spick_ids)

                self.save_results_to_csv()

# if __name__ == "__main__":
#     #### Set Parameters and Initialize the Classes ####

#     #from apply_models_config import CFG
#     # TODO: remove this
#     from apply_models_sfailure_config import CFG
#     applier = apply_models(CFG)
#     applier.apply_to_data()
