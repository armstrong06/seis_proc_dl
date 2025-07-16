import torch
import numpy as np
import h5py
from joblib import load
from scipy.stats import norm
import os
import sys
import time
import logging
from abc import ABC
from sqlalchemy import text
from seis_proc_dl.apply_to_continuous.database_connector import SwagPickerDBConnection

sys.path.append(
    "/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/atucus/swag_modified"
)
from swag import models as swag_models
from swag import utils as swag_utils
from swag.posteriors import SWAG

sys.path.append(
    "/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/mlmodels/intel_cpu_build"
)
# TODO: Better way to import pyuussmlmodels than adding path?
import pyuussmlmodels

# Followed this tutorial https://betterstack.com/community/guides/logging/how-to-start-logging-with-python/
# Just a simple logger for now
logger = logging.getLogger("apply_swag_pickers")
stdoutHandler = logging.StreamHandler(stream=sys.stdout)
fmt = logging.Formatter(
    "%(name)s: %(asctime)s | %(levelname)-8s | %(filename)-25s:%(lineno)-4d | %(process)7d >>> %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
stdoutHandler.setFormatter(fmt)
logger.addHandler(stdoutHandler)
logger.setLevel(logging.DEBUG)


class SwagPicker:
    def __init__(
        self, model_name, checkpoint_file, seed, cov_mat=True, K=20, device="cuda:0"
    ):
        torch.backends.cudnn.benchmark = True

        eps = 1e-12
        model_cfg = getattr(swag_models, model_name)
        self.cov_mat = cov_mat
        self.seed = seed

        self.model = SWAG(
            model_cfg.base,
            no_cov_mat=not cov_mat,
            max_num_models=K,
            *model_cfg.args,
            **model_cfg.kwargs,
        )

        # self.model.cuda()
        self.device = device
        self.model.to(self.device)
        map_location = None
        if self.device == "cpu":
            map_location = torch.device("cpu")

        logger.info("Loading model %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file, map_location=map_location)
        self.model.load_state_dict(checkpoint["state_dict"])
        # self.batchsize = batchsize

    def apply_model(self, dset_loader, N, train_loader, scale=0.5):

        # TODO: I am not sure this is really needed at inference time
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        predictions = np.zeros((len(dset_loader.dataset), N))
        for i in range(N):
            self.model.sample(scale=scale, cov=self.cov_mat)
            swag_utils.bn_update(train_loader, self.model)
            self.model.eval()
            k = 0
            for input in dset_loader:
                # input = input.cuda(non_blocking=True)
                input = input.to(self.device, non_blocking=True)
                torch.manual_seed(i)
                output = self.model(input)
                with torch.no_grad():
                    predictions[k : k + input.size()[0], i : i + 1] = (
                        output.cpu().numpy()
                    )
                k += input.size()[0]

        return predictions


class Dset(torch.utils.data.Dataset):
    def __init__(self, data):
        logger.info(f"Dset shape: {data.shape}")
        self.data = torch.from_numpy(data.transpose((0, 2, 1))).float()

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)


class BaseMultiSWAGPicker(ABC):
    def __init__(self, is_p_picker, device="cuda:0") -> None:
        if is_p_picker:
            self.phase = "P"
        else:
            self.phase = "S"

        self.device = device
        self.swag_model_dir = None
        self.cal_model_file = None

    ## GENERAL FUNCTIONS USED BY WHEN USING DISK OR DB ##
    def load_swag_ensemble(
        self, model1, model2, model3, seeds, cov_mat, k, swag_model_dir=None
    ):

        if swag_model_dir is None:
            if self.swag_model_dir is None:
                logger.warning("Swag model dir needs to be specified")
                return
            swag_model_dir = self.swag_model_dir

        modelname = f"{self.phase}Picker"
        swag1 = SwagPicker(
            modelname,
            os.path.join(swag_model_dir, model1),
            seeds[0],
            cov_mat=cov_mat,
            K=k,
            device=self.device,
        )
        swag2 = SwagPicker(
            modelname,
            os.path.join(swag_model_dir, model2),
            seeds[1],
            cov_mat=cov_mat,
            K=k,
            device=self.device,
        )
        swag3 = SwagPicker(
            modelname,
            os.path.join(swag_model_dir, model3),
            seeds[2],
            cov_mat=cov_mat,
            K=k,
            device=self.device,
        )

        return [swag1, swag2, swag3]

    @staticmethod
    def apply_picker(models, cont_loader, train_loader, N):
        n_examples = cont_loader.dataset.data.shape[0]
        ensemble_outputs = np.zeros((n_examples, N * len(models)))
        total_st = time.time()
        for i, model in enumerate(models):
            st = time.time()
            ensemble_outputs[:, i * N : i * N + N] = model.apply_model(
                cont_loader, N, train_loader
            )
            et = time.time()
            logger.debug(
                f"Average time per batch for model {i}: {(et-st)/len(cont_loader):3.2f} s"
            )
            logger.debug(f"Average time per sample for model {i}: {(et-st)/N:3.2f} s")

        total_et = time.time()
        logger.debug(f"Total time to apply all models {total_et-total_st:4.2f} s")
        return ensemble_outputs

    def get_calibrated_pick_bounds(self, lb, ub, cal_model_file=None):
        if cal_model_file is None:
            if self.cal_model_file is None:
                logger.warning("Calibration model file needs to be specified")
                return
            cal_model_file = self.cal_model_file
        # Transform the lower and upper bounds to be calibrated
        iso_reg_inv = load(cal_model_file)
        lb_transform = iso_reg_inv.transform([lb])[0]
        ub_transform = iso_reg_inv.transform([ub])[0]
        return lb_transform, ub_transform

    @staticmethod
    def calibrate_swag_predictions(y_pred, pred_std, lb_transform, ub_transform):
        y_pred = np.array(y_pred)
        pred_std = np.array(pred_std)
        assert np.all(pred_std > 0), "Standard deviations should be positive"

        # df_data = {"y_pred": y_pred, "std": pred_std}
        # df = pd.DataFrame(data=df_data)
        # y_lb = df.apply(
        #     lambda x: norm.ppf(lb_transform, x["y_pred"], x["std"]), axis=1
        # ).values
        # y_ub = df.apply(
        #     lambda x: norm.ppf(ub_transform, x["y_pred"], x["std"]), axis=1
        # ).values
        # Don't use pandas because don't need it for anything else if using the DB
        def get_bounds(p, loc, scale):
            return norm.ppf(p, loc, scale)

        np_func = np.vectorize(get_bounds, otypes=[float])
        y_lb = np_func(lb_transform, y_pred, pred_std)
        y_ub = np_func(ub_transform, y_pred, pred_std)

        summary = {
            "arrivalTimeShift": y_pred,
            "arrivalTimeShiftSTD": pred_std,
            "arrivalTimeShiftLowerBound": y_lb,
            "arrivalTimeShiftUpperBound": y_ub,
        }

        return summary

    @staticmethod
    def torch_loader(
        filename, path, batch_size, num_workers, shuffle=False, n_examples=-1
    ):

        with h5py.File(f"{path}/{filename}", "r") as f:
            X = f["X"][:]

        if len(X.shape) < 3:
            X = np.expand_dims(X, 2)

        if n_examples > 0:
            X = X[:n_examples, :, :]

        dset = Dset(X)
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        return loader

    ####


class MultiSWAGPicker(BaseMultiSWAGPicker):

    ## FUNCTIONS WHEN SAVING/LOADING TO/DISK ##
    @staticmethod
    def trim_inner_fence(ensemble_predictions):
        q1, q3 = np.percentile(ensemble_predictions, [25, 75], axis=1)
        iqr = q3 - q1
        if1 = q1 - 1.5 * iqr
        if3 = q3 + 1.5 * iqr
        ensemble_stds = np.zeros(ensemble_predictions.shape[0])
        ensemble_medians = np.zeros(ensemble_predictions.shape[0])
        for i in range(len(ensemble_predictions)):
            trimmed = ensemble_predictions[i][
                np.where(
                    np.logical_and(
                        ensemble_predictions[i] > if1[i],
                        ensemble_predictions[i] < if3[i],
                    )
                )
            ]
            ensemble_stds[i] = np.std(trimmed)
            ensemble_medians[i] = np.median(trimmed)

        return ensemble_medians, ensemble_stds

    def load_data(
        self,
        train_file,
        train_path,
        new_data_file,
        new_data_path,
        batchsize=256,
        num_workers=2,
        shuffle_train=False,
    ):
        train_loader = self.torch_loader(
            train_file, train_path, batchsize, num_workers, shuffle_train
        )
        cont_loader = self.torch_loader(
            new_data_file, new_data_path, batchsize, num_workers, shuffle=False
        )
        return train_loader, cont_loader

    def format_and_save(
        self,
        meta_csv_file,
        pred_summary,
        all_predictions,
        outfile_pref,
        region,
        n_meta_rows=-1,
    ):
        import pandas as pd

        columns = [
            "eventIdentifier",
            "network",
            "station",
            "channel",
            "locationCode",
            "phase",
        ]
        if self.phase == "S":
            columns = [
                "eventIdentifier",
                "network",
                "station",
                "verticalChannel",
                "locationCode",
                "phase",
            ]
        meta_df = pd.read_csv(meta_csv_file)[columns]
        if n_meta_rows > 0:
            meta_df = meta_df.iloc[0:n_meta_rows]
        summary_df = pd.DataFrame(pred_summary)
        meta_df = meta_df.join(summary_df)
        # meta_df.loc[:, 'correctedArrivalTime'] = meta_df['estimateArrivalTime'] + meta_df['arrivalTimeShift']
        csv_outfile = os.path.join(
            outfile_pref, f"corrections.{self.phase.lower()}Arrivals.{region}.csv"
        )
        h5_outfile = os.path.join(
            outfile_pref, f"corrections.{self.phase.lower()}Arrivals.{region}.h5"
        )
        logger.info(f"Writing {csv_outfile} and {h5_outfile}")

        meta_df.to_csv(csv_outfile, index=False, float_format="%0.6f")
        with h5py.File(h5_outfile, "w") as f:
            f.create_dataset("X", shape=all_predictions.shape, data=all_predictions)

    ####


class MultiSWAGPickerDB(BaseMultiSWAGPicker):
    def __init__(
        self, is_p_picker, swag_model_dir, cal_model_file, device="cuda:0"
    ) -> None:

        super().__init__(is_p_picker, device)

        self.swag_model_dir = swag_model_dir
        self.cal_model_file = cal_model_file
        self.db_conn = None
        self.cal_loc_type = None
        self.cal_scale_type = None

        if self.phase == "S":
            self.proc_fn = self.process_3c_S
            self.threeC_waveforms = True
        else:
            self.proc_fn = self.process_1c_P
            self.threeC_waveforms = False

    ## FUNCTIONS WHEN USING DB ##
    def start_db_conn(
        self,
        repicker_dict,
        cal_dict,
        session_factory=None,
        cal_loc_type="trim_median",
        cal_scale_type="trim_std",
    ):
        self.db_conn = SwagPickerDBConnection(
            self.phase, session_factory=session_factory
        )

        repicker_params = repicker_dict["params"]
        repicker_params["n_comps"] = 3 if self.threeC_waveforms else 1
        repicker_params["wf_proc_fn_name"] = self.proc_fn.__name__

        with self.db_conn.Session() as session:
            with session.begin():
                # Get/set repicker method
                self.db_conn.add_repicker_method(
                    session,
                    repicker_dict["name"],
                    repicker_dict["desc"],
                    self.swag_model_dir,
                    addition_params_dict=repicker_params,
                )

                # Get/set calibration method
                self.db_conn.add_calibration_method(
                    session,
                    cal_dict["name"],
                    cal_dict["desc"],
                    self.cal_model_file,
                    loc_type=cal_loc_type,
                    scale_type=cal_scale_type,
                )

        self.cal_loc_type = cal_loc_type
        self.cal_scale_type = cal_scale_type

    def torch_loader_from_db(
        self,
        n_samples,
        batch_size,
        num_workers,
        start_date,
        end_date,
        wf_source_list,
        padding=0,
        no_proc=False,
    ):

        proc_fn = self.proc_fn
        if no_proc:
            logger.warning(
                "Turning the waveform processing function off. The RepickerMethod.wf_proc_fn_name will be incorrect!"
            )
            proc_fn = None

        with self.db_conn.Session() as session:
            ids, X = self.db_conn.get_waveforms(
                session,
                n_samples=n_samples,
                threeC_waveforms=self.threeC_waveforms,
                proc_fn=proc_fn,
                start=start_date,
                end=end_date,
                wf_source_list=wf_source_list,
                padding=padding,
                on_event=logger.info,
            )

        dset = Dset(X)
        loader = torch.utils.data.DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        return ids, loader

    def get_multiple_cis(self, y_pred, pred_std, percents):

        all_cal_results = {}
        for p in percents:
            lb_transform, ub_transform = self.get_calibrated_pick_bounds_percent(p)
            cal_dict = self.calibrate_swag_predictions(
                y_pred, pred_std, lb_transform, ub_transform
            )
            all_cal_results[p] = cal_dict

        return all_cal_results

    @staticmethod
    def _get_lb_ub_from_percent(percent):
        dec = percent * 0.01
        d = (1 - dec) / 2
        ub = 1 - d

        return round(d, 2), round(ub, 2)

    def get_calibrated_pick_bounds_percent(self, percent):
        assert percent > 1 and percent < 100, "percent should be > 1 and < 100"
        lb, ub = self._get_lb_ub_from_percent(percent)
        return self.get_calibrated_pick_bounds(lb, ub)

    def calibrate_and_save(
        self, pick_source_ids, ensemble_outputs, ci_percents, start_date, end_date
    ):

        t0 = time.time()
        summary_stats = self.get_summary_stats(ensemble_outputs)
        calibration_results = self.get_multiple_cis(
            summary_stats[self.cal_loc_type],
            summary_stats[self.cal_scale_type],
            ci_percents,
        )
        t1 = time.time()
        logger.debug(f"Total time calibrating and computing stats: {t1-t0:0.2f} s")

        t0 = time.time()

        with self.db_conn.Session() as session:
            with session.begin():
                session.execute(text("SET innodb_lock_wait_timeout = 120"))
                self.db_conn.save_corrections_updated(
                    session,
                    pick_source_ids,
                    ensemble_outputs,
                    summary_stats,
                    calibration_results,
                    start_date,
                    end_date,
                    on_event=logger.debug,
                )
        t1 = time.time()
        logger.debug(f"Total time saving corrections: {t1-t0:0.2f} s")

    def get_summary_stats(self, predictions):
        results = self.trim_dists(predictions)

        stds = np.std(predictions, axis=1)
        means = np.mean(predictions, axis=1)
        medians = np.median(predictions, axis=1)

        results["std"] = stds
        results["mean"] = means
        results["median"] = medians

        return results

    # Speedy version
    @staticmethod
    def trim_dists(predictions):
        q1, q3 = np.percentile(predictions, [25, 75], axis=1)
        iqr = q3 - q1
        if1 = q1 - 1.5 * iqr
        if3 = q3 + 1.5 * iqr
        # Create mask for values outside the inner fence
        bool_arr = np.logical_and(
            predictions > if1[:, None], predictions < if3[:, None]
        )
        # Mask the values
        mx = np.ma.masked_array(predictions, ~bool_arr)
        # Compute trimmed STDs
        trimmed_stds = mx.std(axis=1).data
        # Compute trimmed means and residuals
        trimmed_means = mx.mean(axis=1).data
        # Compute trimmed medians
        trimmed_medians = np.ma.median(mx, axis=1).data

        results = {
            "trim_std": trimmed_stds,
            "trim_mean": trimmed_means,
            "trim_median": trimmed_medians,
            "if_low": if1,
            "if_high": if3,
        }

        return results

    def process_3c_S(self, wfs, padding, desired_sampling_rate=100, normalize=True):
        """Process 3C S data using pyuussmlmodels.Pickers.CNNThreeComponentS.Preprocessing
        Args:
            wfs (np.array): Waveform to process (S, 3)
            desired_sampling_rate (int, optional): Desired sampling rate of the waveform.
            Defaults to 100.

        Returns:
            np.array: Processed waveform (S, 3)
        """
        processor = pyuussmlmodels.Pickers.CNNThreeComponentS.Preprocessing()
        east = wfs[:, 0]
        north = wfs[:, 1]
        vert = wfs[:, 2]
        proc_z, proc_n, proc_e = processor.process(
            vert, north, east, sampling_rate=desired_sampling_rate
        )
        # TODO: I'm pretty sure this won't work if the desired_sampling_rate != current sampling rate
        processed = np.zeros_like(wfs)
        processed[:, 0] = proc_e
        processed[:, 1] = proc_n
        processed[:, 2] = proc_z

        if padding > 0:
            processed = processed[padding:-padding, :]

        if normalize:
            processed = self.normalize_example(processed)

        return processed

    def process_1c_P(self, wf, padding, desired_sampling_rate=100, normalize=True):
        """Wrapper around pyuussmlmodels 1C UNet preprocessing function for use in format_continuous_for_unet.

        Args:
            wf (np.array): 1, 1C waveform (S, 1)
            desired_sampling_rate (int, optional): Desired sampling rate for wf. Defaults to 100.

        Returns:
            np.array: Processed wf (S, 1)
        """
        assert wf.shape[1] == 1, "Incorrect number of channels"
        processor = pyuussmlmodels.Pickers.CNNOneComponentP.Preprocessing()
        processed = processor.process(wf[:, 0], sampling_rate=desired_sampling_rate)[
            :, None
        ]

        if padding > 0:
            processed = processed[padding:-padding, :]

        if normalize:
            processed = self.normalize_example(processed)

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
                norm_vals_inv[nv_ind] = 1 / nv

        return waveform * norm_vals_inv

    ####

    # @staticmethod
    # def process_3c_S():
    #     print("Performing S 3C picker preprocessing test...")
    #     t, vertical, north, east = read_time_series_3c('data/pickers/cnnThreeComponentS/uu.gzu.eh.zne.01.txt')
    #     t, vertical_ref, north_ref, east_ref = read_time_series_3c('data/pickers/cnnThreeComponentS/uu.gzu.eh.zne.01.proc.txt')
    #     assert len(vertical) == 600
    #     assert len(vertical_ref) == 600
    #     sampling_rate = round(1./(t[1] - t[0]))
    #     assert sampling_rate == 100, 'sampling rate should be 100Hz'
    #     preprocessor = pyuussmlmodels.Pickers.CNNThreeComponentS.Preprocessing()
    #     assert preprocessor.target_sampling_rate == 100, 'target sampling rate should be 100 Hz'
    #     assert abs(preprocessor.target_sampling_rate - 1./preprocessor.target_sampling_period) < 1.e-8
    #     v_proc, n_proc, e_proc \
    #         = preprocessor.process(vertical, north, east, sampling_rate = sampling_rate)
    #     assert max(abs(v_proc - vertical_ref)) < 1.e-2
    #     assert max(abs(n_proc - north_ref)) < 1.e-2
    #     assert max(abs(e_proc - east_ref)) < 1.e-2
