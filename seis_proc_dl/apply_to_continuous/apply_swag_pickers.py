import torch
import numpy as np
import pandas as pd
import h5py
from joblib import load
from scipy.stats import norm 
import os
import sys
sys.path.append("/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/atucus/swag_modified")
from swag import models as swag_models
from swag import utils as swag_utils
from swag.posteriors import SWAG
import time

class SwagPicker():
    def __init__(self, model_name, checkpoint_file, seed, cov_mat=True, K=20, device="cuda:0"):
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
            **model_cfg.kwargs)
        
        #self.model.cuda()
        self.device = device
        self.model.to(self.device)

        print("Loading model %s" % checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint["state_dict"])
        #self.batchsize = batchsize

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
                #input = input.cuda(non_blocking=True)
                input = input.to(self.device, non_blocking=True)
                torch.manual_seed(i)
                output = self.model(input)
                with torch.no_grad():
                    predictions[k : k + input.size()[0], i:i+1] = output.cpu().numpy()
                k += input.size()[0]

        return predictions
    
class Dset(torch.utils.data.Dataset):
    def __init__(self, data):
        print(data.shape)
        self.data = torch.from_numpy(data.transpose((0, 2, 1))).float()
        
    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)
        
class MultiSWAGPicker():
    def __init__(self, is_p_picker, device="cuda:0") -> None:
        if is_p_picker:
            self.phase = "P"
        else:
            self.phase = "S"

        self.device = device

    def load_swag_ensemble(self, model_dir, model1, model2, model3, seeds, cov_mat, k):
        modelname = f"{self.phase}Picker"
        swag1 = SwagPicker(modelname, f'{model_dir}/{model1}', seeds[0],
                            cov_mat=cov_mat, K=k, device=self.device)
        swag2 = SwagPicker(modelname, f'{model_dir}/{model2}', seeds[1],
                    cov_mat=cov_mat, K=k, device=self.device)
        swag3 = SwagPicker(modelname, f'{model_dir}/{model3}', seeds[2],
                    cov_mat=cov_mat, K=k, device=self.device)
        
        return [swag1, swag2, swag3]

    def load_data(self, train_file, train_path, 
                  new_data_file, new_data_path, batchsize=256, 
                  num_workers=2, shuffle_train=False):
        train_loader = self.torch_loader(train_file, 
                                         train_path, 
                                         batchsize,
                                         num_workers,
                                         shuffle_train)
        cont_loader = self.torch_loader(new_data_file, 
                                         new_data_path, 
                                         batchsize,
                                         num_workers,
                                         shuffle=False)
        return train_loader, cont_loader

    @staticmethod
    def apply_picker(models, cont_loader, train_loader, N):
        n_examples = cont_loader.dataset.data.shape[0]
        ensemble_outputs = np.zeros((n_examples, N*len(models)))
        for i, model in enumerate(models):
            st = time.time()
            ensemble_outputs[:, i*N:i*N+N] = model.apply_model(cont_loader, N, train_loader)
            et = time.time()
            print(f"Average time per batch for model {i}: {(et-st)/len(cont_loader):3.2f} s")
            print(f"Average time per sample for model {i}: {(et-st)/N:3.2f} s")

        return ensemble_outputs

    @staticmethod
    def get_calibrated_pick_bounds(iso_reg_inv_file, lb, ub):
        # Transform the lower and upper bounds to be calibrated
        iso_reg_inv = load(iso_reg_inv_file)
        lb_transform = iso_reg_inv.transform([lb])[0]
        ub_transform = iso_reg_inv.transform([ub])[0]
        return lb_transform, ub_transform
    
    @staticmethod
    def torch_loader(filename,
                    path,
                    batch_size,
                    num_workers,
                    shuffle=False,
                    n_examples=-1):
        
        with h5py.File(f'{path}/{filename}', 'r') as f:
            X = f['X'][:]

        if len(X.shape) < 3:
            X = np.expand_dims(X, 2)

        if n_examples > 0:
            X = X[:n_examples, :, :]

        dset = Dset(X)
        loader = torch.utils.data.DataLoader(dset, 
                                             batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers,
                                            pin_memory=True)
        return loader
    
    @staticmethod
    def calibrate_swag_predictions(y_pred, pred_std, lb_transform, ub_transform):
        df_data = {"y_pred":y_pred, "std":pred_std}
        df = pd.DataFrame(data=df_data)
        y_lb = df.apply(lambda x: norm.ppf(lb_transform, x["y_pred"], x["std"]), axis=1).values
        y_ub = df.apply(lambda x: norm.ppf(ub_transform, x["y_pred"], x["std"]), axis=1).values

        summary = {"arrivalTimeShift":y_pred, "arrivalTimeShiftSTD": pred_std, "arrivalTimeShiftLowerBound": y_lb, "arrivalTimeShiftUpperBound": y_ub}

        return summary
    
    def format_and_save(self, meta_csv_file, pred_summary, all_predictions, outfile_pref, region, n_meta_rows=-1):
        columns = ["eventIdentifier","network","station","channel","locationCode","phase"]
        if self.phase == "S":
            columns = ["eventIdentifier","network","station","verticalChannel","locationCode","phase"]
        meta_df = pd.read_csv(meta_csv_file)[columns]
        if n_meta_rows > 0:
            meta_df = meta_df.iloc[0:n_meta_rows]
        summary_df = pd.DataFrame(pred_summary)
        meta_df = meta_df.join(summary_df)
        #meta_df.loc[:, 'correctedArrivalTime'] = meta_df['estimateArrivalTime'] + meta_df['arrivalTimeShift']
        csv_outfile = os.path.join(outfile_pref, f"corrections.{self.phase.lower()}Arrivals.{region}.csv")
        h5_outfile = os.path.join(outfile_pref, f"corrections.{self.phase.lower()}Arrivals.{region}.h5")
        print("Writing", csv_outfile, h5_outfile)

        meta_df.to_csv(csv_outfile, index=False, float_format='%0.6f')
        with h5py.File(h5_outfile, "w") as f:
            f.create_dataset("X", shape=all_predictions.shape, data=all_predictions)
    
    @staticmethod
    def trim_inner_fence(ensemble_predictions):
        q1, q3  = np.percentile(ensemble_predictions, [25, 75], axis=1)
        iqr = q3 - q1
        if1 = q1 - 1.5*iqr
        if3 = q3 + 1.5*iqr
        ensemble_stds = np.zeros(ensemble_predictions.shape[0])
        ensemble_medians = np.zeros(ensemble_predictions.shape[0])
        for i in range(len(ensemble_predictions)):
            trimmed = ensemble_predictions[i][np.where(np.logical_and(ensemble_predictions[i] > if1[i], 
                                                                      ensemble_predictions[i] < if3[i]))]
            ensemble_stds[i] = np.std(trimmed)
            ensemble_medians[i] = np.median(trimmed)

        return ensemble_medians, ensemble_stds