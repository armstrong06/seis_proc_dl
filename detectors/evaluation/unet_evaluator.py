import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from utils.model_helpers import clamp_presigmoid_values

class UNetEvaluator():

    def __init__(self, model, batch_size, device, center_window=None, apply_sigmoid=True, minimum_presigmoid_value=None):
        self.min_presigmoid_value = minimum_presigmoid_value
        self.model = model
        self.batch_size = batch_size
        self.apply_sigmoid = apply_sigmoid
        self.device = device
        self.center_window = center_window

    def get_n_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def apply_model(self, X_test):
        n_samples = X_test.shape[1]
        n_examples =  X_test.shape[0]

        # Initialize
        pred_pick_index = np.zeros(n_examples, dtype='i')
        pred_pick_prob = np.zeros(n_examples)
        all_posterior_probs = np.zeros([n_examples, n_samples])

        for i1 in range(0, n_examples, self.batch_size):
            i2 = min(n_examples, i1 + self.batch_size)

            # Gather batch
            X_temp = np.copy(X_test[i1:i2, :, :]) 
            X = torch.from_numpy(X_temp.transpose((0, 2, 1))).float().to(self.device)
            
            # Get predictions
            Y_est = self.model.forward(X)
            if (self.apply_sigmoid):
                if self.min_presigmoid_value is not None:
                    Y_est = clamp_presigmoid_values(Y_est, self.min_presigmoid_value)
                Y_est = torch.sigmoid(Y_est)

            Y_est = Y_est.squeeze()
            all_posterior_probs[i1:i2, :] = Y_est.to('cpu').detach().numpy()
                
            # Pick indices and probabilities
            values, indices = self.get_single_picks(Y_est)
            for k in range(i1,i2):
                pred_pick_prob[k] = values[k-i1]
                pred_pick_index[k] = indices[k-i1]

        return pred_pick_prob, pred_pick_index, all_posterior_probs

    def get_single_picks(self, Y_est):
        # Pick indices and probabilities
        if (self.center_window is None):
            values, indices = Y_est.max(dim=1)
            values = values.to('cpu').flatten()
            indices = indices.to('cpu').flatten()
        else:
            j1 = int(n_samples/2 - self.center_window)
            j2 = int(n_samples/2 + self.center_window)
            Y_sub = Y_est[:,j1:j2]
            values, indices = Y_sub.max(dim=1) 
            values = values.to('cpu').flatten()
            indices = indices.to('cpu').flatten() + j1

        return values, indices


    def tabulate_metrics(self, true_pick_index, est_pick_proba, est_pick_index, epoch,
                        tols = [0.1, 0.25, 0.5, 0.75, 0.9]):
        results = []

        # Only look at signal, not noise
        Y_obs = (true_pick_index >= 0)*1
        n_picks = np.sum(Y_obs)

        # Iterate over different tolerance thresholds for making a pick
        for tol in tols:
            Y_est = (est_pick_proba > tol)*1
            index_resid = np.zeros(len(true_pick_index), dtype='int')

            # Find where there are matching picks (any pick in the window) and get residuals
            j = 0
            for i in range(len(true_pick_index)):
                if (Y_obs[i] == 1 and Y_est[i] == 1):
                    index_resid[j] = true_pick_index[i] - est_pick_index[i] 
                    j = j + 1
            index_resid = np.copy(index_resid[0:j])
            if (len(index_resid) > 0):
                trimmed_mean, trimmed_std = self.compute_outer_fence_mean_standard_deviation(index_resid)
            else:
                trimmed_mean = 0
                trimmed_std = 0
            
            # I had to add label into this or it breaks when 100% accuracte. If there are more than 2 classes, will need to edit this. 
            tn, fp, fn, tp = confusion_matrix(Y_obs, Y_est, labels=[0, 1]).ravel()
            acc  = (tn + tp)/(tp + tn + fp + fn)
            prec   = tp/(tp + fp)
            recall = tp/(tp + fn)
            dic = {"epoch": epoch,
                "n_picks": n_picks,
                "n_picked": len(index_resid),
                "tolerance": tol,
                "accuracy": acc,
                "precision": prec,
                "residual_mean": np.mean(index_resid),
                "residual_std": np.std(index_resid),
                "trimmed_residual_mean": trimmed_mean,
                "trimmed_residual_std": trimmed_std,
                "recall": recall}
            results.append(dic)
        return results

    @staticmethod
    def compute_outer_fence_mean_standard_deviation(residuals):
        
        q1, q3 = np.percentile(residuals, [25, 75])
        iqr = q3 - q1 
        of1 = q1 - 1.5*iqr
        of3 = q3 + 1.5*iqr
        trimmed_residuals = residuals[(residuals > of1) & (residuals < of3)]
        #print(len(trimmed_residuals), len(residuals), of1, of3)
        xmean = np.mean(trimmed_residuals)
        xstd = np.std(trimmed_residuals) 
        return xmean, xstd 