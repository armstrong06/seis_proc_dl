import torch
import time
import os
import numpy as np
import sys
sys.path.append("/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis-proc-dl/utils")
from model_helpers import get_n_params, compute_outer_fence_mean_standard_deviation
from torch.autograd import Variable

class PickerTrainer():
    def __init__(self, network, optimizer, model_path, device):
        self.network = network
        self.optimizer = optimizer
        self.model_path = model_path
        self.device = device

    def train(self, train_loader, val_loader, n_epochs):
        self.network.train()
        print("Number of trainable parameters:", get_n_params(self.network))
        # Want to avoid L2 particularly when we go to the archives which have
        # some, err, interesting examples
        loss = torch.nn.MSELoss()  # Want to avoid L2 particularly when we go to the past
        # When |r| > delta switch from L2 to L1 norm.  Typically, autopicks will
        # be within 0.1 seconds of analyst picks so this seems generous but also
        # will also downweight any really discordant examples
        # loss = torch.nn.HuberLoss(delta = 0.25)
        n_batches = len(train_loader)
        training_start_time = time.time()
        training_rms_baseline = None
        validation_rms_baseline = None

        print(self.network.min_lag, self.network.max_lag)

        print_every = n_batches // 10
        if (not os.path.exists(self.model_path)):
            os.makedirs(self.model_path)

        for epoch in range(n_epochs):
            print("Beginning epoch {}...".format(epoch + 1))
            running_accuracy = 0
            running_loss = 0
            running_sample_count = 0
            total_training_loss = 0
            total_validation_loss = 0
            start_time = time.time()

            n_total_pred = 0
            y_true_all = np.zeros(len(train_loader.dataset), dtype='float')
            y_est_all = np.zeros(len(train_loader.dataset), dtype='float')
            for i, data in enumerate(train_loader, 0):
                # Get inputs/outputs and wrap in variable object
                inputs, y_true = data
                inputs, y_true = Variable(
                    inputs.to(self.device)), Variable(
                    y_true.to(self.device))

                # Set gradients for all parameters to zero
                # print(inputs.shape)
                if (inputs.shape[0] < 2):
                    print("Skipping edge case")
                    continue
                self.optimizer.zero_grad()

                # Forward pass
                y_est = self.network(inputs)

                # Backward pass
                loss_value = loss(y_est, y_true)
                loss_value.backward()
                loss_scalar_value = loss_value.data.cpu().numpy()

                # Update parameters
                self.optimizer.step()

                # Print statistics
                with torch.no_grad():
                    running_loss += loss_scalar_value
                    total_training_loss += loss_scalar_value

                    running_accuracy += loss_scalar_value
                    running_sample_count += torch.numel(y_true)
                    for i_local_pred in range(len(y_true)):
                        y_true_all[n_total_pred] = y_true[i_local_pred].cpu().numpy()
                        y_est_all[n_total_pred] = y_est[i_local_pred].cpu().numpy()
                        # print(y_true_all[n_total_pred], y_est_all[n_total_pred])
                        n_total_pred = n_total_pred + 1

                # Print every n'th batch of an epoch
                if ((i + 1) % (print_every + 1) == 0):
                    print("Epoch {}, {:d}% \t train_loss: {:.4e} "
                          "train_error: {:4.2f} took: {:.4f}s".format(
                        epoch + 1, int(100 * (i + 1) / n_batches),
                        running_loss / print_every,
                        100 * running_accuracy / running_sample_count,
                        time.time() - start_time))
                    running_loss = 0
                    start_time = time.time()
                # end print epoch
            # Loop on no gradient
            # Resize
            y_true_all = y_true_all[0:n_total_pred]
            y_est_all = y_est_all[0:n_total_pred]
            residuals = y_true_all - y_est_all
            training_mean = np.mean(residuals)
            training_std = np.std(residuals)
            training_mean_of, training_std_of = compute_outer_fence_mean_standard_deviation(residuals)
            training_rms = np.sqrt(np.sum(residuals ** 2) / len(y_true_all))
            random_lags = np.random.random_integers(self.network.min_lag, self.network.max_lag, size=n_total_pred)
            if (training_rms_baseline is None):
                residuals = y_true_all - random_lags
                training_mean_baseline = np.mean(residuals)
                training_std_baseline = np.std(residuals)
                training_rms_baseline = np.sqrt(np.sum(residuals ** 2) / len(y_true_all))
            print(
                "Training for epoch (Mean,Std,Outer Fence Mean, Outer Fence Std,RMS,Loss): (%f,%f,%f,%f,%f,%f) (Baseline Mean,Std,RMS~ %f,%f,%f)" % (
                    training_mean, training_std, training_mean_of, training_std_of,
                    training_rms, total_training_loss,
                    training_mean_baseline, training_std_baseline, training_rms_baseline))

            # Validation
            running_sample_count = 0
            running_val_accuracy = 0
            n_total_pred = 0
            y_true_all = np.zeros(len(val_loader.dataset), dtype='float')
            y_est_all = np.zeros(len(val_loader.dataset), dtype='float')
            with torch.no_grad():
                for inputs, y_true in val_loader:

                    # Wrap tensors in Variables
                    inputs, y_true = Variable(
                        inputs.to(self.device)), Variable(
                        y_true.to(self.device))

                    # Forward pass only
                    y_est = self.network(inputs)
                    val_loss = loss(y_est, y_true)
                    total_validation_loss += val_loss.item()

                    for i_local_pred in range(len(y_true)):
                        y_true_all[n_total_pred] = y_true[i_local_pred].cpu().numpy()
                        y_est_all[n_total_pred] = y_est[i_local_pred].cpu().numpy()
                        n_total_pred = n_total_pred + 1
            # Loop on data in training
            y_true_all = y_true_all[0:n_total_pred]
            y_est_all = y_est_all[0:n_total_pred]
            residuals = y_true_all - y_est_all
            validation_mean = np.mean(residuals)
            validation_std = np.std(residuals)
            validation_mean_of, validation_std_of = compute_outer_fence_mean_standard_deviation(residuals)
            validation_rms = np.sqrt(np.sum(residuals ** 2) / len(y_true_all))
            random_lags = np.random.random_integers(self.network.min_lag, self.network.max_lag, size=n_total_pred)
            if (validation_rms_baseline is None):
                residuals = y_true_all - random_lags
                validation_mean_baseline = np.mean(residuals)
                validation_std_baseline = np.std(residuals)
                validation_rms_baseline = np.sqrt(np.sum(residuals ** 2) / len(y_true_all))
            print(
                "Validation (Mean,Std,Outer Fence Mean, Outer Fence Std,RMS,Loss): (%f,%f,%f,%f,%f,%f) (Baseline Mean,Std,RMS ~ %f,%f,%f)" % (
                    validation_mean, validation_std, validation_mean_of, validation_std_of,
                    validation_rms, total_validation_loss,
                    validation_mean_baseline, validation_std_baseline, validation_rms_baseline))

            model_file_name = os.path.join(self.model_path,
                                           'models_%03d.pt' % (epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_mean_of': training_mean_of,
                'validation_mean_of': validation_mean_of,
                'training_std_of': training_std_of,
                'validation_std_of': validation_std_of,
                'training_mean': training_mean,
                'validation_mean': validation_mean,
                'training_std': training_std,
                'validation_std': validation_std,
                'validation_loss': total_validation_loss,
                'training_loss': total_training_loss,
                'validation_rms': validation_rms,
                'training_rms': training_rms,
            }, model_file_name)
            self.network.write_weights_to_hdf5(os.path.join(self.model_path,
                                                            'models_%03d.h5' % (epoch + 1)))