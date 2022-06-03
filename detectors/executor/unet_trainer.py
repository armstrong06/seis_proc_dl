import time
import os
import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import classification_report

from utils.model_helpers import clamp_presigmoid_values

class UNetTrainer():
    def __init__(self, network, optimizer, loss_fn, model_path, device, phase_type="P", detection_thresh=0.5, minimum_presigmoid_value=None):
        self.network = network
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.model_path = model_path
        self.phase_type = phase_type
        self.detection_thresh = detection_thresh
        self.device = device
        self.min_presigmoid_value = minimum_presigmoid_value

        if (not os.path.exists(self.model_path)):
            os.makedirs(self.model_path)

    def train_step(self, inputs, y_true):
        self.network.train()

        # Forward pass        
        presigmoid_output = self.network(inputs)
        if self.min_presigmoid_value is not None:
            presigmoid_output = clamp_presigmoid_values(presigmoid_output, self.min_presigmoid_value)

        # Set gradients for all parameters to zero
        self.optimizer.zero_grad()

        # Backward pass 
        # BCEWithLogitsLoss combines the sigmoid with the loss function
        loss_value = self.loss_fn(presigmoid_output, y_true)
        loss_value.backward()

        # Update parameters
        self.optimizer.step()

        return loss_value, presigmoid_output


    def train(self, train_loader, n_epochs, val_loader=None, verbose=True, save_all_models=True):
        """Train the model"""
        # loss = torch.nn.BCEWithLogitsLoss()
        n_batches = len(train_loader)
        training_start_time = time.time()

        for epoch in range(n_epochs):
            running_loss = 0.0
            running_acc = 0
            print_every = n_batches // 10
            start_time = time.time()
            total_train_loss = 0
            running_sample_count = 0

            for i, batch in enumerate(train_loader, 0):
                
                inputs, y_true = batch
                # Get inputs/outputs and wrap in variable object
                inputs = Variable(inputs.to(self.device))
                y_true = Variable(y_true.to(self.device))

                loss_value, presigmoid_output = self.train_step(inputs, y_true)

                # update loss trackers
                running_loss += loss_value.item()
                total_train_loss += loss_value.item()

                # Calculate accuracy on batch & update running metrics
                running_acc_update, running_sample_count_update = self.check_batch_accuracy(presigmoid_output, y_true)
                running_acc += running_acc_update
                running_sample_count += running_sample_count_update

                # Print every 10th batch of an epoch
                if (i + 1) % (print_every + 1) == 0:
                    if verbose:
                        print("Epoch {}, {:d}% \t train_loss: {:.4e} "
                            "train_acc: {:4.2f}% took: {:.2f}s".format(
                            epoch + 1, int(100 * (i + 1) / n_batches),
                            running_loss / print_every,
                            100*running_acc / running_sample_count,
                            time.time() - start_time))
                    # Reset running loss and time
                    running_loss = 0.0
                    start_time = time.time()

            total_train_loss /= len(train_loader)
            print("Training loss:", total_train_loss)

            total_val_loss = None
            if val_loader is not None:
                # Not using these now, but could in the future if wanted to stop on validation loss
                total_val_acc, total_val_loss = self.compute_validation_accuracy(val_loader)

            if save_all_models or (epoch==n_epochs-1):
                self.save_model(epoch, total_train_loss, total_val_loss=total_val_loss)

        print("Training finished, took {:.2f}s".format(time.time()-training_start_time))

    def check_batch_accuracy(self, presigmoid_outputs, y_true):
        self.network.eval()
        with torch.no_grad():
            scores = torch.sigmoid(presigmoid_outputs)

            # Calculate categorical accuracy
            y_pred = torch.zeros(scores.data.size()).to(self.device)
            y_pred[scores >= self.detection_thresh] = 1

            running_acc_update = (y_pred == y_true).sum().item()
            running_sample_count_update = torch.numel(y_true)

            return running_acc_update, running_sample_count_update

    def save_model(self, epoch, total_train_loss, total_val_loss=None):
        filename = '%s/model_%s_%03d.pt'% (self.model_path, self.phase_type, epoch)
        print(f'writing {filename}' )

        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': total_train_loss, #total_val_loss,
            }

        if total_val_loss is not None:
            save_dict['validation_loss'] = total_val_loss

        torch.save(save_dict, filename)

    def check_validation_accuracy(self, val_loader):
            running_sample_count = 0
            running_val_acc = 0
            total_val_loss = 0
            total_val_acc = 0
            y_pred_all, y_true_all = [], []

            # Set model to evaluation mode
            self.network.eval()
            with torch.no_grad():
                for inputs, y_true in val_loader:
                    # Wrap tensors in Variables
                    inputs = Variable(inputs.to(self.device))
                    y_true = Variable(y_true.to(self.device))

                    # Forward pass only
                    val_outputs = self.network(inputs)
                    if self.min_presigmoid_value is not None:
                        val_outputs = clamp_presigmoid_values(val_outputs)
                    val_outputs = torch.sigmoid(val_outputs)

                    val_loss = self.loss_fn(val_outputs, y_true)
                    total_val_loss += val_loss.item()

                    # Calculate categorical accuracy
                    y_pred = torch.zeros(val_outputs.data.size()).to(self.device)
                    y_pred[val_outputs >= self.detection_thresh] = 1

                    # Compute the pick times
                    y_pred = y_pred.cpu().numpy()
                    y_true = y_true.cpu().numpy()

                    y_pred_all.append(y_pred.flatten())
                    y_true_all.append(y_true.flatten())

                    #y_pred_all.append(y_pred.cpu().numpy().flatten())
                    #y_true_all.append(y_true.cpu().numpy().flatten())

                    running_val_acc += (y_pred == y_true).sum().item()
                    running_sample_count += torch.numel(y_true)

            y_pred_all = np.concatenate(y_pred_all)
            y_true_all = np.concatenate(y_true_all)

            total_val_loss /= len(val_loader)
            total_val_acc = running_val_acc / running_sample_count
            print(
                "Validation loss = {:.4e}   acc = {:4.2f}%".format(
                    total_val_loss,
                    100*total_val_acc))

            print(classification_report(y_true_all, y_pred_all))

            return total_val_acc, total_val_loss