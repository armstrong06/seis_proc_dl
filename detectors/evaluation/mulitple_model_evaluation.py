from operator import mod
import os 
import torch
import time
import numpy as np
import pandas as pd
import h5py

class MultiModelEval():
    def __init__(self, model, model_states_path, epochs, evaluator, output_dir):
        #Initialized model 
        self.model = model
        self.model_states_path = model_states_path
        self.epochs = epochs
        self.evaluator = evaluator

        if (not os.path.exists(output_dir)):
            os.makedirs(output_dir)

        self.output_dir = output_dir

    def load_model_state(self, model_in):
        if (not os.path.exists(model_in)):
            print("Model", model_in, " does not exist")
            return None

        check_point = torch.load(model_in)
        training_loss = {'epoch': check_point['epoch'], 'training_loss': check_point['loss']}
        #print(training_loss)
        self.model.load_state_dict(check_point['model_state_dict'])
        self.model.eval()

        return training_loss

    @staticmethod
    def load_dataset(h5f):
        with h5py.File(h5f) as f:
            X_test = f['X'][:]
            T_test = f['Pick_index'][:]

        return X_test, T_test

    def evaluate_over_models(self, data_path, tols, pick_method, save_proba=False):

        # Not going to save all posterior probs here because that seems uneccessary, 
        # just choose a model and then evaluate using evalutor and save

        start = time.time()
        resids = []
        metrics = []
        X_test, T_test = self.load_dataset(data_path)

        if save_proba:
            probafile = h5py.File(f'{self.output_dir}/proba.h5', "w")
            probafile.create_group("ModelOutputs")

        for epoch in self.epochs:
            model_to_test = os.path.join(self.model_states_path, "*{epoch: 3d}.pt")
            training_loss = self.load_model_state(model_to_test)
            self.evaluator.set_model(self.model)
            post_probs, pick_info = self.evaluator.apply_model(X_test, pick_method=pick_method)
            Y_proba = pick_info[1]
            T_est_index = pick_info[0]

            for i in range(len(T_test)):
                if (T_test[i] < 0):
                    break
                resids.append({'model': epoch,
                                'true_lag': T_test[i],
                                'residual': T_test[i] - T_est_index[i],
                                'probability': Y_proba[i]})

                metric = self.evaluator.tabulate_metrics(T_test, Y_proba, T_est_index, epoch, tols=tols)
                for m in metric:
                    m.update(training_loss)
                    metrics.append(m)

            if save_proba:
                probafile.create_dataset("%s.Y_est"%epoch, data=post_probs)
                probafile.create_dataset("%s.Y_max_proba"%epoch, data=Y_proba)
                probafile.create_dataset("%s.T_est_index"%epoch, data=T_est_index)


        end = time.time()
        print("Total time:", end-start)

        if save_proba:
            probafile.close()
            
        df = pd.DataFrame(metrics) 
        df.to_csv(f'{self.output_dir}/metrics.csv', index=False)

        df_resid = pd.DataFrame(resids)
        df_resid.to_csv(f'{self.output_dir}/residuals.csv', index=False)
