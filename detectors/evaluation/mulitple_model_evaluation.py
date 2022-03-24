from operator import mod
import os 
import torch
import time
import numpy as np
import pandas as pd
import h5py

class MultiModelEval():
    def __init__(self, model, model_states_path, evaluator, output_dir):
        #Initialized model 
        self.model = model
        self.model_states = model_states_path
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

    def evaluate_over_models(self, data_path, tols):

        # Not going to save all posterior probs here because that seems uneccessary, 
        # just choose a model and then evaluate using evalutor and save

        start = time.time()
        resids = []
        metrics = []
        X_test, T_test = self.load_dataset(data_path)

        T_est_index = np.zeros(T_test.shape[0]) - 1

        for model_to_test in self.model_states:
            model_tag = model_to_test.split("/")[-1]
            training_loss = self.load_model_state(model_to_test)
            self.evaluator.set_model(self.model)
            Y_proba, T_est_index, Y_est_all = self.evaluator.apply_model(X_test)

            for i in range(len(T_test)):
                if (T_test[i] < 0):
                    break
                resids.append({'model': model_tag,
                                'true_lag': T_test[i],
                                'residual': T_test[i] - T_est_index[i],
                                'probability': Y_proba[i]})

                metric = self.evaluator.tabulate_metrics(T_test, Y_proba, T_est_index, model_tag=model_tag, tols=tols)
                for m in metric:
                    m.update(training_loss)
                    metrics.append(m)

        end = time.time()
        print("Total time:", end-start)

        df = pd.DataFrame(metrics) 
        df.to_csv(f'{self.output_dir}/metrics.csv', index=False)

        df_resid = pd.DataFrame(resids)
        df_resid.to_csv(f'{self.output_dir}/residuals.csv', index=False)
