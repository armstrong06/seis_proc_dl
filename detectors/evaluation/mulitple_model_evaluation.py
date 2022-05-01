from operator import mod
import os 
import torch
import time
import numpy as np
import pandas as pd
import h5py
import glob

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
            raise Exception
        print(f"Loading {model_in}")
        check_point = torch.load(model_in)
        training_loss = {'epoch': check_point['epoch'], 'training_loss': check_point['loss']}
        #print(training_loss)
        self.model.load_state_dict(check_point['model_state_dict'])
        self.model.eval()

        return training_loss

    @staticmethod
    def load_dataset(h5f, mew=False):
        with h5py.File(h5f, "r") as f:
            X_test = f['X'][:]
            T_test = f['Pick_index'][:]
            if mew:
                Y_test = f["Y"][:]
                T_test2 = f["Pick_index2"][:]

        if mew:
            return X_test, Y_test, T_test, T_test2

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
            model_to_test = glob.glob(os.path.join(self.model_states_path, f"*{epoch:03}.pt"))
            assert len(model_to_test)==1, "Wrong number of model paths found"
            training_loss = self.load_model_state(model_to_test[0])
            self.evaluator.set_model(self.model)
            post_probs, pick_info = self.evaluator.apply_model(X_test, pick_method=pick_method)
            Y_proba = pick_info[1]
            T_est_index = pick_info[0]

            for i in range(len(T_test)):
                # Removing this becuase I want to be able to calculate confusion matrices after the fact
                # if (T_test[i] < 0):
                #     break
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


    def evaluate_over_models_mew(self, data_path, tols, save_proba=False, save_js=False):
        start = time.time()

        metrics_p1 = []
        metrics_p2 = []
        metrics_comb = []
        metrics_js = []
        residual_info = []

        X_test, Y_test, T_test, T_test2 = self.load_dataset(data_path, mew=True)

        if save_proba:
            probafile = h5py.File(f'{self.output_dir}/proba.h5', "w")
            probafile.create_group("ModelOutputs")

        if save_js:
            js_file = h5py.File('{self.output_dir}/jaccard_similarity.h5', "w")

        for epoch in self.epochs:
            print("Testing:", epoch)
            model_to_test = glob.glob(os.path.join(self.model_states_path, f"*{epoch:03}.pt"))
            assert len(model_to_test)==1, "Wrong number of model paths found"
            training_loss = self.load_model_state(model_to_test[0])
            self.evaluator.set_model(self.model)

            post_probs, pick_info = self.evaluator.apply_model(X_test, pick_method="multiple")
            
            T_est_index = pick_info[0]
            Y_proba = pick_info[1]
            widths = pick_info[2]

            for i in range(len(T_test)):
                if T_test[i] < 0:
                    break

                true_lag2 = None
                if i < len(T_test2):
                    true_lag2 = T_test2[i]

                est_picks = T_est_index[i][T_est_index[i] > 0]
                est_proba = Y_proba[i][Y_proba[i] > 0]

                if len(est_picks) == 0:
                    residual_info.append( {'epoch': epoch,
                    'waveform_index':i,
                    'pick_index': None,
                    'probability': None,
                    'true_lag1': T_test[i],
                    'true_lag2': true_lag2,
                    'residual1': None,
                    'residual2': None,
                    'is_match': False,
                    'width': None})
                    #'snr': snrs[i] 
                else:
                    # Didn't put the jaccard similairty in here because it would just be the jaccard similarity using the minimum tolerace value
                    all_resids1 = np.full(len(est_picks), np.nan)
                    all_resids2 = np.full(len(est_picks), np.nan)
                    for pind in range(len(est_picks)):
                        all_resids1[pind] = (T_test[i] - est_picks[pind])
                        if i < len(T_test2):
                            all_resids2[pind] = (T_test2[i]-est_picks[pind])


                    match1 = None
                    match2 = None
                    # TODO: Do I need to handle it when these are the same (usually when only one pick)??
                    if len(all_resids1) > 0 and ~np.isnan(np.unique(all_resids1))[0]:
                        match1 = np.where(abs(all_resids1) == np.min(abs(all_resids1)))[0][0]
                    if len(all_resids2) > 0 and ~np.isnan(np.unique(all_resids2))[0]:
                        match2 = np.where(abs(all_resids2) == np.min(abs(all_resids2)))[0][0]
                    
                    for pind in range(len(est_picks)):
                        # TODO: Is match doesn't really mean anything - no minimum proximity required just that it is the closest pick 
                        is_match = False
                        if pind == match1 or pind == match2:
                            is_match = True

                        residual_info.append( {'epoch': epoch,
                                        'waveform_index':i,
                                        'pick_index': est_picks[pind],
                                        'probability': est_proba[pind],
                                        'true_lag1': T_test[i],
                                        'true_lag2': true_lag2,
                                        'residual1': all_resids1[pind],
                                        'residual2': all_resids2[pind],
                                        'is_match': is_match,
                                        'width': widths[i][pind]})
                                        #'snr': snrs[i] 

            metric_p1, metric_p2, metric_comb, metric_js, js_arrays = self.evaluator.tabulate_metrics_mew(T_test, T_test2, Y_proba, T_est_index, epoch, post_probs,
                                                                                Y_test, tols=tols)

            # TODO: check this - I'm not sure what update is doing
            for m in metric_p1:
                m.update(training_loss)
                metrics_p1.append(m)

            for m in metric_p2:
                m.update(training_loss)
                metrics_p2.append(m)

            for m in metric_comb:
                m.update(training_loss)
                metrics_comb.append(m)

            for js in metric_js:
                metrics_js.append(js)

            if save_proba:
                probafile.create_dataset("%s.Y_est"%epoch, data=post_probs)
                probafile.create_dataset("%s.Y_max_proba"%epoch, data=Y_proba)
                probafile.create_dataset("%s.T_est_index"%epoch, data=T_est_index)

            if save_js:
                group = js_file.create_group("epoch.%02d"%epoch)
                for t in range(len(tols)):
                    group.create_dataset("tol.%0.2f.pick"%tols[t], data=js_arrays[t]["js_pick"])
                    group.create_dataset("tol.%0.2f.proba"%tols[t], data=js_arrays[t]["js_proba"])


        if save_proba:
            probafile.close()

        if save_js:
            js_file.close()

        # Loop
        end = time.time()
        print("Total time:", end-start)
 
        df = pd.DataFrame(metrics_p1)
        df.to_csv(f'{self.output_dir}/metrics_p1.csv', index=False)

        df = pd.DataFrame(metrics_p2)
        df.to_csv(f'{self.output_dir}/metrics_p2.csv', index=False)

        df = pd.DataFrame(metrics_comb)
        df.to_csv(f'{self.output_dir}/metrics_combined.csv', index=False)

        df = pd.DataFrame(metrics_js)
        df.to_csv(f'{self.output_dir}/metrics_js.csv', index=False)

        df_resid = pd.DataFrame(residual_info)
        df_resid.to_csv(f'{self.output_dir}/residuals.csv', index=False)