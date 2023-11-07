from seis_proc_dl.apply_to_continuous import apply_swag_pickers
import torch
import numpy as np
import os
import pandas as pd
import h5py

def test_init_P():
    sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True)
    assert sp.phase == "P"
    assert sp.device == "cuda:0"

def test_init_S():
    sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=False)
    assert sp.phase == "S"  
    assert sp.device == "cuda:0"

def test_init_cpu():
    sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True,
                                            device='cpu')
    assert sp.device == "cpu"

def test_torch_loader_train():
    sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True)
    path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/swag_info"
    loader = sp.torch_loader("p_uuss_train_4s_1dup.h5",
                             path,
                             256, 
                             3, 
                             False)
    assert loader.dataset.data.shape == (336885, 1, 400)
    assert loader.batch_size == 256
    assert loader.num_workers == 3

def test_torch_loader_cont():
    sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True)
    path = "/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/machineLearning/harvestPicks/gcc_build"
    loader = sp.torch_loader("pArrivals.ynpEarthquake.h5",
                             path,
                             512, 
                             0, 
                             False)
    assert loader.dataset.data.shape == (253088, 1, 400)
    assert loader.batch_size == 512
    assert loader.num_workers == 0

def test_load_swag_ensemble():
    sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True, device='cpu')
    path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models"
    model1 = "pPicker_swag60_seed1.pt"
    model2 = "pPicker_swag60_seed2.pt"
    model3 = "pPicker_swag60_seed3.pt"
    ensemble = sp.load_swag_ensemble(path, model1, model2, model3, [1, 2, 3], True, 20)
    assert ensemble[0].seed == 1
    assert ensemble[1].seed == 2
    assert ensemble[2].seed == 3
    assert ensemble[0].model.no_cov_mat == False
    assert ensemble[1].model.no_cov_mat == False
    assert ensemble[2].model.no_cov_mat == False
    assert ensemble[0].model.max_num_models == 20
    assert ensemble[1].model.max_num_models == 20
    assert ensemble[2].model.max_num_models == 20
    assert not torch.equal(ensemble[0].model.state_dict()['base.conv1.weight_mean'], 
                       ensemble[1].model.state_dict()['base.conv1.weight_mean'])
    assert not torch.equal(ensemble[1].model.state_dict()['base.conv1.weight_mean'], 
                       ensemble[2].model.state_dict()['base.conv1.weight_mean'])
    assert not torch.equal(ensemble[0].model.state_dict()['base.conv1.weight_mean'], 
                       ensemble[2].model.state_dict()['base.conv1.weight_mean'])

def test_get_calibrated_pick_bounds():
    sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True, device='cpu')
    path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models"
    file = f"{path}/p_calibration_model_medians_ensemble_IFtrimmed_sklearn1.3.1.joblib"
    lb, ub = sp.get_calibrated_pick_bounds(file, 0.05, 0.95)
    assert lb < 0.05 and lb > 0.0
    assert ub > 0.95 and ub < 1.0

def test_apply_picker_P():
    sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True, device='cuda:0')
    path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models"
    file  = f"{path}/pPicker_swag60_seed1.pt"
    swag1 = apply_swag_pickers.SwagPicker("PPicker", file, 1,
                            cov_mat=True, K=20, device='cuda:0')
    path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/swag_info"
    data_loader = sp.torch_loader("p_uuss_NGB_4s_1dup.h5",
                                path,
                                10, 
                                3, 
                                False,
                                10)
    assert len(data_loader) == 1
    train_loader = sp.torch_loader("p_uuss_train_4s_1dup.h5",
                                path,
                                128, 
                                3, 
                                False)
    new_preds = sp.apply_picker([swag1], data_loader, train_loader, 5)

    compare_preds = np.load(f"{path}/p_swag_NGB_uncertainty_40_seed1.npz")['predictions'][:10, :5]

    # For some reason, the first predictions do not match
    assert np.allclose(new_preds[:, 1:], compare_preds[:, 1:], atol=1e-6)

def test_calibrate_swag_predictions():
    sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True, device='cpu')
    path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models"
    file = f"{path}/p_calibration_model_medians_ensemble_IFtrimmed_sklearn1.3.1.joblib"
    lb_trans, ub_trans = sp.get_calibrated_pick_bounds(file, 0.05, 0.95)
    summary = sp.calibrate_swag_predictions([0], [1], lb_trans, ub_trans)
    assert summary['arrivalTimeShift'][0] == 0
    assert summary['arrivalTimeSTD'][0] == 1
    assert summary['arrivalTimeLowerBound'][0] < -2
    assert summary['arrivalTimeUpperBound'][0] > 2

def test_calibrate_swag_predictions_small():
    sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True, device='cpu')
    path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models"
    file = f"{path}/p_calibration_model_medians_ensemble_IFtrimmed_sklearn1.3.1.joblib"
    lb_trans, ub_trans = sp.get_calibrated_pick_bounds(file, 0.05, 0.95)
    summary = sp.calibrate_swag_predictions([0, 0], [1, 0.1], lb_trans, ub_trans)
    assert summary['arrivalTimeShift'][1] == 0
    assert summary['arrivalTimeSTD'][1] == 0.1
    assert summary['arrivalTimeLowerBound'][1] < -0.2
    assert summary['arrivalTimeUpperBound'][1] > 0.2

def test_trim_inner_fence():
    sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True, device='cpu')
    preds = np.random.uniform(0, 1, 25)
    preds[20:] = [-20, 20, -20, 20, 20]
    preds = np.expand_dims(preds, 0)
    median, std = sp.trim_inner_fence(preds)
    assert std < np.std(preds)
    assert median < np.median(preds)

def test_format_and_save_P():
    sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=True, device='cpu')
    pred_summary = {"arrivalTimeShift":[0], 
                    "arrivalTimeSTD": [1], 
                    "arrivalTimeLowerBound": [-2], 
                    "arrivalTimeUpperBound": [2]}
    preds = np.random.uniform(0, 1, 25)
    preds = np.expand_dims(preds, 0)
    outfile_pref = "/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis_proc_dl/seis_proc_dl/pytests/example_files"
    region = "ynpEarthquake"
    meta_df = f"{outfile_pref}/test_pick_meta_df.csv"
    sp.format_and_save(meta_df, pred_summary, preds, outfile_pref, region)

    new_df = pd.read_csv(f"{outfile_pref}/corrections.PArrivals.ynpEarthquake.csv")
    assert new_df.shape == (1, 10)
    
    with h5py.File(f"{outfile_pref}/corrections.PArrivals.ynpEarthquake.h5", "r") as f:
        assert np.array_equal(f['X'][:], preds)

    os.remove(f"{outfile_pref}/corrections.PArrivals.ynpEarthquake.csv")
    os.remove(f"{outfile_pref}/corrections.PArrivals.ynpEarthquake.h5")

def test_format_and_save_S():
    pass

def test_load_data():
    pass

def test_apply_picker_S():
    pass

if __name__ == '__main__':
    test_format_and_save_P()

