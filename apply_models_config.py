"""Model config in json format for applying models to continous data """

CFG = {
    "model_params":{
        "batch_size":128,
        "center_window":250,
        "sliding_interval":500,
        "unet_window_length":1008,
        "pcnn_window_length":400,
        "scnn_window_length":600,
        "min_presigmoid_value":-70,
    },
    "paths":{
        "model_dir":"/home/armstrong/Research/constant_boxcar_widths_NGB/selected_models",
        "data_dir":"/home/armstrong/Research/constant_boxcar_widths_NGB/data",
        "output_dir": "/home/armstrong/Research/constant_boxcar_widths_NGB/applied_results_20141127"
    },
    "models":{
        "pDetector3c": "pDetectorMew_model_026.pt", 
        "sDetector": "sDetector_model032.pt", 
        "swagPPicker": ["pPicker_swag60_seed1.pt", "pPicker_swag60_seed2.pt", "pPicker_swag60_seed3.pt"], 
        "fmPicker": "fmPicker_model002.pt",
        "swagSPicker": ["sPicker_swag56_seed1.pt", "sPicker_swag56_seed2.pt", "sPicker_swag56_seed3.pt"],
        "pDetector1c":"oneCompPDetectorMEW_model_022.pt"
        },
    "output_file_names":{
        "catalog_file":"20221128_swag_pick_catalog",
        "output_probability_file":None,
    },
    "swag_info":{
        "seeds": [1, 2, 3],
        "K": 20,
        "num_workers": 4,
        "cov_mat":True,
        "p_train_file":"uuss_train_4s_1dup.h5",
        "p_data_path":"/home/armstrong/Research/git_repos/patprob/no_duplicates/uuss_data/p_resampled",
        "s_train_file":"uuss_train_6s_1dup.h5",
        "s_data_path":"/home/armstrong/Research/git_repos/patprob/no_duplicates/uuss_data/s_resampled",
        "pred_out_dir": f"swag_predictions", 
        "P_calibration_file":"p_calibration_model_medians.joblib",
        "S_calibration_file":"s_calibration_model_medians.joblib",
        "calibration_path":None,
        "CI_lower_bound":0.05,
        "CI_upper_bound":0.95,
        "N_P":40,
        "N_S":50,
        "picker_summary_method":"median"
    },
    "options":{
        "device": "cuda:0",
        "single_stat_string": None,
        "debug_s_detector": False,
        "debug_inds_file":None,
        "p_detection_thresh":0.75,
        "s_detection_thresh":0.75
    },
}
