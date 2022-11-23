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
        "model_dir":"/home/armstrong/Research/newer/sg_selected_models",
        "data_dir":"/home/armstrong/Research/apply_models/data",
        "output_dir": "/home/armstrong/Research/newer/applied_results"
    },
    "models":{
        "pDetector3c": "pDetector_model027.pt", 
        "sDetector": "sDetector_model003.pt", 
        "swagPPicker": ["pPicker_model006.pt", "pPicker_model006.pt", "pPicker_model006.pt"], 
        "fmPicker": "fmPicker_model002.pt",
        "swagSPicker": ["sPicker_model012.pt", "sPicker_model012.pt", "sPicker_model012.pt"],
        "pDetector1c":"oneCompPDetector_model029.pt"
        },
    "output_file_names":{
        "catalog_file":"YGV.0403",
        "output_probability_file":None,
    },
    "swag_info":{
        "seeds": [1, 2, 3],
        "K": 20,
        "num_workers": 4,
        "cov_mat":True,
        "N": 30,
        "p_train_file":,
        "p_data_path":,
        "s_train_file":,
        "s_data_path":
        "pred_out_dir": f"{outdir}/swag_predictions", 
        "P_0.05_transform":,
        "P_0.95_transform":,
        "S_0.05_transform",
        "S_0.95_transform"
    },
    "options":{
        "device":"cuda:0",
        "single_stat_string": None,
        "debug_s_detector": False,
        "debug_inds_file":None,
    },
}
