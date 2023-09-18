"""Model config in json format for applying models to continous data """

CFG = {
    "model_params":{
        "batch_size":64,
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
        "output_dir": "/home/armstrong/Research/constant_boxcar_widths_NGB/applied_results_20141127/s_failures"
    },
    "models":{
        "pDetector3c": "pDetectorMew_model_026.pt", 
        "sDetector": "sDetector_model032.pt", 
        "pPicker":'pPicker_ADAM_model003.pt',
        "sPicker":'sPicker_ADAM_model019.pt',
        "fmPicker": "fmPicker_model002.pt",
        "pDetector1c":"oneCompPDetectorMEW_model_022.pt"
        },
    "output_file_names":{
        "catalog_file":"B944",
        "output_probability_file":"B944.probs.h5",
    },
    "swag_info":None,
    "options":{
        "device": "cuda:0",
        "single_stat_string":"B944*EH",
        "debug_s_detector": True,
        "debug_inds_file":None,
        "p_detection_thresh":0.75,
        "s_detection_thresh":0.75
    },
}
