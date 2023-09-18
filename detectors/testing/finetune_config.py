"""Model config in json format for testing 3C, P-picker """

CFG = {
    "data":{
        "maxlag":250,
    },
    "train": {
        "torch_device": "cuda:0",
        "batch_size": 256, 
        "epochs": 30, 
        "learning_rate": 0.001, 
        "detection_threshold":0.5, 
        "train_hdf5_file": "/home/armstrong/Research/new_boxcar_widths/pDetector/uuss_data/combined.train.10s.1dup.h5",
        "validation_hdf5_file": None, 
        "model_out_directory": "/home/armstrong/Research/new_boxcar_widths/pDetector/finetuned"
    }, 
    "model":{
        "num_classes":1, 
        "num_channels":3, 
        "phase_type":"P",
        "minimum_presigmoid_value":-70
    }
}
