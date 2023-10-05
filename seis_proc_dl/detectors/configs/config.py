"""Model config for 3C, P-picker in json format"""

CFG = {
    "data":{
        "maxlag":250,
        
    },
    "train": {
        "torch_device": "cuda:0",
        "batch_size": 512, 
        "epochs": 15, 
        "learning_rate": 0.01, 
        "detection_threshold":0.5, 
        "train_hdf5_file": "/data/train.h5py", 
        "validation_h5py_file": None, 
        "model_out_directory": "/models"
    }, 
    "model":{
        "num_classes":1, 
        "num_channels":3, 
        "phase_type":"P",
        "minimum_presigmoid_value":-70
    }
}