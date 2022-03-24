"""Model config in json format for testing 3C, P-picker """

CFG = {
    "data":{
        "maxlag":250,
    },
    "train": {
        "torch_device": "cuda:0",
        "batch_size": 32, 
        "epochs": 2, 
        "learning_rate": 0.01, 
        "detection_threshold":0.5, 
        "train_hdf5_file": "/home/armstrong/Research/git_repos/seis-proc-dl/detectors/testing/data/p_train.10s.2dup.h5",
        "validation_hdf5_file": None, 
        "validation_h5py_file": None, 
        "model_out_directory": "/home/armstrong/Research/git_repos/seis-proc-dl/detectors/testing"
    }, 
    "model":{
        "num_classes":1, 
        "num_channels":3, 
        "phase_type":"P",
        "minimum_presigmoid_value":-70
    }
}