import sys
sys.path.insert(0, "/home/armstrong/Research/git_repos/seis-proc-dl")
from detectors.models.unet import UNet
from finetune_config import CFG

pref="/home/armstrong/Research/new_boxcar_widths/pDetector/uuss_data"

validate_file = f"{pref}/combined.validate.10s.1dup"
ngb_file = f"{pref}/currenteq.NGB.10s.1dup"
mew_file = f"/home/armstrong/Research/new_boxcar_widths/pDetector_mew/uuss_data/validateP.10s.1dup_synthetic_multievent_waveforms"

for split_type, test_file in zip(["validate"], [validate_file]):
    print(split_type, test_file)
    unet = UNet(CFG)
    if split_type == "validate":
        print("Using df")
        unet.evaluate_specified_models_new(f"{test_file}.h5", range(-1, 30), split_type, batch_size=32, df=f"{test_file}.csv")
    else:
        unet.evaluate_specified_models_new(f"{test_file}.h5", range(-1, 30), split_type, batch_size=32)
