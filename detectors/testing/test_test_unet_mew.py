import sys
sys.path.insert(0, "/home/armstrong/Research/git_repos/seis-proc-dl")
from detectors.models.unet import UNet
from test_mew_config import CFG


test_file = "/home/armstrong/Research/new/p_detector_mew/uuss_data/validateP.10s.1dup_synthetic_multievent_waveforms.h5"

unet = UNet(CFG)

# To test a single model
# unet.load_model_state("/home/armstrong/Research/git_repos/seis-proc-dl/detectors/testing/P_models_32_0.01/model_P_000.pt")
# unet.evaluate(test_file)
########################

# to evaluate multiple models given epochs
unet.evaluate_specified_models(test_file, [0, 1], mew=True)
#############################
