from os import device_encoding
import sys
sys.path.insert(0, "/home/armstrong/Research/git_repos/seis-proc-dl")
from detectors.models.unet import UNet
from test_train_unet_config import CFG

pref = "/home/armstrong/Research/git_repos/seis-proc-dl/detectors/testing/P_models_32_0.01"
specified_models = [f"{pref}/model_P_000.pt", f"{pref}/model_P_001.pt"]
test_file = "/home/armstrong/Research/git_repos/seis-proc-dl/detectors/testing/data/p_validate.10s.2dup.h5"

unet = UNet(CFG)
# unet.load_model_state("/home/armstrong/Research/git_repos/seis-proc-dl/detectors/testing/P_models_32_0.01/model_P_000.pt")
unet.evaluate_specified_models(test_file, specified_models)
