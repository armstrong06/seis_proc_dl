import sys
sys.path.insert(0, "/home/armstrong/Research/git_repos/seis-proc-dl")
from detectors.models.unet import UNet
#sys.path.insert(0, "/home/armstrong/Research/git_repos/seis-proc-dl")
from test_train_unet_config import CFG

unet = UNet(CFG)
unet.train()
