from os import device_encoding
import sys
sys.path.insert(0, "/home/armstrong/Research/git_repos/seis-proc-dl")
from test_train_unet_config import CFG
from evaluation.unet_evaluator import UNetEvaluator


evaluator = UNetEvaluator()
