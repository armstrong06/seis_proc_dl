import sys
sys.path.insert(0, "/home/armstrong/Research/git_repos/seis-proc-dl")
from pick_regressors.cnn_picker import Picker
from pick_regressors.testing.ppicker_config import CFG

picker = Picker(CFG)
#picker.train()

test_pref = "/home/armstrong/Research/newer/sPicker/uuss_data/uuss_NGB"
picker.evaluate_specified_models(f"{test_pref}.h5", f"{test_pref}.csv", range(2), "train")