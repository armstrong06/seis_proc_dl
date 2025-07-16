import argparse
import importlib
import sys
import time

from seis_proc_dl.apply_to_continuous.database_connector import store_source_method_info

### Handle user inputs for when to run the detector and for which station ###
argParser = argparse.ArgumentParser()
argParser.add_argument("--cfg", type=str, help="path to configuration file")
argParser.add_argument(
    "--ncomps", type=int, help="The number of components for the models"
)

args = argParser.parse_args()
assert args.ncomps in [1, 3], "Invalid number of components"


def import_from_path(module_name, file_path):
    """
    From https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
    """
    print(f"Path to config file: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Import config from path (that way this script does not have to be next to the config file)
CFG = import_from_path("detector_config", args.cfg).CFG

t0 = time.time()
store_source_method_info(args.ncomps, CFG)
print(
    f"Total time to store waveform source and detection methods: {time.time() - t0:.2f} s"
)
