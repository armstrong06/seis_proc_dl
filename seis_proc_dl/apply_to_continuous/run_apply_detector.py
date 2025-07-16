import time
import argparse
import sys
import json
import importlib
from seis_proc_dl.apply_to_continuous import apply_detectors

# from detector_config import CFG

### Handle user inputs for when to run the detector and for which station ###
argParser = argparse.ArgumentParser()
argParser.add_argument("--cfg", type=str, help="path to configuration file")
argParser.add_argument("-net", "--net", type=str, help="network code")
argParser.add_argument("-s", "--stat", type=str, help="station code")
argParser.add_argument("-l", "--loc", type=str, help="location code")
argParser.add_argument(
    "-c", "--chan", type=str, help="First two letter of the channel code"
)
argParser.add_argument("-y", "--year", type=int, help="Start year")
argParser.add_argument("-m", "--month", type=int, help="Start month")
argParser.add_argument("-d", "--day", type=int, help="Start day")
argParser.add_argument("-n", "--n", type=int, help="The number of days to analyze")
argParser.add_argument(
    "--ncomps", type=int, help="The number of components for the models"
)

args = argParser.parse_args()

assert args.year >= 2002 and args.year <= 2025, "Year is invalid"
assert args.month > 0 and args.month < 13, "Month is invalid"
assert args.day > 0 and args.day <= 31, "Day is invalid"
assert args.n > 0, "Number of days is invalid"
assert args.ncomps in [1, 3], "Invalid number of components"

#############


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

stime = time.time()
print(args.net, args.stat, args.loc, args.chan, args.year, args.month, args.day, args.n)
applier = apply_detectors.ApplyDetector(args.ncomps, CFG)
applier.apply_to_multiple_days(
    args.net,
    args.stat,
    args.loc.strip('"'),
    args.chan,
    args.year,
    args.month,
    args.day,
    args.n,
)
etime = time.time()
print(f"Total time: {etime-stime:0.6f} s")
