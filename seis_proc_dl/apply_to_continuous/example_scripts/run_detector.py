import time
import argparse
import sys
from seis_proc_dl.apply_to_continuous import apply_detectors
from detector_config import CFG

### Handle user inputs for when to run the detector and for which station ###
argParser = argparse.ArgumentParser()
argParser.add_argument("-s", "--stat", type=str, help="station code")
argParser.add_argument("-c", "--chan", type=str, help="First two letter of the channel code")
argParser.add_argument("-y", "--year", type=int, help="Start year")
argParser.add_argument("-m", "--month", type=int, help="Start month")
argParser.add_argument("-d", "--day", type=int, help="Start day")
argParser.add_argument("-n", "--n", type=int, help="The number of days to analyze")

args = argParser.parse_args()

assert args.year >= 2002 and args.year <= 2022, "Year is invalid"
assert args.month > 0 and args.month < 13, "Month is invalid"
assert args.day > 0 and args.day <= 31, "Day is invalid"
assert args.n > 0, "Number of days is invalid"

#############

stime = time.time()
print(args.stat, args.chan, args.year, args.month, args.day, args.n)
applier = apply_detectors.ApplyDetector(3, CFG)
applier.apply_to_multiple_days(args.stat, args.chan, args.year, args.month, args.day, args.n)
etime = time.time()
print(f"Total time: {etime-stime:0.6f} s")

