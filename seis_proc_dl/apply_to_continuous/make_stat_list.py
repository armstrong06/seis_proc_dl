import obspy
import os
import numpy as np
import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("--ddir", type=str, help="path to data directory",
                       default="/uufs/chpc.utah.edu/common/home/koper-group3/alysha/ys_data/downloaded_all_data")
argParser.add_argument("-y", "--year", type=str, help="year to gather stations")
argParser.add_argument("-c", "--ncomps", type=int, help="number of components")
argParser.add_argument("-s", "--stattype", type=str, help="first letter(s) of the channel code", default="[HEB]")
argParser.add_argument("--outdir", type=str, help="output directory", default=None)
argParser.add_argument("-f", "--outfile", type=str, help="output filename", default=None)
argParser.add_argument("-n", "--nstats", type=int, help="limit on the number of stations", default=None)
args = argParser.parse_args()

year = args.year
ncomps = args.ncomps
stat_type = args.stattype
outdir = args.outdir
outfile = args.outfile
ddir = args.ddir
nstats = args.nstats

if stat_type is None:
    stat_type = "[HEB]"

if outfile is None and nstats is None:
    outfile = f"station.list.{year}.{ncomps}C.{stat_type}H.txt"
elif nstats is not None:
    outfile = f"station.list.{year}.{ncomps}C.{stat_type}H.{nstats}stats.txt"

if outdir is not None:
    outfile = os.path.join(outdir, outfile)

assert ncomps in [1, 3], "Invalid number of components"

inv = obspy.read_inventory(os.path.join(ddir, str(year), "stations/*"))

chan_list =  [chan[:-1] for chan in inv.select(channel=f"{stat_type}??").get_contents()['channels']]

uniq_chans, chan_cnts = np.unique(chan_list, return_counts=True)

if ncomps == 3:
    selected_chans = uniq_chans[np.where(chan_cnts>=ncomps)[0]]
else:
    selected_chans = uniq_chans[np.where(chan_cnts==1)[0]]

if nstats is not None:
    selected_chans = selected_chans[0:nstats]

with open(os.path.join(outfile), "w") as f:
    f.write(f"{len(selected_chans)}\n")
    for chan in selected_chans:
        chan_comps = chan.split(".")
        f.write(f"{chan_comps[1]} {chan_comps[3]}\n")