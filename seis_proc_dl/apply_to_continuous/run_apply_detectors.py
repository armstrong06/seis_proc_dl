from seis_proc_dl.apply_to_continuous import apply_detectors
import glob
import argparse
import datetime
import os

# Path to the miniseed files
data_dir = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/ys_data/downloaded_all_data"
# Path to the trained detector models
models_path = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models"
# Specific phase models
p_model = os.path.join(models_path, "pDetectorMew_model_026.pt")
onec_p_model = os.path.join(models_path, "oneCompPDetectorMEW_model_022.pt")
s_model = os.path.join(models_path, "sDetector_model032.pt")
# The size of the detector input in samples
window_length = 1008
# The length of the sliding window in samples
sliding_interval = 500
# The samples +- the center of the detector output to return 
center_window = 250 
# The number of samples on either side of the center window 
window_edge_npts = 254
# The output directory for the posterior probs and the waveform metadata 
outdir = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/ys_data/detector_output"

def load_1c_detector():
    """Initialize the 1C P Phase Detector for CHPC

    Returns:
        object: One Component P Phase Detector
    """
    pd = apply_detectors.PhaseDetector(onec_p_model,
                                        1,
                                        "P",
                                        min_presigmoid_value=-70,
                                        device="cpu",
                                        num_torch_threads=2)
    return pd

def load_3c_detectors():
    """Initialize the 3C P and S Phase Detectors for CHPC

    Returns:
        tuple: 3C P Detector, 3C S Detector
    """
    p_pd = apply_detectors.PhaseDetector(p_model,
                                         3,
                                         "P",
                                        min_presigmoid_value=-70,
                                        device="cpu",
                                        num_torch_threads=2)
    
    s_pd = apply_detectors.PhaseDetector(s_model,
                                        3,
                                        "S",
                                        min_presigmoid_value=-70,
                                        device="cpu",
                                        num_torch_threads=2)
    
    return p_pd, s_pd

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

### Set the start date and time increment of the files ###
date = datetime.date(args.year, args.month, args.day)
delta = datetime.timedelta(days=1)

### Load the DataLoader and PhaseDetector(s), which only need to be initialized once ###
dataloader = apply_detectors.DataLoader()
s_detector = None
if args.chan == "EH":
    p_detector = load_1c_detector()
    p_proc_func = dataloader.process_1c_P
else:
    p_detector, s_detector = load_3c_detectors()
    p_proc_func = dataloader.process_3c_P

### Iterate over the specified number of days ###
for _ in range(args.n):
    ### The data files are organized Y/m/d, get the appropriate date/station files ###
    date_str = date.strftime("%Y/%m/%d")
    files = glob.glob(os.path.join(data_dir, date_str, f'*{args.stat}*{args.chan}*'))

    ### Make the output dirs have the same structure as the data dirs ###
    date_outdir = os.path.join(outdir, date_str)

    ### If there are no files for that station/day, move to the next day ###
    if len(files) == 0:
        print(f'No data for {date_str}')
        continue

    ### There will always be a P detector, either 1c or 3c ###
    if args.chan == "EH":
        dataloader.load_1c_data(files[0])
    else:
        dataloader.load_3c_data(files[0], files[1], files[2])

    p_data, start_pad_npts, end_pad_npts = dataloader.format_continuous_for_unet(window_length,
                                                                                sliding_interval,
                                                                                p_proc_func,
                                                                                normalize=True
                                                                                )
    p_probs_outfile_name = p_detector.make_outfile_name(files[0], date_outdir)
    p_cont_post_probs = p_detector.get_continuous_post_probs(p_data, 
                                                             center_window,
                                                             window_edge_npts,
                                                             start_pad_npts=start_pad_npts, 
                                                             end_pad_npts=end_pad_npts)
    p_detector.save_post_probs(p_probs_outfile_name, p_cont_post_probs, dataloader.metadata)

    ### If 3C, run the S detector as well ###
    if s_detector is not None:
        s_data, _, _ = dataloader.format_continuous_for_unet(window_length,
                                                            sliding_interval,
                                                            dataloader.process_3c_S
                                                            )
        s_probs_outfile_name = s_detector.make_outfile_name(files[0], date_outdir)

        s_cont_post_probs = s_detector.get_continuous_post_probs(s_data, 
                                                                 center_window,
                                                                 window_edge_npts,
                                                                 start_pad_npts=start_pad_npts, 
                                                                 end_pad_npts=end_pad_npts)
        s_detector.save_post_probs(s_probs_outfile_name, s_cont_post_probs, dataloader.metadata)

    ### Save the station meta info (including gaps) to a file in the same dir as the post probs. ###
    ### Only need one file per station/day pair ###
    meta_outfile_name =  dataloader.make_outfile_name(files[0], date_outdir)
    dataloader.write_data_info(meta_outfile_name)

    ### Move on to the next day ###
    date += delta

# print(date)

# print("args=%s" % args)

# print("args.stat=%s" % args.stat)