import os
import argparse
import json
import time
from seis_proc_dl.apply_to_continuous import apply_swag_pickers

### Handle user inputs for picker ###
argParser = argparse.ArgumentParser()
#argParser.add_argument("-p", "--is_p", type=bool, help='True if P arrival, False if S')
argParser.add_argument("-p", "--is_p", action=argparse.BooleanOptionalAction, help='True if P arrival, False if S')
argParser.add_argument('-s1',"--swag_model1", type=str, help='first swag model file name')
argParser.add_argument('-s2',"--swag_model2", type=str, help='second swag model file name')
argParser.add_argument('-s3',"--swag_model3", type=str, help='third swag model file name')
argParser.add_argument('-tf',"--train_file", type=str, help='training data file (no path)')
argParser.add_argument('-df',"--data_file", type=str, help='new data h5 file name (no path)')
argParser.add_argument('-dm',"--meta_file", type=str, help='new data meta (csv) file name (no path)')
argParser.add_argument('-cm',"--cal_file", type=str, help='calibration model file name')
argParser.add_argument('-n',"--N", type=int, help='Number of samples to draw from each model')
argParser.add_argument('-r',"--region", type=str, help='region for the output file name', default='ynpEarthquake')

argParser.add_argument('-tb',"--train_batchsize", type=int, help='batch size for the training data bn_update', default=512)
argParser.add_argument('-db',"--data_batchsize", type=int, help='batch size for the new data', default=512)
argParser.add_argument('-tw',"--train_n_workers", type=int, help='number of workers for the training data loader', default=4)
argParser.add_argument('-dw',"--data_n_workers", type=int, help='number of workers for the data loader', default=4)
argParser.add_argument('-e',"--n_data_examples", type=int, help='number of data examples to use. -1 for all examples (default)', default=-1)

argParser.add_argument("-d", "--device", type=str, help='device to use', default='cuda:0')
argParser.add_argument('-m',"--model_path", type=str, help='path to the stored models', 
        default="/uufs/chpc.utah.edu/common/home/koper-group3/alysha/selected_models")
argParser.add_argument('-s',"--seeds", type=list, help='intial seeds for the models', default=[1, 2, 3])
#argParser.add_argument('-c',"--cov_mat", type=bool, help='swag cov_mat', default=True)
argParser.add_argument('-c',"--cov_mat", action=argparse.BooleanOptionalAction, help='swag cov_mat', default=True)
argParser.add_argument('-k',"--K", type=int, help='max number of swag models stored', default=20)
argParser.add_argument('-tp',"--train_path", type=str, help='training data path',
                        default="/uufs/chpc.utah.edu/common/home/koper-group3/alysha/swag_info")
argParser.add_argument('-dp',"--data_path", type=str, help='new data path',
                        default="/uufs/chpc.utah.edu/common/home/koper-group4/bbaker/machineLearning/harvestPicks/gcc_build")
#argParser.add_argument('-st',"--shuffle_train", type=bool, help='shuffle the training data for bn_update', default=False)
argParser.add_argument('-st',"--shuffle_train", action=argparse.BooleanOptionalAction, help='shuffle the training data for bn_update', default=False)
argParser.add_argument('-lb',"--lb", type=float, help='lower bound for the calibrated credible intervals', default=0.05)
argParser.add_argument('-ub',"--ub", type=float, help='upper bound for the calibrated credible intervals', default=0.95)
argParser.add_argument('-o',"--output_dir", type=str, help='output file path',
                        default="/uufs/chpc.utah.edu/common/home/koper-group3/alysha/harvestPicks")
#argParser.add_argument('--save_args', type=bool, help='write args to json file in the output dir', default=True )
argParser.add_argument('--save_args', action=argparse.BooleanOptionalAction, help='write args to json file in the output dir', default=True)
args = argParser.parse_args()

assert os.path.exists(os.path.join(args.model_path, args.swag_model1)), 'SWAG model 1 path incorrect'
assert os.path.exists(os.path.join(args.model_path, args.swag_model2)), 'SWAG model 2 path incorrect'
assert os.path.exists(os.path.join(args.model_path, args.swag_model3)), 'SWAG model 3 path incorrect'
assert os.path.exists(os.path.join(args.model_path, args.cal_file)), 'Calibration file path incorrect'
assert os.path.exists(os.path.join(args.train_path, args.train_file)), 'Train file path incorrect'
assert os.path.exists(os.path.join(args.data_path, args.data_file)), 'Data h5 file path incorrect'
assert os.path.exists(os.path.join(args.data_path, args.meta_file)), 'Data meta csv file path incorrect'
assert os.path.exists(args.output_dir), 'output directory does not exist'
assert args.device in ['cpu', 'cuda:0', 'cuda:1'], 'invalid device type'
assert len(args.seeds) == 3, "Incorrect number of seeds"
assert args.lb < args.ub, "lower bound must be less than upper bound"
assert args.lb >=0 and args.lb < 1, "lower bound value is incorrect"
assert args.ub >0 and args.ub <= 1, "upper bound value is incorrect"

print("is_p:", args.is_p)
print("cov_mat:", args.cov_mat)
print("shuffle_train:", args.shuffle_train)
print("save_args:", args.save_args)
if args.save_args:
    phase = "P"
    if not args.is_p:
        phase = "S"
    arg_outfile = os.path.join(args.output_dir, f"args.{phase}.{args.region}.json")
    print("writing", arg_outfile)
    with open(arg_outfile, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

#############
st = time.time()
# Initialize the picker
sp = apply_swag_pickers.MultiSWAGPicker(is_p_picker=args.is_p, device=args.device)
# Load the training data for bn_updates
train_loader = sp.torch_loader(args.train_file,
                             args.train_path,
                             args.train_batchsize, 
                             args.train_n_workers, 
                             shuffle=args.shuffle_train)
# Load the new estimated picks
data_loader = sp.torch_loader(args.data_file,
                             args.data_path,
                             args.data_batchsize,
                             args.data_n_workers,
                             shuffle=False,
                             n_examples=args.n_data_examples)
# Load the MultiSWAG ensemble
ensemble = sp.load_swag_ensemble(args.model_path, args.swag_model1, args.swag_model2, args.swag_model3, args.seeds, args.cov_mat, args.K)
# Get the calibrated lower and upper bounds
cal_path = os.path.join(args.model_path, args.cal_file)
lb_t, ub_t = sp.get_calibrated_pick_bounds(cal_path, args.lb, args.ub)
# Get the posterior predictive distributions for each pick
new_preds = sp.apply_picker(ensemble, data_loader, train_loader, args.N)
# Trim the outliers and compute the pick correction (median) and std
trimmed_medians, trimmed_stds = sp.trim_inner_fence(new_preds)
# Calibrate the predictions and output the MultiSWAG summary info for each pick
summary = sp.calibrate_swag_predictions(trimmed_medians, trimmed_stds, lb_t, ub_t)
# Save the pick summary information in a csv file and all the picks to an h5 file
meta_path = os.path.join(args.data_path, args.meta_file)
sp.format_and_save(meta_path, summary, new_preds, args.output_dir, args.region, n_meta_rows=args.n_data_examples)

et = time.time()
print(f"Total time: {et-st:2.3f} s")
