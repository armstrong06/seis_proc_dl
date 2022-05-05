import sys
sys.path.append("/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis-proc-dl/data_processing")
from split_data_cnn import SplitData
import pandas as pd
import h5py

pref = "/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/stead"
catalog_df_filename = f"{pref}/SStead_2000.csv"
catalog_h5_filename = f"{pref}/SStead_2000.h5"
outdir = f'{pref}/s_resampled_picker/s_picker'

df = pd.read_csv(catalog_df_filename)
hf = h5py.File(catalog_h5_filename, "r")

spliter = SplitData()
spliter.split_event_wise_and_write(df, hf, output_file_root=outdir,
                               train_size = 0.8,
                               validation_size = 0.1,
                               test_size = 0.1,
                               min_training_quality = -1, is_stead=True)