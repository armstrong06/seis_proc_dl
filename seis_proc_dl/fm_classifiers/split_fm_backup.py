import sys
sys.path.append("/uufs/chpc.utah.edu/common/home/u1072028/PycharmProjects/seis-proc-dl/data_processing")
from split_data_cnn import SplitData, FirstMotionHelper
import extract_events
import pandas as pd
import h5py
import numpy as np

np.random.seed(388382)
flip_some_blasts = False
max_source_receiver_distance = 350 #120
downsample_historical_validation_test_waveforms = True
# It seems like the number of up picks is about equal to the number
# of down picks.  However, the number of down picks tends to be half
# the number of up picks (and this isn't just our catalog).
upsample_down_class = False
# Alternatively just upsample all minority classes
upsample_minority = True

pref = "/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/sPicker"
outdir = f'{pref}/s_resampled_picker/uuss'

# NGB swarm bounds
lat_min = 44.680
lat_max = 44.870
lon_max = 110.850
lon_min = 110.640
date_min = 130901  # YYMMDD
date_max = 140630  # YYMMDD

bounds = {"lat_min": lat_min,
          "lat_max": lat_max,
          "lon_max": lon_max,
          "lon_min": lon_min,
          "date_min": date_min,
          "date_max": date_max}

print("Loading catalogs...")
current_earthquake_catalog_df = pd.read_csv(f'{pref}/current_earthquake_catalog_s.csv')
current_earthquake_catalog_h5 = h5py.File(f'{pref}/current_earthquake_catalog_s.h5', 'r')
historical_earthquake_catalog_df = pd.read_csv(f'{pref}/historical_earthquake_catalog_s.csv')
historical_earthquake_catalog_h5 = h5py.File(f'{pref}/historical_earthquake_catalog_s.h5', 'r')
current_blast_catalog_df = pd.read_csv(f'{pref}/current_blast_catalog_s.csv')
current_blast_catalog_h5 = h5py.File(f'{pref}/current_blast_catalog_s.h5', 'r')

assert np.sum( (current_blast_catalog_df['first_motion'] ==-1)*1 ) == 0, 'blasts should not have negative first motions'

# current_earthquake_catalog_NGB_df, current_earthquake_catalog_noNGB_df = extract_events.separate_events(current_earthquake_catalog_df, bounds)
# # There are no NGB events for the given time period in blast or historical eq catalog
# X_NGB_eqc, y_NGB_eqc = extract_events.grab_from_h5files(current_earthquake_catalog_h5["X"][:, :],
#                                                        current_earthquake_catalog_h5["Y"][:],
#                                                         current_earthquake_catalog_NGB_df)

spliter = SplitData()
spliter.make_directory(outdir)

print("Partitioning current earthquake catalog")
X_train_eqc_h5, y_train_eqc_h5, train_eqc_df, \
X_validation_eqc_h5, y_validation_eqc_h5, validation_eqc_df, \
X_test_eqc_h5, y_test_eqc_h5, test_eqc_df, NGB_data \
    = spliter.split_event_wise(current_earthquake_catalog_df, current_earthquake_catalog_h5,
                       train_size=0.8, validation_size=0.1, test_size=0.1, extract_event_bounds=bounds)

print("Partitioning historical earthquake catalog")
X_train_eqa_h5, y_train_eqa_h5, train_eqa_df, \
X_validation_eqa_h5, y_validation_eqa_h5, validation_eqa_df, \
X_test_eqa_h5, y_test_eqa_h5, test_eqa_df \
    = spliter.split_event_wise(historical_earthquake_catalog_df, historical_earthquake_catalog_h5,
                       train_size=0.1, validation_size=0.01, test_size=0.89,
                       min_training_quality=0.75)

print("Partitioning current blast catalog")
X_train_blc_h5, y_train_blc_h5, train_blc_df, \
X_validation_blc_h5, y_validation_blc_h5, validation_blc_df, \
X_test_blc_h5, y_test_blc_h5, test_blc_df \
    = spliter.split_event_wise(current_blast_catalog_df, current_blast_catalog_h5,
                       train_size=0.6, validation_size=0.1, test_size=0.3,
                       min_training_quality=1.0)

if (downsample_historical_validation_test_waveforms):
    print("Downsampling historical validation/test waveforms to be size of current earthquakes")
    n_want = len(y_validation_eqc_h5)
    if (n_want < len(validation_eqa_df)):
        keep_rows = np.random.choice(len(validation_eqa_df), size = n_want, replace = False)
        X_validation_eqa_h5 = X_validation_eqa_h5[keep_rows,:]
        y_validation_eqa_h5 = y_validation_eqa_h5[keep_rows]
        validation_eqa_df = validation_eqa_df.iloc[keep_rows]

    n_want = len(y_test_eqc_h5)
    if (n_want < len(test_eqa_df)):
        keep_rows = np.random.choice(len(test_eqa_df), size = n_want, replace = False)
        X_test_eqa_h5 = X_test_eqa_h5[keep_rows,:]
        y_test_eqa_h5 = y_test_eqa_h5[keep_rows]
        test_eqa_df = test_eqa_df.iloc[keep_rows]

print("Concatenating catalogs...")
X_train = np.concatenate([X_train_eqc_h5, X_train_blc_h5, X_train_eqa_h5])
y_train = np.concatenate([y_train_eqc_h5, y_train_blc_h5, y_train_eqa_h5])
train_df = pd.concat([train_eqc_df, train_blc_df, train_eqa_df])

print("Training number of up:", np.sum( (y_train == 1)*1) )
print("Training number of down:", np.sum( (y_train ==-1)*1 ) )
print("Training number of unknown:", np.sum( (y_train == 0)*1 ) )

if (upsample_down_class):
    print("Upsampling down class")
    X_train, y_train, train_df = FirstMotionHelper.upsample_down(X_train, y_train, train_df)
    assert np.sum( (y_train != train_df.first_motion)*1 ) == 0, 'df polarities do not match labels'
    print("Training number of up:", np.sum( (y_train == 1)*1) )
    print("Training number of down:", np.sum( (y_train ==-1)*1 ) )
    print("Training number of unknown:", np.sum( (y_train == 0)*1 ) )
if (upsample_minority):
    print("Upsampling minority class")
    X_train, y_train, train_df = FirstMotionHelper.upsample_minority_class(X_train, y_train, train_df)
    print("Number of up:", np.sum(1*(train_df.first_motion == 1)))
    print("Number of down:", np.sum(1*(train_df.first_motion ==-1)))
    print("Number of unknown", np.sum(1*(train_df.first_motion == 0)))

print("Removing all training distances greater than:", max_source_receiver_distance)
print(len(y_train))
lkeep = train_df.source_receiver_distance < max_source_receiver_distance
train_df = train_df[lkeep]
X_train = X_train[lkeep,:]
y_train = y_train[lkeep]
print(len(y_train))

X_validation = np.concatenate([X_validation_eqc_h5, X_validation_blc_h5, X_validation_eqa_h5])
y_validation = np.concatenate([y_validation_eqc_h5, y_validation_blc_h5, y_validation_eqa_h5])
validation_df = pd.concat([validation_eqc_df, validation_blc_df, validation_eqa_df])

X_test = np.concatenate([X_test_eqc_h5, X_test_blc_h5, X_test_eqa_h5])
y_test = np.concatenate([y_test_eqc_h5, y_test_blc_h5, y_test_eqa_h5])
test_df = pd.concat([test_eqc_df, test_blc_df, test_eqa_df])

spliter.write_data_and_dataframes(outdir,
                              X_train, y_train, train_df,
                              X_validation, y_validation, validation_df,
                              X_test, y_test, test_df)

extract_events.write_h5file(NGB_data[0], NGB_data[1], f"{outdir}_NGB.h5")
NGB_data[2].to_csv(f"{outdir}_NGB.csv", index=False)
