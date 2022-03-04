"""
Remove certain events  from a data set given location and time bounds and save them separately
Author: Alysha Armstrong 9/9/2021
"""

import pandas as pd
import h5py
import numpy as np
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pyplot as plt

def read_h5file(file):
    h5 = h5py.File(file, "r")
    X = h5["X"][:]
    Y = h5["Y"][:]
    h5.close()

    return (X, Y)

def write_h5file(X, Y, outfile, T=None):
    hfout = h5py.File(outfile, 'w')
    hfout.create_dataset('X', data=X)
    hfout.create_dataset('Y', data=Y)
    if T is not None:
        hfout.create_dataset('Pick_index', data=T)
    hfout.close()
    print("data saved in", outfile)

def separate_events(evmeta, bounds, to_plot=True):
    # put index in order so grab correct waveforms later
    evmeta["original_rows"] = np.arange(len(evmeta))
    evmeta['qc_rows'] = np.copy(np.arange(len(evmeta)))
    # add date column to format origin time like date_min & date_max
    evmeta["date"] = evmeta["origin_time"].apply(lambda x: int(UTCDateTime(x).strftime("%y%m%d")))

    # Filter the metadata to get events within the bounds to extract
    extracted_event_meta = evmeta.loc[
        (evmeta["date"] >= bounds["date_min"]) & (evmeta["date"] <= bounds["date_max"]) & (evmeta["event_lat"] >= bounds["lat_min"])
        & (evmeta["event_lat"] <= bounds["lat_max"]) & (-1 * evmeta["event_lon"] >= bounds["lon_min"]) & (
                    -1 * evmeta["event_lon"] <= bounds["lon_max"])]

    # plot extracted events to make sure locations make sense
    if to_plot:
        plt.scatter(extracted_event_meta["event_lon"], extracted_event_meta["event_lat"])
        plt.title("Locations of Extracted Events")
        plt.show()

    # Get indices of events to keep
    kept_event_inds = np.delete(np.arange(len(evmeta)), extracted_event_meta["original_rows"])
    assert ~np.any(
        np.isin(kept_event_inds, extracted_event_meta["original_rows"])), "IDs are not unique between kept and extracted events"

    # make new df of metadata for kept waveforms
    kept_event_meta = evmeta[evmeta["original_rows"].isin(kept_event_inds)]
    assert len(kept_event_inds) == len(evmeta) - len(extracted_event_meta), "# of indices are not matching up"
    assert len(kept_event_meta) == len(evmeta) - len(extracted_event_meta), "# of events in metadata are not matching up"

    return extracted_event_meta, kept_event_meta

def grab_from_h5files(X, Y, meta):
    # Grab all the data not including the extracted events
    if len(X.shape) == 3:
        event_X = X[meta["original_rows"], :, :]
    else:
        event_X = X[meta["original_rows"], :]

    event_Y = Y[meta["original_rows"]]
    print(event_X.shape, event_Y.shape)

    assert event_X.shape[0] == len(meta)
    assert event_Y.shape[0] == len(meta)
    assert X.shape[1] == event_X.shape[1]
    if len(X.shape) == 3:
        assert X.shape[2] == event_X.shape[2]

    return event_X, event_Y

def write_data_files(h5_file, meta_df, outfile_pref):
    X, Y =  grab_from_h5files(h5_file["X"][:], h5_file["Y"][:], meta_df)
    write_h5file(X, Y, outfile_pref+".h5")
    meta_df.to_csv(outfile_pref+".csv", index=False)

if __name__ == "__main__":

    ################ Start Setting Params ####################
    # Set parameters/files
    pref = "/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/threeComponent"
    infile_pref = "%s/current_earthquake_catalog_3C"%pref

    # output file paths for data with events removed
    kept_outfile = "%s/noNGB_current_earthquake_catalog_3C.h5"%pref
    kept_outmeta = "%s/noNGB_current_earthquake_catalog_3C.csv"%pref

    # output file path for the data that was evicted out of the original data set
    extracted_outfile = "%s/NGB_current_earthquake_catalog_3C.h5"%pref
    extracted_outmeta = "%s/NGB_current_earthquake_catalog_3C.csv"%pref

    # Remove events within these bounds
    lat_min = 44.680
    lat_max = 44.870
    lon_max = 110.850
    lon_min = 110.640
    date_min = 130901   #YYMMDD
    date_max = 140630   #YYMMDD
    ################ End Set Params ####################

    assert lon_min < lon_max, "lon min must be less than max"
    assert lat_min < lat_max, "lat min must be less than max"
    assert date_min < date_max, "date min must be less than max"

    bounds = {"lat_min":lat_min,
    "lat_max":lat_max,
    "lon_max":lon_max,
    "lon_min":lon_min,
    "date_min":date_min,
    "date_max":date_max}

    # read in metadata, add column for indices
    evmeta = pd.read_csv(infile_pref+".csv")

    # Read in h5 file
    X, Y = read_h5file(infile_pref+".h5")
    assert len(evmeta) == X.shape[0], "Number of events in csv and h5 files do not match"

    # Start processing steps...
    extracted_event_meta, kept_event_meta = separate_events(evmeta, bounds)

    print(X.shape, Y.shape)
    # Grab all the data not including the extracted events
    kept_event_X, kept_event_Y =  grab_from_h5files(X, Y, kept_event_meta)
    # Grab the data for the extracted events
    extracted_event_X, extracted_event_Y =  grab_from_h5files(X, Y, extracted_event_meta)

    # Just some sanity checks
    assert X.shape[0] - kept_event_X.shape[0] == extracted_event_X.shape[0]
    assert Y.shape[0] - kept_event_Y.shape[0] == extracted_event_Y.shape[0]

    # Write files for kept events and extracted events
    write_h5file(kept_event_X, kept_event_Y, kept_outfile)
    kept_event_meta.to_csv(kept_outmeta, index=False)

    write_h5file(extracted_event_X, extracted_event_Y, extracted_outfile)
    extracted_event_meta.to_csv(extracted_outmeta, index=False)