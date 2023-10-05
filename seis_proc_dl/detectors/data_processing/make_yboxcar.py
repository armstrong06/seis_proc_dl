import h5py
import numpy as np
import pandas as pd

def read_h5file(file, num_events=None):
    h5 = h5py.File(file, "r")
    if num_events is None:
        num_events = h5["X"][:].shape[0]
    X = h5["X"][:num_events]
    Y = h5["Y"][:num_events]
    T = h5["Pick_index"][:num_events]
    h5.close()

    return (X, Y, T)

def find_widths_by_quality(weights, Y):
    values = {0: set(), 1: set(), 2: set()}
    for i in range(Y.shape[0]):
        y = Y[i, :, :]
        width = len(np.where(y > 0)[0])
        values[weights[i]].add(width)

    for key in iter(values):
        if len(values[key]) == 1:
            val = next(iter(values[key]))
            values[key] = val

    return values

def add_boxcar(df, values, X, Y, T=None, dt=0.01):
    """

    :param df:
    :param X:
    :param Y:
    :param T: Is None if the data has not been split yet and the Y data is just the second that has the pick
    :return:
    """
    pick_quals = df["pick_quality"].values
    quality_to_jiggle = {1.0: 0, 0.75: 1, 0.5: 2}
    Y_boxcar = np.zeros((len(df), X.shape[1], 1))

    for i in range(len(df)):
        width = values[quality_to_jiggle[pick_quals[i]]]
        y_new = np.zeros((Y_boxcar.shape[1], 1))

        if T is None:
            # change seconds to samples
            pick_sample = int(Y[i, 0]*(1/dt))
        else:
            pick_sample = T[i]

        start_ind = pick_sample - width // 2
        end_ind = pick_sample + width // 2 + 1
        assert end_ind - start_ind == width, "widths do not match"
        y_new[start_ind:end_ind] = 1
        Y_boxcar[i, :] = y_new

    return Y_boxcar

if __name__ == "__main__":
    # Find the boxcar widths that ben assigned based on the jiggle_weight
    # pref = "/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data"
    # ptraining_file = "%s/resampled_noNGB/trainP.10s.1dup.h5"%pref
    # ptraining_df = pd.read_csv("%s/resampled_noNGB/trainP.10s.1dup.df.csv"%pref)
    #
    # X, Y, T = read_h5file(ptraining_file, num_events=len(ptraining_df))
    # print(X.shape, Y.shape, T.shape)
    # values = find_widths_by_quality(ptraining_df["jiggle_weight"], Y)
    # print(values)
    ############
    # This is the result from above code
    values = {0: 21, 1: 31, 2: 51}

    # Add boxcars to new data & save
    pref = "/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/threeComponent/resampled_10s"
    file = "%s/validateP.10s.1dup.h5"%pref
    df = pd.read_csv("%s/validateP.10s.1dup.df.csv"%pref)
    outfile = "%s/validateP.10s.1dup.h5"%pref

    X, Y, T = read_h5file(file)
    print(X.shape, Y.shape, T.shape)
    Y_boxcar = add_boxcar(df, values, X, Y, T)

    hfout = h5py.File(outfile, 'w')
    hfout.create_dataset('X', data=X)
    hfout.create_dataset('Y', data=Y_boxcar)
    hfout.create_dataset('Pick_index', data=T)
    print(hfout['X'].shape, hfout['Y'].shape, hfout['Pick_index'].shape)
    print(hfout.keys())
    hfout.close()