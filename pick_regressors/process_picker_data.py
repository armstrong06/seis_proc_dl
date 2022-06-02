import numpy as np
import sys

def randomize_start_times_and_normalize(X_in, time_series_len=600,
                                        max_dt=0.7, dt=0.01,
                                        n_duplicate=1, random_seed=2412045):
    """
    Uniformly randomize's the start times to within +/- max_dt seconds of
    the trace center and normalizes the amplitudes to [-1,1].

    Parameter
    ---------
    X_in : np.ndarray
       The n_obs x n_samples matrix of observed time series.  By this point
       all observations have been uniformly resampled.
    time_series_len : integer
       This is the desired output length.  This cannot exceed X_in.shape[1].
    max_dt : double
       The maximum time lag in seconds.  For example, 0.5 means the traces
       can be shifted +/- 0.5 seconds about the input trace center.
    dt : double
       The sampling period in seconds.

    Returns
    -------
    X_out : np.ndarray
       The n_obs x time_series_len matrix of signals that were normalized to
       +/- 1 and whose start times were randomizes to +/- max_dt of their
       original start time.
    random_lag : np.array
       The time shift to add to the original pick time to obtain the new
       pick time.
    """
    np.random.seed(random_seed)
    max_shift = int(max_dt / dt)
    n_obs = X_in.shape[0]
    n_samples = X_in.shape[1]
    n_channels = X_in.shape[2]
    print("nObservations,nSamples,nChannels", n_obs, n_samples, n_channels)
    n_distribution = n_samples - time_series_len
    if (n_distribution < 1):
        sys.exit("time_series_len =", time_series_len, "cannot exceed input trace length =", n_samples)
    random_lag = np.random.random_integers(-max_shift, +max_shift, size=n_obs * n_duplicate)
    X_out = np.zeros([len(random_lag), time_series_len, 3], dtype='float')
    ibeg = int(n_samples / 2) - int(time_series_len / 2)  # e.g., 100
    print("Beginning sample to which random lags are added:", ibeg)
    print("Min/max lag:", min(random_lag), max(random_lag))
    for iduplicate in range(n_duplicate):
        for iobs in range(n_obs):
            isrc = iobs
            idst = iduplicate * n_obs + iobs
            # In some respect, the sign doesn't matter.  But in practice, it will
            # conceptually simpler if we add a correction to an initial pick.
            # If the lag is -0.3 (-30 samples), ibeg = 100, and t_pick = 200, then
            # the trace will start at 100 - -30 = 130 samples.  The pick will be at
            # 200 - 130 = 70 samples instead of 100 samples.  In this case, let's
            # assume the pick, t_0, is very late.  The new pick will be corrected
            # by adding the result of the network to the pick - i.e,. t_0 + (-0.3).
            i1 = ibeg - random_lag[idst]  # shift is t - tau
            i2 = i1 + time_series_len

            if X_in.shape[2] == 3:
                # From the detectors, data is ordered ENZ
                en = np.copy(X_in[iobs, i1:i2, 0])
                nn = np.copy(X_in[iobs, i1:i2, 1])
                zn = np.copy(X_in[iobs, i1:i2, 2])
                # norm = max( max(abs(zn)), max(abs(nn)), max(abs(en)) )
                # normi = 1/norm
                # Normalize the components seperatley
                norm_zn = np.max(abs(zn))
                norm_nn = np.max(abs(nn))
                norm_en = np.max(abs(en))
                norm_zni = 1 / norm_zn
                norm_nni = 1 / norm_nn
                norm_eni = 1 / norm_en
                if (norm_nn < 1.e-14):
                    norm_nni = 0
                    print("Division by zero for example:", iobs)
                if (norm_en < 1.e-14):
                    norm_eni = 0
                    print("Division by zero for example:", iobs)
                if (norm_zn < 1.e-14):
                    norm_zni = 0
                    print("Division by zero for example:", iobs)
                # Remember to normalize - switching order to ENZ to be the same as the detectors - less confusing for me
                X_out[idst, :, 0] = en[:] * norm_eni
                X_out[idst, :, 1] = nn[:] * norm_nni
                X_out[idst, :, 2] = zn[:] * norm_zni
            else:
                zn = np.copy(X_in[iobs, i1:i2, 0])
                norm_zn = np.max(abs(zn))
                norm_zni = 1 / norm_zn
                if (norm_zn < 1.e-14):
                    norm_zni = 0
                    print("Division by zero for example:", iobs)
                X_out[idst, :, 0] = zn[:] * norm_zni

    return X_out, random_lag * dt