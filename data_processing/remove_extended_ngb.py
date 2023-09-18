"""
 for split_type in ["train", "test", "validate"]:
   meta_df = pd.read_csv(f"{pref}/*{split_type}.csv")
   with h5py.File(f"{pref}/*{split_type}.h5", "r") as f:
        X = f["X"][:]
        Y = f["Y"][:]
   ngbe_inds = np.where((meta_df.event_lat >= lat_min) & (meta_df.event_lat <= lat_max) &
            (meta_df.event_lon <= lat_min) & (meta_df.event_lon >= lat_max) &
            (meta_df.date <= time_max) & (meta_df.date >= time_min))[0]
   ngbe_df = meta_df.iloc[ngbe_inds]
   ngbe_X = X[ngbe_inds, :, :].copy()
   ngbe_Y = Y[ngbe_inds, :].copy()

   non_ngbe_inds = np.isin(np.arange(X.shape[0]), ngbe_inds, invert=True)
   split_X = X[non_ngbe_inds, :, :].copy()
   split_Y = X[non_ngb_inds, :].copy()

"""