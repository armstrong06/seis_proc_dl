#!/usr/bin/env python3
# Purpose : Present a strategy for sampling events for multi-event data
#           augmentation assuming a Richter-Gutenberg magnitude distribution.
# Author : Ben Baker, edits by Alysha Armstrong
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os

def plot_sampling_distributions(df, title, bins = [0, 1, 2, 3, 4, 5, 6], min_ml = 0, filter=True, figdir=None):
    """
    :param df:
    :param title:
    :param bins:
    :param min_ml:
    :param filter:
    :param figdir:
    :return:
    """
    # Pick one scale and stick with it
    if filter:
        df_work = df[(df['magnitude_type'] == 'l') & (df['magnitude'] >= min_ml)]
    else:
        df_work = df
    counts, bins_from_hist = np.histogram(df_work.magnitude, bins=bins)
    counts = counts / np.sum(counts)  # Normalize -> this makes each bin a probability
    # This is how you would sample from from this non-uniform distribution
    random_events = np.random.choice(a=bins[0:len(bins) - 1], size=len(df), p=counts)
    # Put magnitude which will be from low bin edge unambiguously into bin by adding 0.5
    counts_r, bins_r = np.histogram(random_events + 0.5, bins=bins)
    print("Verify here that empirical sampling is working.")
    print("The following arrays should have similar values:")
    print("Observed distribution:", counts)
    print("Simulated distribution:", counts_r / np.sum(counts_r))

    plt.hist([df_work.magnitude, random_events], bins, density=True, histtype="step", label=["observed", "simulated"])
    plt.title(title)
    plt.xlabel("Local magnitude")
    plt.ylabel("Probability")
    plt.legend()
    if figdir is not None:
        plt.savefig(figdir)
        plt.close()
    else:
        plt.show()

def match_sampling_distributions(df_tomatch, df_tosample, n_waveforms, title, bins = [0, 1, 2, 3, 4, 5, 6], min_ml = 0):
    """
    Try to match the catalog distribution for a filtered data set
    :param df_tomatch:
    :param df_tosample:
    :param n_waveforms:
    :param title:
    :param bins:
    :param min_ml:
    :return:
    """
    # Pick one scale and stick with it
    df_work = df_tomatch[(df_tomatch['magnitude_type'] == 'l') & (df_tomatch['magnitude'] >= min_ml)]
    counts, bins_from_hist = np.histogram(df_work.magnitude, bins=bins)
    counts = counts / np.sum(counts)  # Normalize -> this makes each bin a probability
    # This is how you would sample from from this non-uniform distribution
    random_events = np.random.choice(a=bins[0:len(bins) - 1], size=len(df_tomatch), p=counts)
    # Put magnitude which will be from low bin edge unambiguously into bin by adding 0.5
    counts_r, bins_r = np.histogram(random_events + 0.5, bins=bins)
    print("Verify here that empirical sampling is working.")
    print("The following arrays should have similar values:")
    print("Observed distribution:", counts)
    print("Simulated distribution:", counts_r / np.sum(counts_r))

    all_rows = []
    for i in range(len(bins)-1):
        df_mag_range = df_tosample.loc[(df_tosample["magnitude"] >= bins[i]) & (df_tosample["magnitude"] < bins[i+1])]
        mag_range_inds = df_mag_range.index.values
        random_rows = np.random.randint(low = 0, high = len(mag_range_inds), size = round(n_waveforms*counts[i]))
        all_rows.append(mag_range_inds[random_rows[:]][:])

    all_rows = np.concatenate(all_rows)
    if len(all_rows)%2:
        all_rows = all_rows[:-1]

    df_sampled = df_tosample.loc[all_rows]
    df_work_tosample = df_tosample[(df_tosample['magnitude_type'] == 'l') & (df_tosample['magnitude'] >= min_ml)]
    df_work_sampled = df_sampled[(df_sampled['magnitude_type'] == 'l') & (df_sampled['magnitude'] >= min_ml)]
    plt.hist([df_work_tosample.magnitude, df_work_sampled.magnitude, df_sampled.magnitude, df_work.magnitude], bins, density=True,
             histtype="step",  label=["partial catalog", "match whole catalog - ml", "match whole catalog - all", "whole catalog"],
             color=["black", "green", "red", "blue"])
    plt.title(title)
    plt.xlabel("Local magnitude")
    plt.ylabel("Probability")
    plt.legend()
    plt.show()

    # The rows are currently group by magnitude bin, shuffle them to get more variety in mags of event pairs
    # Commenting this out because I decided to add a constraint on the first event as well - not exactly
    # the same as the training constraint here, but at least events will be closer together
    # shuffle_inds = np.arange(len(df_sampled))
    # np.random.shuffle(shuffle_inds)
    # df_sampled = df_sampled.iloc[shuffle_inds]

    # Make sure M2 >= M1-0.5 by flipping event order if req not met
    new_inds = []
    for ind in range(0, len(df_sampled), 2):
        first_ev = ind
        second_ev = ind+1
        if df_sampled.iloc[ind + 1].magnitude < df_sampled.iloc[ind].magnitude-0.5:
            first_ev = ind+1
            second_ev = ind
        new_inds.append(first_ev)
        new_inds.append(second_ev)

    df_sampled = df_sampled.iloc[new_inds]

    return df_sampled

def generate_sampling_distribution(df,
                                   n_waveforms = 5000,
                                   bins = [0, 1, 2, 3, 4, 5, 6],
                                   min_ml = 0,
                                   z_indicator="channelz", plot_dist=False):
    """
    Example of how to generate a non-uniform magnitude sampling distribution based on 
    the observed local magnitude frequency distribution in the catalog - Ben wrote most of this
    """
    # Pick one scale and stick with it
    df_work = df[ (df['magnitude_type'] == 'l') & (df['magnitude'] >= min_ml) ]
    counts, bins_from_hist = np.histogram(df_work.magnitude, bins = bins)
    counts = counts/np.sum(counts) # Normalize -> this makes each bin a probability
    # This is how you would sample from from this non-uniform distribution
    random_events = np.random.choice(a = bins[0:len(bins)-1], size = len(df), p = counts)
    # Put magnitude which will be from low bin edge unambiguously into bin by adding 0.5
    counts_r, bins_r = np.histogram(random_events + 0.5, bins = bins) 
    print("Verify here that empirical sampling is working.")
    print("The following arrays should have similar values:")
    print("Observed distribution:", counts)
    print("Simulated distribution:", counts_r/np.sum(counts_r))
    # Generate a list of random waveforms to which a duplicate event will be assigned
    # Naturally, our sampling will reflect the implicit magnitude distribution.
    df_pairs = pd.DataFrame(columns = df.columns)
    random_rows = np.random.randint(low = 0, high = len(df), size = n_waveforms)
    for random_row in random_rows:
        df_row = df.iloc[random_row]
        evid = df_row.evid
        net = df_row.network
        sta = df_row.station
        cha = df_row[z_indicator]
        loc = df_row.location
        mag = df_row.magnitude

        # Get a subset of matching events (but don't let me match to myself - hence the check on evid)
        df_subset = df[ (df.evid != evid) & (df.network == net) & (df.station == sta) & (df[z_indicator] == cha) &
                        (df.location == loc) & (df.magnitude >= mag-0.5) & (df.magnitude <= mag+1.5)]

        if (len(df_subset) < 1):
            continue
        # Find a pair
        other_random_row = np.random.randint(low = 0, high = len(df_subset), size = 1)
        df_pairs = df_pairs.append(df.iloc[random_row, :], ignore_index = True)
        df_pairs = df_pairs.append(df_subset.iloc[other_random_row, :], ignore_index = True)
    print(df_pairs.head(20))
    df_work = df_pairs[ (df_pairs['magnitude_type'] == 'l') & (df_pairs['magnitude'] >= min_ml) ]
    counts_work, bins_work = np.histogram(df_work.magnitude, bins = bins)
    counts_work = counts_work/np.sum(counts_work)
    print("Magnitude distribution in sampled catalog")
    print(counts_work)

    if plot_dist:
        plt.hist(df_work.magnitude, bins, density=True, histtype="step")
        plt.title("Magnitude distribution in sampled catalog")
        plt.xlabel("Local magnitude")
        plt.ylabel("Probability")
        plt.show()

    return df_pairs, counts_work

def clean_df(df, join_mags=None, duplicate_evids=None, pick_indicator="pick_time", quality_req=None, sr_dist_max=None, min_mag=0):
    """
    Remove events from the dataframe that are within min_sep seconds of each other. Adds a new
    column to the dataframe called xind, which is the corresponding waveform index in the h5 files
    :param df: original dataframe
    :param duplicate_evids: list of evids to remove - I thought just removing close events would get all the duplicate
                            events but there must be some stations that only have picks for one event and so the evid
                            remains
    :param pick_indicator: column name for the arrival time
    :return:
    """
    # Do this before any filtering to ensure indices correspond!!!
    df["xind"] = df.index

    print(df.shape)
    if duplicate_evids is not None:
        df = df[np.isin(df.evid, duplicate_evids.evid, invert=True)]
    print(df.shape)

    if quality_req is not None:
        print("Applying quality requirment...")
        print(df.shape)
        if np.isin("pick_quality", df.columns):
            df = df[np.isin(df["pick_quality"], quality_req)]
        else:
            df = df[np.isin(df["jiggle_weight"], quality_req)]
        print(df.shape)

    if sr_dist_max is not None:
        print("Applying source-receiver distance requirment...")
        print(df.shape)
        df = df[df["source_receiver_distance"] < sr_dist_max]
        print(df.shape)


    if join_mags is not None:
        print("Adding magnitude data")
        mag_cat = join_mags[["evid", "magnitude", "magnitude_type"]].drop_duplicates().merge(df['evid'].drop_duplicates(), how="right")
        df = df.merge(mag_cat, on="evid")
        print(df.shape)

    def remove_close_events(stat_df, min_sep=2, pick_indicator="pick_time"):
        stat_df = stat_df.sort_values(pick_indicator)
        stat_df["sind"] = np.arange(0, len(stat_df))
        shift = stat_df[pick_indicator].shift(-1)
        match_inds = stat_df[(shift - stat_df[pick_indicator] < min_sep)]["sind"].values
        match_inds = np.unique(np.concatenate([match_inds, match_inds + 1]))
        match_df = stat_df[np.isin(stat_df["sind"],match_inds, invert=True)].sort_values("sind")

        return match_df

    print("Removing events with magnitude less than set threshold...")
    df = df[df["magnitude"] > min_mag]
    print(df.shape)
    print("Removing picks that have another pick within 8 seconds of it...")
    new_df = []
    for stat in df["station"].unique():
        stat_df = df[df["station"] == stat]
        filter_stat_df = remove_close_events(stat_df, min_sep=8, pick_indicator=pick_indicator)
        new_df.append(filter_stat_df)

    new_df = pd.concat(new_df).drop("sind", axis=1)
    #new_df["xind"] = new_df.index
    print(new_df.shape)
    return new_df

def read_h5file(file):
    h5 = h5py.File(file, "r")
    X = h5["X"][:]
    Y = h5["Y"][:]
    T = h5["Pick_index"][:]
    h5.close()

    return (X, Y, T)

def write_h5file(X, Y, T, outfile_pref):
    outfile = "%s_synthetic_multievent_waveforms.h5"%outfile_pref
    hfout = h5py.File(outfile, 'w')
    hfout.create_dataset('X', data=X)
    hfout.create_dataset('Y', data=Y)
    hfout.create_dataset('Pick_index', data=T[:, 0])
    hfout.create_dataset('Pick_index2', data=T[:, 1])
    print(hfout['X'].shape, hfout['Y'].shape, hfout["Pick_index"].shape, hfout["Pick_index2"].shape)
    hfout.close()
    print("data saved in", outfile)

def plot_combied_waveforms(Xdata, Y, Tdata, metadata, figdir):
    # 0 - E/2, 1-N/1
    comp_names = {0: "E", 1:"N", 2:"Z"}
    for i in range(3):
        fig, axes = plt.subplots(4, figsize=(6, 7))
        plt.suptitle(
            "%s - %s channel - %s (%s %s), %s(%s %s)" % (
            metadata["station"], comp_names[i], metadata["evid1"], metadata["mag1"], metadata["mag1_type"],
            metadata["evid2"], metadata["mag2"], metadata["mag2_type"]))

        if metadata["mag1"] < metadata["mag2"]:
            axes[0].plot(range(Xdata[0].shape[0]), Xdata[4][:, i], label="tr1-original", color="black")
            axes[0].plot(range(Xdata[0].shape[0]), Xdata[0][:, i], label="tr1-rescaled by %0.2f"%metadata["rescale_factor"], color="red")
            axes[1].plot(range(Xdata[0].shape[0]), Xdata[1][:, i], label="tr2", color="blue")
        else:
            axes[1].plot(range(Xdata[0].shape[0]), Xdata[4][:, i], label="tr2-original", color="black")
            axes[1].plot(range(Xdata[0].shape[0]), Xdata[1][:, i], label="tr2-rescaled by %0.2f"%metadata["rescale_factor"], color="blue")
            axes[0].plot(range(Xdata[0].shape[0]), Xdata[0][:, i], label="tr1", color="red")

        axes[0].axvline(Tdata[0], color="red", linestyle="--")
        axes[1].axvline(Tdata[1], color="blue", linestyle="--")
        axes[2].plot(range(Xdata[0].shape[0]), Xdata[2][:, i], label="tr2-shifted", color="blue")
        axes[2].axvline(Tdata[2], color="blue", linestyle="--")
        axes[3].plot(range(Xdata[0].shape[0]), Xdata[3][:, i], label="tr1+tr2-shifted", color="purple")
        axes[3].axvline(Tdata[0], color="red", linestyle="--")
        axes[3].axvline(Tdata[2], color="blue", linestyle="--")
        ax3_twin = axes[3].twinx()
        ax3_twin.set_ylabel("Probability")
        ax3_twin.plot(range(Xdata[0].shape[0]), Y, color="gray", label="Y")
        ax3_twin.set_ylim([0, 1])
        ax3_twin.legend(loc=3)
        for ax in axes.flatten():
            ax.legend(loc=2)

        if figdir is not None:
            plt.savefig("%s/%s_%s_%s_%s.jpg"%(figdir, metadata["station"], metadata["evid1"], metadata["evid2"],
                                              comp_names[i]))
            plt.close()
        else:
            plt.show()

def get_max_boxcar_width(Y):
    boxcar_widths = np.zeros(Y.shape[0], dtype="int")
    for i in range(Y.shape[0]):
        boxcar_widths[i] = len(np.where(Y[i, :, :] > 0)[0])

    return np.max(boxcar_widths)

def rescale_waveform(x_large, x_small, mag_diff, to_plot=False, normalize_separate=True):
    assert mag_diff >= 0, "Magnitude difference must be non-negative"
    #print("magnitude difference", mag_diff)
    # TODO: Check if I need to rescale each trace individual or if it already does that
    # rescale value is very similar between components now that the input traces were normalized separate. But adds in
    # potential divide by 0 errors => removing
    # if normalize_separate:
    #     m1_max = np.max(abs(x_large), axis=0)
    #     m2_max = np.max(abs(x_small), axis=0)
    # else:
    m1_max = np.max(abs(x_large))
    m2_max = np.max(abs(x_small))
    #print("M1 max amplitude", m1_max)
    #print("M2 max amplituide", m2_max)
    new_max = m1_max * 1 / 10 ** (mag_diff)
    #print("reduce M2 amplitude by", 10 ** mag_diff)
    #print("new max of m2 should be", new_max)
    rescale_factor = m2_max / new_max
    m2_new = x_small / rescale_factor
    #print("new max is", np.max(abs(m2_new)))
    if to_plot:
        fig, axes = plt.subplots(3)
        axes[0].plot(range(x_small.shape[0]), x_small[:, 0])
        axes[0].plot(range(x_small.shape[0]), m2_new[:, 0])
        axes[1].plot(range(x_small.shape[0]), x_small[:, 1])
        axes[1].plot(range(x_small.shape[0]), m2_new[:, 1])
        axes[2].plot(range(x_small.shape[0]), x_small[:, 2])
        axes[2].plot(range(x_small.shape[0]), m2_new[:, 2])
        plt.suptitle(("Magnitude difference: %0.2f - Reduced by: %0.2f"%(mag_diff,rescale_factor)))
        plt.show()
        #plt.close()
    #print("new max with other equation", np.max(abs(x_small * m1_max / (m2_max * 10 ** mag_diff))))
    #print("calculate magnitude difference by log10(amplitude ratio)", np.log10(m1_max / np.max(abs(m2_new))))
    return m2_new, x_small, rescale_factor

def combine_waveforms(pairs_df, waveform_tuple, outfile_pref, min_sep=300, end_buffer=100, to_plot=False, figdir=None,
                      to_scale=True, normalize_separate=True):
    X = waveform_tuple[0]
    Y = waveform_tuple[1]
    T = waveform_tuple[2]

    boxcar_widths = get_max_boxcar_width(Y)
    new_catalog = []
    comb_df = [["evid1", "evid2", "network", "station", "pick_index1", "shift", "pick_index2", "magnitude1",
               "magnitude1_type", "magnitude2", "magnitude2_type", "reduction_factor", "pre_p_samples"]]
    Y_aug = np.zeros((len(pairs_df)//2, Y.shape[1], 1))
    X_aug = np.zeros((len(pairs_df)//2, X.shape[1], X.shape[2]))
    T_aug = np.zeros((len(pairs_df)//2, 2))
    cnt = 0
    for index in np.arange(0, len(pairs_df)-1, 2):
        row1 = pairs_df.iloc[index]
        row2 = pairs_df.iloc[index+1]
        ind1 = row1["xind"]
        ind2 = row2["xind"]

        metadata = {
            "evid1": row1["evid"],
            "evid2": row2["evid"],
            "network": row1["network"],
            "station": row1["station"],
            "mag1": row1["magnitude"],
            "mag2": row2["magnitude"],
            "mag1_type": row1["magnitude_type"],
            "mag2_type": row2["magnitude_type"]
        }

        x1 = X[ind1, :, :].copy()
        x2 = X[ind2, :, :].copy()

        # TODO: should I demean? They already seem to be pretty small
        x1 = x1 - np.mean(x1, axis=0)
        x2 = x2 - np.mean(x2, axis=0)

        if to_scale:
            if metadata["mag1"] > metadata["mag2"]:
                x2, unscaled, rescale_factor = rescale_waveform(x1, x2, metadata["mag1"]-metadata["mag2"])
            else:
                x1, unscaled, rescale_factor = rescale_waveform(x2, x1, metadata["mag2"]-metadata["mag1"])
        else:
            # TODO: here
            if metadata["mag1"] > metadata["mag2"]:
                unscaled = x2
            else:
                unscaled= x1
            rescale_factor = 1

        # if len(rescale_factor) > 1:
        #     metadata["rescale_factorE"] = rescale_factor[0]
        #     metadata["rescale_factorN"] = rescale_factor[1]
        #     metadata["rescale_factorZ"] = rescale_factor[2]
        # else:
        #     metadata["rescale_factorE"] = rescale_factor
        #     metadata["rescale_factorN"] = rescale_factor
        #     metadata["rescale_factorZ"] = rescale_factor
        metadata["rescale_factor"] = rescale_factor

        pre_p = boxcar_widths//2 + 5  # samples - 25 is the max halfwidth of a boxcar
        #x2_shift = np.zeros((X.shape[1], X.shape[2]))
        if np.all(abs(x2[T[ind2]-pre_p] - 0) < 1e-1):
            fill_val = x2[T[ind2]-pre_p, :]
        else:
            tmp = np.unique(np.where(np.isclose(x2[:T[ind2] - pre_p + 1, :], 0, atol=1e-1)), return_counts=True)
            tmp_ind = tmp[0][np.where(tmp[1] == 3)]
            if len(tmp_ind) > 0:
                tmp_ind = tmp_ind[-1]
            else:
                print("No reasonable pre_p sample")
                continue
            #assert (T[ind2]-pre_p) - tmp_ind < 50, "Including too much information before 2nd P arrival"
            if (T[ind2]-pre_p) - tmp_ind > 50:
                print("Moved pre_p too far!")
                continue
            else:
                fill_val = x2[tmp_ind, :]
                pre_p = T[ind2] - tmp_ind

        #x2_shift = np.full((X.shape[1], X.shape[2]), fill_val)
        x2_shift = np.zeros((X.shape[1], X.shape[2]))

        # check that there is enough room for the second waveform
        if T[ind1] + min_sep > X.shape[1] - end_buffer - pre_p:
            #print("Can't do that")
            continue
        # second waveform is already far enough from the first waveform, don't do any shifting
        elif T[ind2] - T[ind1] > min_sep:
            #print("here")
            shift = 0
            x2_shift[T[ind2] - pre_p:, :] = x2[T[ind2] - pre_p:, :]
        # second waveform is too close to the first, shift it
        else:
            # find the maximum possible separation based on the where the first pick is & the end_buffer requirement
            max_sep = X.shape[1] - T[ind1] - end_buffer
            sep = np.random.randint(min_sep, max_sep)
            # calculate the shift for the Pick_index
            shift = T[ind1] + sep - T[ind2]
            assert shift > 0, "Shift is negative"
            # which sample the shifted 2nd waveform should start at
            start_sample_shift = (T[ind1] + sep - pre_p)
            # where to cut the second waveform
            start_sample_2 = T[ind2] - pre_p
            # where to end the second waveform so the shift array has X.shape[1] samples
            end_sample = X.shape[1] - start_sample_shift + start_sample_2
            assert end_sample - start_sample_2 > end_buffer + pre_p, "2 waveform too small"
            # shift waveform and 0 before the pick-pre_p
            x2_shift[start_sample_shift:, :] = x2[start_sample_2:end_sample, :]

        # remove the mean from x2_shift since the beginning of waveform is a continuation of the first sample from cut signal and not zero
        #print(np.mean(x2_shift, axis=0))
        #print(np.mean(x2_shift[:, 0]), np.mean(x2_shift[:, 1]), np.mean(x2_shift[:, 2]))
        x2_shift = x2_shift - np.mean(x2_shift, axis=0)
        halfwidth = len(np.where(Y[ind2, :] > 0)[0]) // 2
        assert halfwidth < 20, "Boxcar should not be this wide for quality reqs"
        y_new = Y[ind1, :].copy()
        y_new[T[ind2] + shift - halfwidth:T[ind2] + shift + halfwidth + 1] = 1
        assert len(np.arange(T[ind2] + shift - halfwidth, T[ind2] + shift + halfwidth + 1)) == len(np.where(Y[ind2, :] > 0)[0])

        new_catalog.append(row1)
        new_catalog.append(row2)

        comb_df.append([metadata["evid1"],metadata["evid2"], metadata["network"],
                                              metadata["station"], T[ind1], shift, T[ind2] + shift, metadata["mag1"],
                                              metadata["mag1_type"], metadata["mag2"], metadata["mag2_type"],
                                              metadata["rescale_factor"], pre_p])
        x_new = x1 + x2_shift
        
        # normalize
        # TODO: Rescale each trace individually
        #x_new = x_new/np.max(abs(x_new))
        # min/max rescale
        if normalize_separate:
            X_normalizer = np.amax(np.abs(x_new), axis=0)
            assert X_normalizer.shape[0] is X.shape[2], "Normalizer is wrong shape for selected norm type"
        else:
            X_normalizer = np.amax(np.abs(x_new))
            assert X_normalizer.shape[0] is 1, "Normalizer is wrong shape for selected norm type"

        for norm_ind in range(len(X_normalizer)):
            if X_normalizer[norm_ind] != 0:
                x_new[:, norm_ind] = x_new[:, norm_ind] / X_normalizer[norm_ind]
            else:
                x_new[:, norm_ind] = x_new[:, norm_ind]

        assert np.max(abs(x_new)) <= 1

        #print(np.max(abs(x_new)))
        Y_aug[cnt, :] = y_new
        X_aug[cnt, :, :] = x_new
        T_aug[cnt, 0] = T[ind1]
        T_aug[cnt, 1] = T[ind2] + shift
        cnt += 1
        if to_plot and cnt % 100 == 0:
            plot_combied_waveforms([x1, x2, x2_shift, x_new, unscaled], y_new, [T[ind1], T[ind2], T[ind2]+shift], metadata, figdir)

    print(X_aug.shape, Y_aug.shape, T_aug.shape, cnt)
    X_aug = X_aug[:cnt, :, :]
    Y_aug = Y_aug[:cnt, :, :]
    T_aug = T_aug[:cnt, :]
    print(X_aug.shape, Y_aug.shape, T_aug.shape, cnt)
    write_h5file(X_aug, Y_aug, T_aug, outfile_pref)
    pd.DataFrame(new_catalog).to_csv("%s_synthetic_multievent_catalog.df.csv"%outfile_pref, index=False)
    pd.DataFrame(comb_df[1:], columns=comb_df[0]).to_csv("%s_synthetic_multievent_summary_info.df.csv"%outfile_pref, index=False)

if __name__ == "__main__":
    pref_entire_cat = "/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/uuss2021"
    duplicate_pref = "/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/data"
    pref = "/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/uuss2021/p_resampled_10s"
    outpref = "%s/synthetic_multievent_waveforms"%pref
    outfile_pref = "%s/trainP.10s.1dup"%outpref
    figdir = "%s/train_figs"%outpref

    if not os.path.exists(outpref):
        os.makedirs(outpref)

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    # Read in entire 3C catalog
    entire_cat_df = pd.read_csv("%s/P_current_earthquake_catalog.csv"%pref_entire_cat)
    entire_cat_df = entire_cat_df[entire_cat_df["phase"] == "P"]

    # Read in dataset for plotting
    df = pd.read_csv("%s/currenteq.train.10s.1dup.csv"%pref, dtype={'location'  : object})
    df = df[df.phase == "P"]
    waveform_tuple = read_h5file("%s/currenteq.train.10s.1dup.h5"%pref)

    # Read in potential duplicate evids
    duplicates_df = pd.read_csv("%s/possibleDuplicates_currentEarthquakeArrivalInformation3C.csv"%duplicate_pref, dtype={'location'  : object})
    duplicated_df = duplicates_df["evid"].unique()

    # Clean up the dataset for plotting
    # df_clean = clean_df(df, duplicate_evids=duplicates_df, pick_indicator="pick_time",
    #                     join_mags=entire_cat_df[["evid", "magnitude", "magnitude_type"]], quality_req=[0, 1], sr_dist_max=40)
    df_clean = clean_df(df, duplicate_evids=duplicates_df, pick_indicator="arrival_time",
                        join_mags=None, quality_req=[1.0, 0.75], sr_dist_max=40)

    assert np.all(np.isin(df_clean.evid, duplicates_df.evid, invert=True)), "Duplicate events still here"

    ################ Training ################
    plot_sampling_distributions(df_clean, "Magnitude distribution of filtered training data",
                              figdir="%s/filtered_training_dist.jpg"%figdir)

    plot_sampling_distributions(entire_cat_df, "Magnitude distribution of entire 3C catalog",
                                figdir="%s/3C_catalog_dist.jpg"%figdir)
    df_pairs, sampled_dist = generate_sampling_distribution(df_clean, n_waveforms=40000, z_indicator="channelz")
    plot_sampling_distributions(df_pairs, "Magnitude distribution of sampled training catalog",
                                figdir="%s/sampled_filtered_training_dist.jpg"%figdir)
    # combine the waveforms
    combine_waveforms(df_pairs, waveform_tuple, outfile_pref, to_plot=True, figdir=figdir)

    ################ Validation ################
    # # Only use this for validation/test data because much smaller size
    # df_match_dist = match_sampling_distributions(entire_cat_df, df_clean, 15000, "Magnitude distribution of different data sets")
    # #Plot distributions and make dataframe of events to combine
    # plot_sampling_distributions(df, "Magnitude distribution of validation data", figdir="%s/original_validation_dist.jpg"%figdir)
    # plot_sampling_distributions(df_match_dist, "Mag dist of filtered validation data made to match whole catalog dist",
    #                             figdir="%s/filtered_validation_dist.jpg"%figdir)
    # plot_sampling_distributions(df_match_dist, "Mag dist of filtered validation data made to match whole catalog dist - all mag types",
    #                             filter=False, figdir="%s/filtered_validation_dist_allmagtypes.jpg"%figdir)
    # plot_sampling_distributions(entire_cat_df, "Magnitude distribution of entire 3C catalog",
    #                             figdir="%s/3C_catalog_dist.jpg"%figdir)
    # #combine the waveforms
    # combine_waveforms(df_match_dist, waveform_tuple, outfile_pref, to_plot=True, figdir=figdir)
