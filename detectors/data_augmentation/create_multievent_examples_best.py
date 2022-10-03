#!/usr/bin/env python3
# Purpose : Present a strategy for sampling events for multi-event data
#           augmentation assuming a Richter-Gutenberg magnitude distribution.
# Author : Ben Baker, edits by Alysha Armstrong
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import os
import obspy

class CreateMEW():

    def __init__(self, wf_length, center_window_size, boxcar_width, min_wf2=150, min_sep_req=300, wf1_ub_buffer=100,
                 wf2_pre_p = 10, taper_max_length=0.09):
        """

        :param wf_length: The output size (in samples) of the waveforms
        :param center_window_size: The number of samples on either side of the center of the waveform a pick should be
        :param boxcar_width: Number of samples the target should be on either side of the pick
        :param min_wf2: Minimum length (in samples) of the second waveform
        :param min_sep_req: Minimum separation requirement for all pairs, may be larger depending on SR distance of wf1
        :param wf1_ub_buffer: Buffer between the first event arrival and the upper edge of the center window
        :param wf2_pre_p: How many samples before the second event P to trim the waveform before tapering
        :param taper_max_length: Maximum length (in seconds) of the left taper applied to wf2
        """

        self.wf_length = wf_length
        self.center_window_size = center_window_size
        self.boxcar_width = boxcar_width
        self.min_wf2 = min_wf2
        self.min_sep_req = min_sep_req
        self.wf1_ub_buffer = wf1_ub_buffer
        self.wf2_pre_p = wf2_pre_p

        self.center_window_ub = wf_length // 2 + center_window_size
        self.center_window_lb = wf_length // 2 - center_window_size

        self.taper_max_length = taper_max_length


    @staticmethod
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

    @staticmethod
    def match_sampling_distributions(df_tomatch, df_tosample, n_waveforms, title,
                                     bins = [0, 1, 2, 3, 4, 5, 6], min_ml = 0,
                                     z_indicator="channelz", min_sep=0.5, max_sep=1.5):
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
        # Get magnitude probabilty distribution of the entire catalog
        df_work = df_tomatch[(df_tomatch['magnitude_type'] == 'l') & (df_tomatch['magnitude'] >= min_ml)]
        counts, bins_from_hist = np.histogram(df_work.magnitude, bins=bins)
        counts = counts / np.sum(counts)  # Normalize -> this makes each bin a probability

        # Sample from the smaller catalog using the probability distribution of the entire catalog
        all_rows = []
        for i in range(len(bins)-1):
            df_mag_range = df_tosample.loc[(df_tosample["magnitude"] >= bins[i]) & (df_tosample["magnitude"] < bins[i+1])]
            mag_range_inds = df_mag_range.index.values
            random_rows = np.random.randint(low = 0, high = len(mag_range_inds), size = round(n_waveforms*counts[i]))
            all_rows.append(mag_range_inds[random_rows[:]][:])
        all_rows = np.concatenate(all_rows)
        df_sampled = df_tosample.loc[all_rows]

        # Plot comparision of various magnitude distributions
        df_work= df_tomatch[(df_tomatch['magnitude_type'] == 'l') & (df_tomatch['magnitude'] >= min_ml)]
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

        # Find a second event for each sampled event - shouldn't mess with the distribution too much because primarily sampling from
        # within the same magnitude bin - maybe a little outside
        df_pairs = pd.DataFrame(columns=df.columns)
        for i, df_row in df_sampled.iterrows():
            evid = df_row.evid
            net = df_row.network
            sta = df_row.station
            cha = df_row[z_indicator]
            loc = df_row.location
            mag = df_row.magnitude

            # Get a subset of matching events (but don't let me match to myself - hence the check on evid)
            df_subset = df_tosample[ (df_tosample.evid != evid) & (df_tosample.network == net) & (df_tosample.station == sta) & (df_tosample[z_indicator] == cha) &
                            (df_tosample.location == loc) & (df_tosample.magnitude >= mag-min_sep) & (df_tosample.magnitude <= mag+max_sep)]

            if (len(df_subset) < 1):
                continue
            # Find a pair
            other_random_row = np.random.randint(low = 0, high = len(df_subset), size = 1)
            df_pairs = df_pairs.append(df_row, ignore_index = True)
            df_pairs = df_pairs.append(df_subset.iloc[other_random_row, :], ignore_index = True)

        df_work = df_pairs[ (df_pairs['magnitude_type'] == 'l') & (df_pairs['magnitude'] >= min_ml) ]
        counts_work, bins_work = np.histogram(df_work.magnitude, bins = bins)
        counts_work = counts_work/np.sum(counts_work)
        print("Magnitude distribution in sampled catalog")
        print(counts_work)

        return df_pairs

    @staticmethod
    def generate_sampling_distribution(df,
                                       n_waveforms = 5000,
                                       bins = [0, 1, 2, 3, 4, 5, 6],
                                       min_ml = 0,
                                       z_indicator="channelz", plot_dist=False, min_sep=0.5, max_sep=1.5):
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
                            (df.location == loc) & (df.magnitude >= mag-min_sep) & (df.magnitude <= mag+max_sep)]

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

    @staticmethod
    def clean_df(df, pick_indicator="pick_time", quality_req=None, sr_dist_max=None, min_mag=0):
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

        if quality_req is not None:
            print(f"Keep only {quality_req} quality picks...")
            print(df.shape)
            if np.isin("pick_quality", df.columns):
                df = df[np.isin(df["pick_quality"], quality_req)]
            else:
                df = df[np.isin(df["jiggle_weight"], quality_req)]
            print(df.shape)

        if sr_dist_max is not None:
            print(f"Applying source-receiver distance requirement of {sr_dist_max}")
            print(df.shape)
            df = df[df["source_receiver_distance"] < sr_dist_max]
            print(df.shape)

        def remove_close_events(stat_df, min_sep=2, pick_indicator="pick_time"):
            stat_df = stat_df.sort_values(pick_indicator)
            stat_df["sind"] = np.arange(0, len(stat_df))
            shift = stat_df[pick_indicator].shift(-1)
            match_inds = stat_df[(shift - stat_df[pick_indicator] < min_sep)]["sind"].values
            match_inds = np.unique(np.concatenate([match_inds, match_inds + 1]))
            match_df = stat_df[np.isin(stat_df["sind"],match_inds, invert=True)].sort_values("sind")
            close_event_df = stat_df[np.isin(stat_df["sind"],match_inds, invert=False)].sort_values("sind")
            return match_df, close_event_df

        print(f"Removing events with magnitude less than {min_mag}..")
        df = df[df["magnitude"] > min_mag]
        print(df.shape)

        print("Removing picks that have another pick within 8 seconds of it...")
        new_df = []
        close_event_df = []
        for stat in df["station"].unique():
            stat_df = df[df["station"] == stat]
            filter_stat_df, station_close_event_df = remove_close_events(stat_df, min_sep=8, pick_indicator=pick_indicator)
            new_df.append(filter_stat_df)
            close_event_df.append(station_close_event_df)
        new_df = pd.concat(new_df).drop("sind", axis=1)
        close_event_df = pd.concat(close_event_df).drop("sind", axis=1)

        print(new_df.shape)
        print("Catalog close events:", close_event_df.shape)
        print("Catalog close events - no arrival time duplicates",
              close_event_df.drop_duplicates(subset=["station", "arrival_time"]).shape)
        return new_df, close_event_df

    @staticmethod
    def read_h5file(file):
        h5 = h5py.File(file, "r")
        X = h5["X"][:]
        # Y = h5["Y"][:]
        # T = h5["Pick_index"][:]
        h5.close()

        if len(X.shape) < 3:
            X = np.expand_dims(X, axis=2)
        return (X)

    @staticmethod
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

    @staticmethod
    def rescale_waveform(x_large, x_small, mag_diff, to_plot=False):
        assert mag_diff >= 0, "Magnitude difference must be non-negative"

        large_max = np.max(abs(x_large))
        small_max = np.max(abs(x_small))

        # print("Large event max amplitude", large_max)
        # print("Small event max amplituide", small_max)
        new_small_max = large_max * 1 / 10 ** (mag_diff)
        # print("reduce small event amplitude by", 10 ** mag_diff)
        # print("new max of small event should be", new_small_max)
        rescale_factor = small_max / new_small_max
        x_small_rescaled = x_small / rescale_factor
        # print("new max is", np.max(abs(x_small_rescaled)))
        if to_plot:
            fig, axes = plt.subplots(3)
            axes[0].plot(range(x_small.shape[0]), x_small[:, 0])
            axes[0].plot(range(x_small.shape[0]), x_small_rescaled[:, 0])
            axes[1].plot(range(x_small.shape[0]), x_small[:, 1])
            axes[1].plot(range(x_small.shape[0]), x_small_rescaled[:, 1])
            axes[2].plot(range(x_small.shape[0]), x_small[:, 2])
            axes[2].plot(range(x_small.shape[0]), x_small_rescaled[:, 2])
            plt.suptitle((f"Magnitude difference: {mag_diff:0.2f} - Reduced by: {rescale_factor}"))
            plt.show()

        return x_small_rescaled, x_small, rescale_factor

    def get_random_shifts(self, sr_dist):
         ##### Randomly shift the waveforms
         # If the SR-based min_sep is too big, reduce the size
        min_sep_ub = np.min([int(sr_dist / 6 * 100) + 50, self.wf_length-self.min_wf2-self.center_window_lb-50])
        min_sep = np.max([self.min_sep_req, min_sep_ub])

        max_first_event_sample = np.min([(self.wf_length - self.min_wf2) - min_sep ,
                                         self.center_window_ub - self.wf1_ub_buffer])
        first_event_sample = np.random.randint(self.center_window_lb, max_first_event_sample)

        # Keep the second event boxcar out of the center for easier testing
        min_second_event_sample = np.max([self.center_window_ub + self.boxcar_width, first_event_sample + min_sep])
        max_second_event_sample = self.wf_length - self.min_wf2
        second_event_sample = np.random.randint(min_second_event_sample, max_second_event_sample)

        # Shift for moving the second pick into the center and the first in front of the center
        # Keep the first event boxcar out of the center for easier testing
        random_shift_lb = np.max([first_event_sample - (self.center_window_lb - self.boxcar_width - 1),
                                   second_event_sample - (self.center_window_ub - 1)])
        random_shift_ub = np.min([first_event_sample - self.boxcar_width - 1, second_event_sample - self.center_window_lb])
        random_shift = np.random.randint(random_shift_lb, random_shift_ub)

        assert (first_event_sample >= self.center_window_lb and first_event_sample <= self.center_window_ub) and \
               second_event_sample > self.center_window_ub, "WF1 should be the only one in the center"
        assert first_event_sample - random_shift < self.center_window_lb and \
               (second_event_sample - random_shift >= self.center_window_lb and \
                second_event_sample - random_shift <= self.center_window_ub), "WF2 should be the only one in the center"

        return first_event_sample, second_event_sample, random_shift

    def process_waveform_pair(self, wf1, wf2, mag_diff):
        """
        Normalize and rescale waveform par
        :param wf1: First waveform time-series
        :param wf2: Second waveform time-series
        :param mag_diff: the magnitude difference of wf1-wf2
        :return: normalized wf1, normalized wf2, rescale factor
        """
        # Normalize the traces together so relative information is preserved
        wf1_max = np.max(abs(wf1))
        wf1_norm = wf1 / wf1_max

        wf2_max = np.max(abs(wf2))
        wf2_norm = wf2 / wf2_max

        # Remove the mean from each trace
        wf1_norm = wf1_norm - np.mean(wf1_norm, axis=0)
        wf2_norm = wf2_norm - np.mean(wf2_norm, axis=0)

        rescale_factor = 1.0
        # Rescale the magnitudes by comparing the maximum amplitude of the waveform
        if mag_diff > 0:
            wf2_norm, x_small_original, rescale_factor = self.rescale_waveform(wf1_norm, wf2_norm, mag_diff, to_plot=False)
        elif mag_diff < 0:
            wf1_norm, x_small_original, rescale_factor = self.rescale_waveform(wf2_norm, wf1_norm, abs(mag_diff), to_plot=False)

        return wf1_norm, wf2_norm, rescale_factor

    def join_waveform_pairs(self, wf1_norm, wf2_norm, first_event_sample, second_event_sample, random_shift, plot_info=None):
        """

        :param self:
        :param wf1_norm: normalized first waveform
        :param wf2_norm: normalized second waveform
        :param first_event_sample: sample of first P-arrival when it is in center window
        :param second_event_sample: sample fo the second P-arrival when the first is in the center window
        :param random_shift: Shift to apply to the waveform and picks to make the second event in the center window
        :param plot_info: List - [plot_title, plot_output_name]
        :return:
        """
        org_pick_sample = wf1_norm.shape[0] // 2
        n_channels = wf1_norm.shape[1]
        # Trim WF1 to be the length needed to add wf2_trim to it
        wf1_trim = wf1_norm[org_pick_sample - first_event_sample:]

        # Trim WF2 to start just before the arrival
        wf2_trim = np.copy(
            wf2_norm[org_pick_sample - self.wf2_pre_p:org_pick_sample + (wf1_trim.shape[0] - second_event_sample)])

        # Make an obspy stream to taper it
        channels = {0:"Z"}
        if  n_channels == 3:
            channels = {0: "E", 1: "N", 2: "Z"}

        st = obspy.Stream()
        for ind in range(wf1_norm.shape[1]):
            tr = obspy.Trace(np.copy(wf2_trim[:, ind]))
            tr.stats.channel = channels[ind]
            tr.stats.delta = 0.01
            tr.stats.sampling_rate = 100
            st += tr

        st.taper(type="cosine", max_percentage=None, max_length=self.taper_max_length, side="left")

        # Convert Stream back to an array
        if n_channels == 3:
            wf2_taper = np.concatenate([st[0].data.reshape(-1, 1), st[1].data.reshape(-1, 1), st[2].data.reshape(-1, 1)],
                                   axis=1)
        else:
            wf2_taper = st[0].data.reshape(-1, 1)

        # Pad the front of wf2 with zeros so length is wf1_trim
        wf2_padded = np.zeros_like(wf1_trim)
        # Start wf2_buffer before the pick sample
        wf2_padded[second_event_sample - self.wf2_pre_p:] = wf2_taper

        #combined_wf = wf1_trim + wf2_padded
        combined_wf = np.add(wf1_trim, wf2_padded)
        boxcar = np.zeros(len(wf1_trim))
        boxcar[first_event_sample - self.boxcar_width:first_event_sample + self.boxcar_width + 1] = 1
        boxcar[second_event_sample - self.boxcar_width:second_event_sample + self.boxcar_width + 1] = 1

        combined_wf1_centered = np.copy(combined_wf)[0:self.wf_length]
        combined_wf2_centered = np.copy(combined_wf)[random_shift:random_shift + self.wf_length]

        boxcar_wf1_centered = boxcar[0:self.wf_length]
        boxcar_wf2_centered = boxcar[random_shift:random_shift + self.wf_length]

        # combined_wf1_centered_norm = combined_wf1_centered / np.max(abs(combined_wf1_centered), axis=0)
        # combined_wf2_centered_norm = combined_wf2_centered / np.max(abs(combined_wf2_centered), axis=0)
        combined_wf1_centered_norm = self.normalize_waveforms(combined_wf1_centered)
        combined_wf2_centered_norm = self.normalize_waveforms(combined_wf2_centered)

        if plot_info is not None:
            self.plot_waveforms(first_event_sample, second_event_sample, wf1_trim, wf2_trim, wf2_padded, combined_wf,
                                combined_wf1_centered_norm, combined_wf2_centered_norm, boxcar, boxcar_wf1_centered,
                                boxcar_wf2_centered, plot_info[0], plot_info[1])

        return combined_wf1_centered_norm, boxcar_wf1_centered, combined_wf2_centered_norm, boxcar_wf2_centered

    @staticmethod
    def normalize_waveforms(x_new, normalize_separate=True):
        """
        Normalize waveforms - shouldn't break if a trace is all zeros
        :param x_new: waveform time series data
        :param normalize_separate: True normalizes each trace separately
        :return:
        """
        if normalize_separate:
            X_normalizer = np.amax(np.abs(x_new), axis=0)
            assert X_normalizer.shape[0] is x_new.shape[1], "Normalizer is wrong shape for selected norm type"
        else:
            X_normalizer = np.amax(np.abs(x_new))
            assert X_normalizer.shape[0] is 1, "Normalizer is wrong shape for selected norm type"

        for norm_ind in range(len(X_normalizer)):
            if X_normalizer[norm_ind] != 0:
                x_new[:, norm_ind] = x_new[:, norm_ind] / X_normalizer[norm_ind]
            else:
                x_new[:, norm_ind] = x_new[:, norm_ind]

        assert np.max(abs(x_new)) <= 1

        return x_new

    def plot_waveforms(self, first_event_sample, second_event_sample, wf1_trim, wf2_trim, wf2_padded, combined_wf,
                       combined_wf1_centered_norm, combined_wf2_centered_norm, boxcar, boxcar_wf1_centered,
                       boxcar_wf2_centered, title, output_name, taper_plot_length=50):

        n_channels = wf1_trim.shape[1]
        bbox = dict(facecolor="white", alpha=0.5, edgecolor="white")

        fig, ax = plt.subplots(4, 2, figsize=(10, 10))

        fig.suptitle(title, y=0.95)

        ax = ax.flatten(order="F")

        ax[0].axvline(first_event_sample, color="r")
        if n_channels ==3:
            ax[0].plot(range(len(wf1_trim)), wf1_trim[:, 0], label="E")
            ax[0].plot(range(len(wf1_trim)), wf1_trim[:, 1], label="N")
            ax[0].plot(range(len(wf1_trim)), wf1_trim[:, 2], label="Z")
        else:
            ax[0].plot(range(len(wf1_trim)), wf1_trim[:, 0], label="Z")

        ax[0].legend()
        ax[0].text(0.1, 0.85, "Wf1", transform=ax[0].transAxes, bbox=bbox)

        ax[1].axvline(self.wf2_pre_p, color="r")
        ax[1].plot(range(len(wf2_trim)), wf2_trim)
        ax[1].text(0.1, 0.85, "Wf2", transform=ax[1].transAxes, bbox=bbox)

        # ax[2].plot(np.arange(0, center_window_lb+wf2_buffer), wf2_trim[:center_window_lb+wf2_buffer, 2], label="untapered")
        # ax[2].plot(np.arange(0, center_window_lb+wf2_buffer), st[2].data[:center_window_lb+wf2_buffer], label="tapered")
        ax[2].plot(np.arange(0, taper_plot_length), wf2_trim[:taper_plot_length, -1], label="untapered")
        ax[2].plot(np.arange(0, taper_plot_length),
                   wf2_padded[second_event_sample-self.wf2_pre_p:second_event_sample+taper_plot_length-self.wf2_pre_p, -1],
                   label="tapered")
        ax[2].axvline(self.wf2_pre_p, color="r")
        ax[2].legend(loc="upper right");
        ax[2].text(0.02, 0.85, "Wf2 taper", transform=ax[2].transAxes, bbox=bbox)

        ax[3].axvline(second_event_sample, color="r")
        ax[3].plot(range(len(wf2_padded)), wf2_padded)
        ax[3].text(0.02, 0.85, "Wf2 zero pad", transform=ax[3].transAxes, bbox=bbox)

        ax[4].plot(range(len(wf1_trim)), combined_wf[:, :])
        ax[4].plot(np.arange(len(wf1_trim)), boxcar, color="red")
        ax[4].text(0.02, 0.1, "Combined", transform=ax[4].transAxes)

        ax[5].plot(range(self.wf_length), combined_wf1_centered_norm)
        ax[5].plot(range(self.wf_length), boxcar_wf1_centered)
        ax[5].text(0.02, 0.1, "Combined - center wf1", transform=ax[5].transAxes, bbox=bbox)
        ax[5].axvline(self.center_window_lb, color="gray", linestyle="--")
        ax[5].axvline(self.center_window_ub, color="gray", linestyle="--")

        ax[6].plot(range(self.wf_length), combined_wf2_centered_norm)
        ax[6].plot(range(self.wf_length), boxcar_wf2_centered)
        ax[6].text(0.02, 0.1, "Combined - center wf2", transform=ax[6].transAxes, bbox=bbox)
        ax[6].axvline(self.center_window_lb, color="gray", linestyle="--")
        ax[6].axvline(self.center_window_ub, color="gray", linestyle="--")

        wf2_zoom_lb = second_event_sample - 100
        wf2_zoom_ub = second_event_sample + 100
        ax[7].axvline(second_event_sample, color="red", alpha=0.5)
        ax[7].plot(range(wf2_zoom_lb, wf2_zoom_ub), combined_wf1_centered_norm[wf2_zoom_lb:wf2_zoom_ub])
        ax[7].plot(range(wf2_zoom_lb, wf2_zoom_ub), boxcar_wf1_centered[wf2_zoom_lb:wf2_zoom_ub])
        ax[7].text(0.02, 0.1, "Combined - wf2 pick", transform=ax[7].transAxes, bbox=bbox)

        plt.savefig(output_name) #, facecolor="white")
        plt.close()

    def combine_mew_waveforms(self, pairs_df, X, outfile_pref, figdir=None, figure_interval=100):
        new_catalog = []
        comb_df = [["evid1", "evid2", "network", "station", "magnitude1", "magnitude1_type", "magnitude2",
                    "magnitude2_type", "wf1_T1", "wf1_T2", "wf2_shift", "rescale_factor"]]

        Y_aug = np.zeros((len(pairs_df), self.wf_length, 1))
        X_aug = np.zeros((len(pairs_df), self.wf_length, X.shape[2]))
        T_aug = np.zeros((len(pairs_df), 2))
        cnt = 0
        for index in np.arange(0, len(pairs_df)-1, 2):
            row1 = pairs_df.iloc[index]
            row2 = pairs_df.iloc[index+1]
            ind1 = row1["original_rows"]
            ind2 = row2["original_rows"]
            assert row1["station"] == row2["station"], "Stations are not the same"

            plot_info = None
            if (figdir is not None) and cnt % figure_interval == 0:
                plot_title =  f'{row1["network"]}.{row1["station"]} {row1["evid"]}-{row1["magnitude"]} {row1["magnitude_type"]}, ' \
                              f'{row2["evid"]}-{row2["magnitude"]} {row2["magnitude_type"]}'
                plot_name = f"{figdir}/{row1['network']}_{row1['station']}_{row1['evid']}_{row2['evid']}.png"
                plot_info = [plot_title, plot_name]

            wf1 = X[ind1, :, :].copy()
            wf2 = X[ind2, :, :].copy()

            # Process and rescale waveforms
            mag_diff = row1["magnitude"] - row2["magnitude"]
            wf1_norm, wf2_norm, rescale_factor = self.process_waveform_pair(wf1, wf2, mag_diff)

            # Get random samples to place the picks
            first_event_sample, second_event_sample, random_shift = self.get_random_shifts(row1["source_receiver_distance"])

            # Combine waveforms and create boxcars - plot if necessary
            combined_wf1_centered_norm, boxcar_wf1_centered, combined_wf2_centered_norm, \
            boxcar_wf2_centered = self.join_waveform_pairs(wf1_norm, wf2_norm, first_event_sample,
                                                           second_event_sample, random_shift, plot_info=plot_info)

            # Save information for catalog
            new_catalog.append(row1)
            new_catalog.append(row2)

            comb_df.append([row1["evid"],row2["evid"], row1["network"], row1["station"], row1["magnitude"],
                            row1["magnitude_type"], row2["magnitude"], row2["magnitude_type"], first_event_sample,
                            second_event_sample, random_shift, rescale_factor])

            # Add first combined waveform to file
            Y_aug[cnt, :] = np.expand_dims(boxcar_wf1_centered, axis=1)
            X_aug[cnt, :, :] = combined_wf1_centered_norm
            T_aug[cnt, 0] = first_event_sample
            T_aug[cnt, 1] = second_event_sample
            # Add second combined waveform to file - Put the pick in the center in Pick_index (T[:, 0])
            Y_aug[cnt+1, :] = np.expand_dims(boxcar_wf2_centered, axis=1)
            X_aug[cnt+1, :, :] = combined_wf2_centered_norm
            T_aug[cnt+1, 0] = second_event_sample - random_shift
            T_aug[cnt+1, 1] = first_event_sample - random_shift
            cnt += 2

        print(X_aug.shape, Y_aug.shape, T_aug.shape, cnt)
        self.write_h5file(X_aug, Y_aug, T_aug, outfile_pref)

        pd.DataFrame(new_catalog).to_csv("%s_synthetic_multievent_catalog.df.csv"%outfile_pref,
                                         index=False, float_format="%.7f")
        pd.DataFrame(comb_df[1:], columns=comb_df[0]).to_csv("%s_synthetic_multievent_summary_info.df.csv"%outfile_pref,
                                                             index=False, float_format="%.7f")

if __name__ == "__main__":
    split_type = "train"

    pref_entire_cat = "/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/oneCompPdetector"
    pref = "/uufs/chpc.utah.edu/common/home/koper-group1/alysha/Yellowstone/data/waveformArchive/oneCompPdetector/onecomp_p_resampled_10s"
    #outpref = f"{pref}/synthetic_multievent_waveforms"
    outpref = f"{pref}/synthetic_multievent_waveforms"

    outfile_pref = f"{outpref}/{split_type}P.10s.1dup"
    figdir = f"{outpref}/{split_type}_figs"

    if not os.path.exists(outpref):
        os.makedirs(outpref)

    if not os.path.exists(figdir):
        os.makedirs(figdir)

    mew = CreateMEW(1008, 250, 25)

    # Print mew attributes
    attrs = vars(mew)
    print(', '.join("%s: %s" % item for item in attrs.items()))

    n_waveforms = 40000
    min_magnitude_sep = 0.1
    max_magnitude_sep = 1.5

    # Read in entire 3C catalog
    entire_cat_df = pd.read_csv(f"{pref_entire_cat}/current_earthquake_catalog_1c.csv")
    entire_cat_df = entire_cat_df[entire_cat_df["phase"] == "P"]

    # Read in untrimmed/unnormalized data
    X = mew.read_h5file(f"{pref_entire_cat}/current_earthquake_catalog_1c.h5")

    # Read in split catalog
    df = pd.read_csv(f"{pref}/currenteq.{split_type}.10s.1dup.csv", dtype={'location'  : object})
    df = df[df.phase == "P"]


    # Clean up the dataset
    df_clean, catalog_close_event_df = mew.clean_df(df, pick_indicator="arrival_time", quality_req=[1.0, 0.75], sr_dist_max=35)

    print(f"Minimum magnitude separation: -{min_magnitude_sep}")
    print(f"Maximum magnitude separation: {max_magnitude_sep}")
    
    mew.plot_sampling_distributions(df_clean, "Magnitude distribution of filtered training data",
                                    figdir="%s/filtered_training_dist.jpg" % figdir)

    mew.plot_sampling_distributions(entire_cat_df, "Magnitude distribution of entire 3C catalog",
                                    figdir="%s/catalog_dist.jpg" % figdir)
    df_pairs, sampled_dist = mew.generate_sampling_distribution(df_clean, n_waveforms=n_waveforms, z_indicator="channelz",
                                                                min_sep=min_magnitude_sep, max_sep=max_magnitude_sep)
    mew.plot_sampling_distributions(df_pairs, "Magnitude distribution of sampled training catalog",
                                    figdir="%s/sampled_filtered_training_dist.jpg" % figdir)
    # combine the waveforms
    mew.combine_mew_waveforms(df_pairs, X, outfile_pref, figdir=figdir, figure_interval=100)

    ################ Training ################
    # if split_type == "train":
    #     mew.plot_sampling_distributions(df_clean, "Magnitude distribution of filtered training data",
    #                               figdir="%s/filtered_training_dist.jpg"%figdir)
    #
    #     mew.plot_sampling_distributions(entire_cat_df, "Magnitude distribution of entire 3C catalog",
    #                                 figdir="%s/3C_catalog_dist.jpg"%figdir)
    #     df_pairs, sampled_dist = mew.generate_sampling_distribution(df_clean, n_waveforms=1000, z_indicator="channelz",
    #                                                                 min_sep=min_magnitude_sep, max_sep=max_magnitude_sep)
    #     mew.plot_sampling_distributions(df_pairs, "Magnitude distribution of sampled training catalog",
    #                                 figdir="%s/sampled_filtered_training_dist.jpg"%figdir)
    #     # combine the waveforms
    #     mew.combine_mew_waveforms(df_pairs, X, outfile_pref, figdir=figdir)
    # elif split_type == "validate":
    #     ################ Validation ################
    #     # Only use this for validation/test data because much smaller size
    #     df_match_dist = mew.match_sampling_distributions(entire_cat_df, df_clean, 15000,
    #                                                      "Magnitude distribution of different data sets",
    #                                                      z_indicator="channelz", min_sep=min_magnitude_sep,
    #                                                      max_sep=max_magnitude_sep)
    #     #Plot distributions and make dataframe of events to combine
    #     mew.plot_sampling_distributions(df, "Magnitude distribution of validation data", figdir="%s/original_validation_dist.jpg"%figdir)
    #     mew.plot_sampling_distributions(df_match_dist, "Mag dist of filtered validation data made to match whole catalog dist",
    #                                 figdir="%s/filtered_validation_dist.jpg"%figdir)
    #     mew.plot_sampling_distributions(df_match_dist, "Mag dist of filtered validation data made to match whole catalog dist - all mag types",
    #                                 filter=False, figdir="%s/filtered_validation_dist_allmagtypes.jpg"%figdir)
    #     mew.plot_sampling_distributions(entire_cat_df, "Magnitude distribution of entire 3C catalog",
    #                                 figdir="%s/3C_catalog_dist.jpg"%figdir)
    #     #combine the waveforms
    #     mew.combine_mew_waveforms(df_match_dist, X, outfile_pref, figdir=figdir)
