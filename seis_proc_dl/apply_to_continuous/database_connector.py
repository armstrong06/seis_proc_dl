import sys
import obspy
import numpy as np
import os
from copy import deepcopy
from datetime import timedelta
from tables import open_file
from seis_proc_db import database
from seis_proc_db import services
from seis_proc_db import pytables_backend
from seis_proc_db.config import DETECTION_GAP_BUFFER_SECONDS
from seis_proc_db.tables import Waveform, DailyContDataInfo, Channel, DLDetection
from seis_proc_dl.utils.config_apply_detectors import Config

# TODO: Think I need to not store ORM objects, use them in the same session only. Just store id's and get other info as needed.

def store_source_method_info(ncomps, config):
    config = Config.from_json(config)
    with database.Session() as session:
        with session.begin():
            services.upsert_waveform_source(
                session,
                name="extract-dailyContData",
                details=(
                    "Waveform snippets are from the daily mseed file that has been loaded and "
                    "processed using seis-proc-dl.apply_to_continuous.apply_detectors.DataLoader. "
                    "Mseed files were resampled to 100 Hz beforehand. No additional processing is applied."
                ),
                path=None,
                filt_low=None,
                filt_high=None,
                detrend=None,
                normalize=None,
                common_samp_rate=100.0,
            )
            if ncomps == 1:
               services.upsert_detection_method(
                    session,
                    name=config.database.det_method_1c_P.name,
                    details=config.database.det_method_1c_P.desc,
                    path=config.paths.one_comp_p_model,
                    phase="P",
                )
            elif ncomps == 3:
                services.upsert_detection_method(
                    session,
                    name=config.database.det_method_3c_P.name,
                    details=config.database.det_method_3c_P.desc,
                    path=config.paths.three_comp_p_model,
                    phase="P",
                )
                services.upsert_detection_method(
                    session,
                    name=config.database.det_method_3c_S.name,
                    details=config.database.det_method_3c_S.desc,
                    path=config.paths.three_comp_s_model,
                    phase="S",
                )

class DailyDetectionDBInfo:
    def __init__(self, date):
        self.date = date
        self.contdatainfo_id = None
        # TODO If I use bulk inserts, there won't be any detections/picks to store from the insert
        self.detections = None
        self.picks = None
        self.dldet_output_id_P = None
        self.dldet_output_id_S = None
        self.start_time = None
        self.samp_rate = None


class ChannelInfo:
    def __init__(self, channels, total_ndays):
        self.ondate = None
        self.offdate = None
        self.channel_ids = None
        self.ndays = total_ndays

        if channels is not None and len(channels) > 0:
            self.ondate = np.min([chan.ondate for chan in channels])
            ends = [chan.offdate for chan in channels]
            self.offdate = None if None in ends else np.max(ends)
            self.channel_ids = self.gather_chan_ids(channels)

    def gather_chan_ids(self, channels):
        channel_ids = {}
        for chan in channels:
            channel_ids[chan.seed_code] = chan.id
        return channel_ids


class DetectorDBConnection:
    EXPECTED_DAILY_P_PICKS = 1000
    EXPECTED_DAILY_S_PICKS = 1000
    MAX_WAVEFORMS_PER_STORAGE = 150_000

    def __init__(self, ncomps, session_factory=None, logger=None):
        self.Session = session_factory or database.Session
        self.ncomps = ncomps
        self.initial_year = None

        self.station_name = None
        self.seed_code = None
        self.net = None
        self.loc = None
        self.last_channel_date = None

        self.station_id = None
        self.channel_info = None
        self.p_detection_method_id = None
        self.s_detection_method_id = None
        self.daily_info = None
        self.wf_source_id = None

        # BasePyTables storage
        # Wavefroms will need a storage for each channel - keep them in dicts
        self.prev_waveform_storage_dict_P = None
        self.prev_waveform_storage_dict_S = None
        self.waveform_storage_dict_P = None
        self.waveform_storage_dict_S = None
        self.n_P_waveform_storage_entries = 0
        self.n_S_waveform_storage_entries = 0
        # Only need one storage per phase for detection outputs
        self.detout_storage_P = None
        self.detout_storage_S = None
        # Store gap buffer info
        self.DETECTION_GAP_BUFFER_SECONDS = DETECTION_GAP_BUFFER_SECONDS
        self.logger = logger

    def get_channel_dates(self, session, date, net, stat, loc, seed_code):
        """Returns the start and end times of the relevant channels for a station"""
        # The database will handel the "?" differently
        if len(seed_code) == 3 and self.ncomps == 3:
            seed_code = seed_code[:-1]
        self.seed_code = seed_code
        self.station_name = stat
        self.net = net
        self.loc = loc
        self.initial_year = date.year

        # Get all channels for this station name and channel type
        all_channels = services.get_common_station_channels_by_name(
            session, stat, seed_code, net=net, loc=loc
        )

        # Get the Station object and the Channel objects for the appropriate date
        selected_stat, selected_channels = (
            services.get_operating_channels_by_station_name(
                session, stat, seed_code, date, net=net, loc=loc
            )
        )

        # Set the start date to the minimum date for all channels of the desired type
        start_date = np.min([chan.ondate for chan in all_channels])
        # Set the end date to the maximum date for all channels of the desired type
        ends = [chan.offdate for chan in all_channels]
        end_date = None if None in ends else np.max(ends)

        if selected_stat is not None:
            # If there is a selected stat,
            assert (
                len(selected_channels) == self.ncomps
            ), f"Number of channels selected ({len(selected_channels)}) does not agree with the number of components ({self.ncomps})"
            # Store the station
            self.station_id = selected_stat.id
            # Store the current channels
            total_ndays = services.get_similar_channel_total_ndays(
                session, net, stat, loc, selected_channels[0].seed_code
            )
            self.channel_info = ChannelInfo(selected_channels, total_ndays)

        self.last_channel_date = end_date
        return start_date, end_date

    def get_waveform_source(self, session, name):
        wf_source = services.get_waveform_source(session, name)
        if wf_source is None:
            raise ValueError(f"No waveform source with name {name} found in the db")
        
        self.wf_source_id = wf_source.id

    def get_detection_method(self, session, phase, name):
        det_method = services.get_detection_method(session, name)
        if det_method is None:
            raise ValueError(f"No detection method with name {name} found in the db")
        
        if phase == "P":
            self.p_detection_method_id = det_method.id
        elif phase == "S":
            self.s_detection_method_id = det_method.id
        else:
            raise ValueError("Invalid Phase for Detection Method")

    def add_waveform_source(
        self,
        session,
        name,
        desc,
        path=None,
        filt_low=None,
        filt_high=None,
        detrend=None,
        normalize=None,
        common_samp_rate=None,
    ):
        """Add a waveform source to the database. If it already exists, update it."""

        services.upsert_waveform_source(
            session,
            name,
            details=desc,
            path=path,
            filt_low=filt_low,
            filt_high=filt_high,
            detrend=detrend,
            normalize=normalize,
            common_samp_rate=common_samp_rate,
        )
        session.flush()
        self.wf_source_id = services.get_waveform_source(session, name).id

    def add_detection_method(self, session, name, desc, path, phase):
        """Add a detection method to the database. If it already exists, update it."""
        services.upsert_detection_method(
            session, name, phase=phase, details=desc, path=path
        )
        if phase == "P":
            self.p_detection_method_id = services.get_detection_method(session, name).id
        elif phase == "S":
            self.s_detection_method_id = services.get_detection_method(session, name).id
        else:
            raise ValueError("Invalid Phase for Detection Method")

    def start_new_day(self, session, date):
        self.daily_info = DailyDetectionDBInfo(date)
        valid_channels = self.validate_channels_for_date(date)
        if not valid_channels:
            channel_continues = self.update_channels(session, date)

            if not channel_continues:
                return False

        return True

    def validate_channels_for_date(self, date):
        if (
            self.channel_info is not None
            and date >= self.channel_info.ondate
            and (self.channel_info.offdate is None or date <= self.channel_info.offdate)
        ):
            return True

        return False

    def update_channels(self, session, date):
        # Get the Station object and the Channel objects for the appropriate date
        selected_station, selected_channels = (
            services.get_operating_channels_by_station_name(
                session,
                self.station_name,
                self.seed_code,
                date,
                net=self.net,
                loc=self.loc,
            )
        )

        if selected_channels is None or len(selected_channels) == 0:
            self.channel_info = None  # ChannelInfo([], 0)
            return False

        if self.station_id is None:
            self.station_id = selected_station.id

        total_ndays = services.get_similar_channel_total_ndays(
            session,
            self.net,
            self.station_name,
            self.loc,
            selected_channels[0].seed_code,
        )

        self.channel_info = ChannelInfo(selected_channels, total_ndays)

        self._reset_waveform_P_storages()
        if self.ncomps == 3:
            self._reset_waveform_S_storages()
        return True

    def save_data_info(self, session, date, metadata_dict, error=None):
        """Add cont data info into the database. If it already exists, checks that the
        information is the same"""
        # TODO: Maybe I should add processing method table and make it part of the PK?
        # For now, I am just going to assume there is one processing method and the
        # results will be the same every time. Hopefully that's fine...

        started_new_day = self.start_new_day(session, date)
        if not started_new_day and (
            self.last_channel_date is None or date <= self.last_channel_date
        ):
            error = "gap_between_new_channels"
        elif not started_new_day:
            raise ValueError(f"Cannot start processing {date}, no channel info exists")

        db_dict = {
            "sta_id": self.station_id,
            "chan_pref": self.seed_code,
            "chan_loc": self.loc,
            "ncomps": self.ncomps,
            "date": date,
            "error": error,
        }

        if metadata_dict is not None:
            db_dict = db_dict | {
                "samp_rate": metadata_dict["sampling_rate"],
                "dt": metadata_dict["dt"],
                "orig_npts": metadata_dict["original_npts"],
                "orig_start": metadata_dict["original_starttime"],
                "orig_end": metadata_dict["original_endtime"],
                "proc_npts": metadata_dict["npts"] if error is None else None,
                "proc_start": (metadata_dict["starttime"] if error is None else None),
                "proc_end": None,  # just get this from proc_start, samp_rate, and proc_npts
                "prev_appended": metadata_dict["previous_appended"],
            }

        contdatainfo = services.get_contdatainfo(
            session,
            self.station_id,
            self.seed_code,
            self.ncomps,
            date,
            chan_loc=self.loc,
        )

        if contdatainfo is None:
            contdatainfo = services.insert_contdatainfo(session, db_dict)
            session.flush()
        elif metadata_dict is None:
            if contdatainfo.error != db_dict["error"]:
                info_str = (
                    f"{db_dict['sta_id']}, {db_dict['chan_pref']}, {db_dict['date']}"
                )
                raise ValueError(
                    f"DailyContDataInfo {info_str} row already exists but the error when loading has changed"
                )
        else:
            # I think I did this because I'm fine with the entry existing already,
            # as long as all the values are the same. just calling insert and catching
            # the IntegrityError when there is a duplicate entry would not check
            if (
                contdatainfo.samp_rate != db_dict["samp_rate"]
                or contdatainfo.dt != db_dict["dt"]
                or contdatainfo.orig_npts != db_dict["orig_npts"]
                or contdatainfo.orig_start != db_dict["orig_start"]
                or contdatainfo.proc_start != db_dict["proc_start"]
                or contdatainfo.proc_npts != db_dict["proc_npts"]
                or contdatainfo.prev_appended != db_dict["prev_appended"]
                or contdatainfo.error != db_dict["error"]
            ):
                info_str = (
                    f"{db_dict['sta_id']}, {db_dict['chan_pref']}, {db_dict['date']}"
                )
                raise ValueError(
                    f"DailyContDataInfo {info_str} row already exists but the values have changed"
                )

        self.daily_info.contdatainfo_id = contdatainfo.id
        self.daily_info.start_time = contdatainfo.proc_start
        self.daily_info.samp_rate = contdatainfo.samp_rate

    def save_gaps(self, session, formatted_gaps):
        """Add gaps into the table. If many gaps in the same time period, combine them
        into one 'effective' gap. gaps should be a list of lists containing the gap seed_code, startime and endtime
        """

        if formatted_gaps is None or len(formatted_gaps) == 0:
            return

        services.insert_gaps(session, formatted_gaps)
        session.flush()

    def format_gaps(self, gaps, min_gap_sep_seconds):
        if gaps is None or len(gaps) == 0:
            return

        assert len(gaps[0]) == 3, "Expected just three values in the gap"

        formatted_gaps = []
        for seed_code in self.channel_info.channel_ids.keys():
            chan_id = self.channel_info.channel_ids[seed_code]
            # Get only the gaps for one channel
            chan_gaps = list(filter(lambda x: x[0] == seed_code, gaps))
            # chan_inds = np.where(gap_chans == seed_code)[0]
            # chan_starts = gap_starts[chan_inds]
            # chan_ends = gap_ends[chan_inds]
            # Format the gaps for inserting
            chan_gaps = self.format_channel_gaps(
                chan_gaps, chan_id, min_gap_sep_seconds
            )
            formatted_gaps += chan_gaps

        return formatted_gaps

    def format_channel_gaps(self, gaps, chan_id, min_gap_sep_seconds):
        if gaps is None:
            return []

        gaps.sort(key=lambda x: x[1])
        data_id = self.daily_info.contdatainfo_id
        merged = []
        for current in gaps:
            if not merged:
                merged.append(self.convert_gap_to_dict(current, data_id, chan_id))
            else:
                previous = merged[-1]
                # Compare the start time of the current gap to the end time of the last one
                gap_delta = (current[1] - previous["end"]).total_seconds()
                assert gap_delta >= 0, ValueError("Two adjacent gaps are overlapping")
                if gap_delta < min_gap_sep_seconds:
                    previous["end"] = current[2]
                    previous["avail_sig_sec"] += gap_delta
                else:
                    merged.append(self.convert_gap_to_dict(current, data_id, chan_id))

        return merged

    @staticmethod
    def convert_gap_to_dict(gap, data_id, chan_id):
        d = {
            "data_id": data_id,
            "chan_id": chan_id,
            # (gap[1] if type(gap[1]) == datetime else gap[1].datetime),
            "start": gap[1],
            # (gap[2] if type(gap[2]) == datetime else gap[2].datetime),
            "end": gap[2],
            "avail_sig_sec": 0.0,
        }
        return d

    def save_detections(self, session, detections):
        """Add detections above a threshold into the database. Do not add detections if
        they exist within a gap. The tresholding and gap checking is handled earlier."""

        if len(detections) == 0:
            return

        # services.bulk_insert_dldetections_with_gap_check(session, detections)
        services.bulk_insert_dldetections(session, detections)
        session.flush()

    def get_dldet_fk_ids(self, is_p=True):
        d = {
            "data": self.daily_info.contdatainfo_id,
        }
        if is_p:
            d["method"] = self.p_detection_method_id
            d["detout"] = self.daily_info.dldet_output_id_P
        else:
            d["method"] = self.s_detection_method_id
            d["detout"] = self.daily_info.dldet_output_id_S
        return d

    def save_P_post_probs(
        self, session, data, expected_array_length=8640000, on_event=None
    ):
        if self.detout_storage_P is None:
            self.detout_storage_P = self._open_dldetection_output_storage(
                expected_array_length=expected_array_length,
                phase="P",
                det_method_id=self.p_detection_method_id,
                on_event=on_event,
                year=self.initial_year,
            )

        if len(data) < expected_array_length:
            tmp = np.zeros((int(expected_array_length - len(data)),), dtype=np.uint8)
            data = np.concatenate([data, tmp])

        detout_id = self._save_detection_output(
            session,
            self.detout_storage_P,
            data,
            self.p_detection_method_id,
        )
        self.daily_info.dldet_output_id_P = detout_id

    def save_S_post_probs(
        self, session, data, expected_array_length=8640000, on_event=None
    ):
        if self.detout_storage_S is None:
            self.detout_storage_S = self._open_dldetection_output_storage(
                expected_array_length=expected_array_length,
                phase="S",
                det_method_id=self.s_detection_method_id,
                on_event=on_event,
                year=self.initial_year,
            )

        if len(data) < expected_array_length:
            tmp = np.zeros((int(expected_array_length - len(data)),), dtype=np.uint8)
            data = np.concatenate([data, tmp])

        detout_id = self._save_detection_output(
            session, self.detout_storage_S, data, self.s_detection_method_id
        )
        self.daily_info.dldet_output_id_S = detout_id

    def _open_dldetection_output_storage(
        self, expected_array_length, phase, det_method_id, year=None, on_event=None
    ):
        storage = pytables_backend.DLDetectorOutputStorage(
            expected_array_length=expected_array_length,
            net=self.net,
            sta=self.station_name,
            loc=self.loc,
            seed_code=self.seed_code,
            ncomps=self.ncomps,
            phase=phase,
            det_method_id=det_method_id,
            on_event=on_event,
            expectedrows=self.channel_info.ndays,
            year=year,
        )
        return storage

    def _save_detection_output(self, session, storage, data, det_method_id):
        detout_id = None

        try:
            storage.start_transaction()
            detout = services.insert_dldetector_output_pytable(
                session,
                storage,
                self.daily_info.contdatainfo_id,
                det_method_id,
                data.astype(np.uint8),
            )
            detout_id = detout.id
        except Exception as e:
            storage.rollback()
            self.close_open_pytables()
            raise e

        session.flush()
        storage.commit()
        return detout_id

    def _get_waveform_storages(self, session, is_p, common_wf_details):
        if is_p:
            prev_storage = self.prev_waveform_storage_dict_P
            curr_storage = self.waveform_storage_dict_P
            curr_count = self.n_P_waveform_storage_entries
            phase = "P"
        else:
            prev_storage = self.prev_waveform_storage_dict_S
            curr_storage = self.waveform_storage_dict_S
            curr_count = self.n_S_waveform_storage_entries
            phase = "S"

        if curr_storage is None or (curr_count >= self.MAX_WAVEFORMS_PER_STORAGE):
            if curr_storage is not None:
                if prev_storage is not None:
                    for key, stor in prev_storage.items():
                        stor.close()

                prev_storage = curr_storage
                curr_storage = None

            new_storage = {}
            for seed_code, chan_id in self.channel_info.channel_ids.items():
                # TODO: implement this function
                storage_number, hdf_file, count = services.get_waveform_storage_number(
                    session,
                    chan_id,
                    self.wf_source_id,
                    phase,
                    self.MAX_WAVEFORMS_PER_STORAGE,
                )
                new_storage[chan_id] = pytables_backend.WaveformStorage(
                    expected_array_length=common_wf_details["expected_array_length"],
                    net=self.net,
                    sta=self.station_name,
                    loc=self.loc,
                    seed_code=seed_code,
                    ncomps=self.ncomps,
                    storage_number=storage_number,
                    phase=common_wf_details["phase"],
                    wf_source_id=self.wf_source_id,
                    on_event=common_wf_details["on_event"],
                    expectedrows=self.MAX_WAVEFORMS_PER_STORAGE,
                    year=self.initial_year,
                    # expectedrows=(
                    #     self.EXPECTED_DAILY_P_PICKS * self.channel_info.ndays
                    #     if is_p
                    #     else self.EXPECTED_DAILY_S_PICKS * self.channel_info.ndays
                    # ),
                )

            if is_p:
                self.waveform_storage_dict_P = new_storage
                self.n_P_waveform_storage_entries = count
                self.prev_waveform_storage_dict_P = prev_storage
            else:
                self.waveform_storage_dict_S = new_storage
                self.n_S_waveform_storage_entries = count
                self.prev_waveform_storage_dict_S = prev_storage

            return new_storage
        else:
            return curr_storage

    def _reset_waveform_P_storages(self):
        if self.prev_waveform_storage_dict_P is not None:
            for key, stor in self.prev_waveform_storage_dict_P.items():
                stor.close()

        self.prev_waveform_storage_dict_P = self.waveform_storage_dict_P
        self.waveform_storage_dict_P = None

    def _reset_waveform_S_storages(self):
        if self.prev_waveform_storage_dict_S is not None:
            for key, stor in self.prev_waveform_storage_dict_S.items():
                stor.close()

        self.prev_waveform_storage_dict_S = self.waveform_storage_dict_S
        self.waveform_storage_dict_S = None

    def save_picks_from_detections(
        self,
        session,
        pick_thresh,
        is_p,
        auth,
        continuous_data,
        seconds_around_pick,
        use_pytables=True,
        on_event=None,
    ):
        """Add detections above a certain threshold into the pick table, with the
        necessary additional information"""

        common_wf_details = {
            "use_pytables": use_pytables,
            "on_event": on_event,
        }
        storage_dict = None

        # Get ids
        data_id = self.daily_info.contdatainfo_id
        if is_p:
            method_id = self.p_detection_method_id
            phase = "P"
        else:
            method_id = self.s_detection_method_id
            phase = "S"
        # Compute the number of samples to grab on either side of the detection
        cdi = session.get(DailyContDataInfo, data_id)
        cdi_date = cdi.date
        cdi_prev_appended = cdi.prev_appended
        samples_around_pick = int(seconds_around_pick * cdi.samp_rate)
        total_npts = len(continuous_data)
        total_expected_samples = samples_around_pick * 2 + 1

        common_wf_details["expected_array_length"] = total_expected_samples
        common_wf_details["phase"] = phase
        common_wf_details["data_id"] = data_id

        # Get all detections for the contdatainfo and method greater than the pick_thresh
        dldets = services.get_dldetections(
            session, data_id, method_id, pick_thresh, phase=phase
        )

        proc_dldets = []
        # Iterate over the detections
        for det in dldets:
            pick_waveform_details = {}

            ## Compute the start and end inds for waveforms in pytable
            ## Rely on the default values in the storage object
            wf_start_ind = 0
            wf_end_ind = total_expected_samples
            ##

            # Compute the relevant waveform information for all channels
            i1 = det.sample - samples_around_pick
            i2 = det.sample + samples_around_pick + 1
            if i1 < 0:
                wf_start_ind = abs(i1)
                i1 = 0
            # TODO: Check if this needs a -1
            if i2 > total_npts:
                wf_end_ind -= i2 - total_npts
                i2 = total_npts

            pick_cont_data = deepcopy(continuous_data[i1:i2, :])
            wf_start = cdi.proc_start + timedelta(seconds=(i1 * cdi.dt))
            wf_end = cdi.proc_start + timedelta(seconds=(i2 * cdi.dt))
            pick_waveform_details["npts"] = i2 - i1
            pick_waveform_details["wf_start"] = wf_start
            pick_waveform_details["wf_end"] = wf_end
            pick_waveform_details["wf_start_ind"] = wf_start_ind
            pick_waveform_details["wf_end_ind"] = wf_end_ind
            pick_waveform_details["pick_cont_data"] = pick_cont_data
            pick_waveform_details["det_time"] = det.time
            pick_waveform_details["det_id"] = det.id
            pick_waveform_details["det_phase"] = det.phase
            proc_dldets.append(pick_waveform_details)

        prev_storage_dict = None
        try:
            #### GET THE PYTABLES STORAGE
            storage_dict = None
            if use_pytables:
                storage_dict = self._get_waveform_storages(
                    session, is_p, common_wf_details
                )

                # Start a transaction for these
                for _, chan_storage in storage_dict.items():
                    chan_storage.start_transaction()

                if is_p:
                    prev_storage_dict = self.prev_waveform_storage_dict_P
                else:
                    prev_storage_dict = self.prev_waveform_storage_dict_S

                # Start a transaction for these
                if prev_storage_dict is not None:
                    for _, chan_storage in prev_storage_dict.items():
                        chan_storage.start_transaction()
            #####
            for det_dict in proc_dldets:
                # Check if the detection is on the previous day, if so need to check
                # for existing picks and handle accordingly
                insert_new_pick = True
                if det_dict["det_time"].date() < cdi_date:
                    assert (
                        cdi_prev_appended
                    ), "Previous data was not appended, yet there is a detection on the previous day..."

                    insert_new_pick = self._potentially_modify_pick_and_waveform(
                        session,
                        storage_dict,
                        prev_storage_dict,
                        det_dict,
                        common_wf_details,
                    )

                if insert_new_pick:
                    # Create a pick from the detection
                    pick = services.insert_pick(
                        session,
                        self.station_id,
                        self.seed_code,
                        self.loc,
                        det_dict["det_phase"],
                        det_dict["det_time"],
                        auth,
                        detid=det_dict["det_id"],
                    )
                    self._insert_new_waveforms(
                        session,
                        storage_dict,
                        pick,
                        det_dict,
                        common_wf_details,
                    )

                    if is_p:
                        self.n_P_waveform_storage_entries += 1
                    else:
                        self.n_S_waveform_storage_entries += 1

        except Exception as e:
            if use_pytables and storage_dict is not None:
                # Rollback the pytables updates if an error occurs
                for _, chan_storage in storage_dict.items():
                    chan_storage.rollback()
                if prev_storage_dict is not None:
                    for _, chan_storage in prev_storage_dict.items():
                        chan_storage.rollback()
                self.close_open_pytables()

            raise e

        if use_pytables:
            # Commit the pytables updates if everything went well in the database transaction
            for _, chan_storage in storage_dict.items():
                chan_storage.commit()
            if prev_storage_dict is not None:
                for _, chan_storage in prev_storage_dict.items():
                    chan_storage.commit()

    def _insert_new_waveforms(
        self, session, storage_dict, pick, pick_wf_details, common_wf_details
    ):
        pick_cont_data = pick_wf_details["pick_cont_data"]
        for seed_code in self.channel_info.channel_ids.keys():
            chan_id = self.channel_info.channel_ids[seed_code]

            # Get the appropriate channel index of the data
            chan_ind = get_channel_data_index(self.ncomps, seed_code)

            # Get just the channel of interest
            wf_data = pick_cont_data[:, chan_ind]

            if not common_wf_details["use_pytables"]:
                # Create the waveform object
                wf = Waveform(
                    data_id=common_wf_details["data_id"],
                    chan_id=chan_id,
                    data=wf_data.tolist(),
                    start=pick_wf_details["wf_start"],
                    end=pick_wf_details["wf_end"],
                    wf_source_id=self.wf_source_id,
                )
                # Add the waveform to the pick
                pick.wfs.add(wf)
            else:
                # Need to flush to get the pick id
                session.flush()
                chan_storage = storage_dict[chan_id]
                pytables_wf_data = np.zeros(
                    common_wf_details["expected_array_length"], dtype=np.float32
                )
                pytables_wf_data[
                    pick_wf_details["wf_start_ind"] : pick_wf_details["wf_end_ind"]
                ] = wf_data
                _ = services.insert_waveform_pytable(
                    session,
                    chan_storage,
                    data=pytables_wf_data,
                    data_id=common_wf_details["data_id"],
                    chan_id=chan_id,
                    pick_id=pick.id,
                    wf_source_id=self.wf_source_id,
                    start=pick_wf_details["wf_start"],
                    end=pick_wf_details["wf_end"],
                    signal_start_ind=pick_wf_details["wf_start_ind"],
                    signal_end_ind=pick_wf_details["wf_end_ind"],
                )

    def _potentially_modify_pick_and_waveform(
        self,
        session,
        storage_dict,
        prev_storage_dict,
        pick_waveform_details,
        common_waveform_details,
    ):
        new_pick_needed = True
        # Check if there are any picks with a ptime close to det.time
        close_picks = services.get_picks(
            session,
            self.station_id,
            self.seed_code,
            self.loc,
            common_waveform_details["phase"],
            min_time=pick_waveform_details["det_time"] - timedelta(seconds=0.1),
            max_time=pick_waveform_details["det_time"] + timedelta(seconds=0.1),
        )

        # If there are no close picks, then insert_new_pick = True
        if len(close_picks) == 0:
            return new_pick_needed
        if len(close_picks) > 1:
            self.close_open_pytables()
            raise NotImplementedError(
                "There are multiple close picks in the previous day's data..."
            )

        # If made it to this point, will update or keep an existing pick
        new_pick_needed = False

        # There's only one pick
        pick = close_picks[0]
        prev_data_id = session.get(DLDetection, pick.detid).data_id

        ## Get close waveform or waveform_info.
        if not common_waveform_details["use_pytables"]:
            # Get the waveforms for these picks
            close_wfs = services.get_waveforms(session, pick.id, data_id=prev_data_id)
            prev_inserted_npts = len(close_wfs[0].data)
        else:
            ## This will get waveform info instead of waveform, but they can be treated similarly
            ## Just the data will need to also be grabbed from a pytable.
            close_wfs = services.get_waveform_infos(
                session, pick.id, data_id=prev_data_id, wf_source_id=self.wf_source_id
            )
            prev_inserted_npts = close_wfs[0].duration_samples
        ##

        # Only update the pick and waveforms if the waveforms would have more continuous data available
        if prev_inserted_npts > pick_waveform_details["npts"]:
            return new_pick_needed

        # If so, update the pick time and detection id, everything else should be the same
        pick.ptime = pick_waveform_details["det_time"]
        pick.detid = pick_waveform_details["det_id"]

        # This will iterate over Waveform if not using pytables and WaveformInfo otherwise
        for wf in close_wfs:
            seed_code = session.get(Channel, wf.chan_id).seed_code
            # Get the appropriate channel index of the data
            chan_ind = get_channel_data_index(self.ncomps, seed_code)
            wf.start = pick_waveform_details["wf_start"]
            wf.end = pick_waveform_details["wf_end"]
            pick_cont_data = pick_waveform_details["pick_cont_data"]
            wf.data_id = common_waveform_details["data_id"]
            if not common_waveform_details["use_pytables"]:
                # Get just the channel of interest
                wf.data = pick_cont_data[:, chan_ind].tolist()
            else:
                # Pytables expects a fixed length array
                wf_data = np.zeros(
                    common_waveform_details["expected_array_length"], dtype=np.float32
                )
                wf_start_ind = pick_waveform_details["wf_start_ind"]
                wf_end_ind = pick_waveform_details["wf_end_ind"]
                wf_data[wf_start_ind:wf_end_ind] = pick_cont_data[:, chan_ind]
                # Modify the pytable entry
                if (
                    wf.chan_id in storage_dict.keys()
                    and storage_dict[wf.chan_id].file_name == wf.hdf_file.name
                ):
                    storage_dict[wf.chan_id].modify(
                        wf.id, wf_data, wf_start_ind, wf_end_ind
                    )
                elif (
                    prev_storage_dict is not None
                    and wf.chan_id in prev_storage_dict.keys()
                ):
                    prev_storage_dict[wf.chan_id].modify(
                        wf.id, wf_data, wf_start_ind, wf_end_ind
                    )
                else:
                    raise ValueError(
                        "wf.chan_id not in the previous or current storage dict"
                    )
            # TODO: Might want to update this in case the channel switched between days...
            # But I think it makes sense to still assign it to the previous day's channel
            # wf.chan_id = self.channel_info.channel_ids[seed_code]

        return new_pick_needed

    def close_open_pytables(self):
        if self.detout_storage_P is not None:
            self.detout_storage_P.close()
            self.detout_storage_P = None

        if self.detout_storage_S is not None:
            self.detout_storage_S.close()
            self.detout_storage_S = None

        if self.waveform_storage_dict_P is not None:
            for key, stor in self.waveform_storage_dict_P.items():
                stor.close()
            self.waveform_storage_dict_P = None
        if self.waveform_storage_dict_S is not None:
            for key, stor in self.waveform_storage_dict_S.items():
                stor.close()
            self.waveform_storage_dict_S = None
        if self.prev_waveform_storage_dict_P is not None:
            for key, stor in self.prev_waveform_storage_dict_P.items():
                stor.close()
            self.prev_waveform_storage_dict_P = None
        if self.prev_waveform_storage_dict_S is not None:
            for key, stor in self.prev_waveform_storage_dict_S.items():
                stor.close()
            self.prev_waveform_storage_dict_S = None


class SwagPickerDBConnection:

    def __init__(self, phase, session_factory=None):
        self.Session = session_factory or database.Session
        # self.is_p = True if phase == "P" else False
        self.phase = phase
        self.repicker_method_id = None
        self.calibration_method_id = None

    def add_repicker_method(self, session, name, desc, path, addition_params_dict={}):
        """Add a repicker method to the database. If it already exists, update it."""
        services.upsert_repicker_method(
            session,
            name,
            phase=self.phase,
            details=desc,
            path=path,
            **addition_params_dict,
        )
        self.repicker_method_id = services.get_repicker_method(session, name).id

    def add_calibration_method(self, session, name, desc, path, loc_type, scale_type):
        """Add a calibration method to the database. If it already exists, update it."""

        services.upsert_calibration_method(
            session,
            name,
            phase=self.phase,
            details=desc,
            path=path,
            loc_type=loc_type,
            scale_type=scale_type,
        )
        self.calibration_method_id = services.get_calibration_method(session, name).id

    def get_waveforms(
        self,
        session,
        n_samples,
        threeC_waveforms,
        proc_fn,
        start,
        end,
        wf_source_list,
        padding,
        on_event=None,
    ):
        return services.Waveforms().gather_waveforms(
            session,
            n_samples=n_samples,
            threeC_waveforms=threeC_waveforms,
            channel_index_mapping_fn=get_channel_data_index,
            wf_process_fn=proc_fn,
            phase=self.phase,
            start=start,
            end=end,
            sources=wf_source_list,
            include_multiple_wf_sources=False,
            pad_samples=padding,
            on_event=on_event,
        )

    def save_corrections(
        self,
        session,
        all_ids,
        ensemble_outputs,
        summary_stats,
        cal_results,
        start_date,
        end_date,
        on_event=None,
    ):
        n_wfs = ensemble_outputs.shape[0]
        n_picks = ensemble_outputs.shape[1]
        # Open the pytable to store the predictions
        storage = self._open_picker_output_storage(
            n_picks, n_wfs, start_date, end_date, on_event
        )

        try:
            storage.start_transaction()
            # Iterate over the picks
            for i in np.arange(0, n_wfs):
                i_ids = all_ids[i]
                # Save the prediction results
                corr = services.insert_pick_correction_pytable(
                    session=session,
                    storage=storage,
                    pick_id=i_ids["pick_id"],
                    method_id=self.repicker_method_id,
                    wf_source_id=i_ids["wf_source_id"],
                    median=summary_stats["median"][i],
                    mean=summary_stats["mean"][i],
                    std=summary_stats["std"][i],
                    if_low=summary_stats["if_low"][i],
                    if_high=summary_stats["if_high"][i],
                    trim_median=summary_stats["trim_median"][i],
                    trim_mean=summary_stats["trim_mean"][i],
                    trim_std=summary_stats["trim_std"][i],
                    predictions=ensemble_outputs[i],
                )
                for perc in cal_results.keys():
                    # Add the CI information
                    _ = services.insert_ci(
                        session,
                        corr_id=corr.id,
                        method_id=self.calibration_method_id,
                        percent=perc,
                        lb=cal_results[perc]["arrivalTimeShiftLowerBound"][i],
                        ub=cal_results[perc]["arrivalTimeShiftUpperBound"][i],
                    )

        except Exception as e:
            storage.rollback()
            storage.close()
            raise e

        storage.commit()

    def _open_picker_output_storage(
        self, n_total_picks, n_wfs, start, end, on_event=None
    ):
        storage = pytables_backend.SwagPicksStorage(
            expected_array_length=n_total_picks,
            start=start,
            end=end,
            phase=self.phase,
            repicker_method_id=self.repicker_method_id,
            expectedrows=n_wfs,
            on_event=on_event,
        )
        return storage


def get_channel_data_index(ncomps, seed_code):
    # Get the appropriate channel index of the data
    if ncomps == 1:
        chan_ind = 0
    elif seed_code in ["EHZ", "BHZ", "HHZ"]:
        chan_ind = 2
    elif seed_code in ["EHE", "EH1", "BHE", "BH1", "HHE", "HH1"]:
        chan_ind = 0
    elif seed_code in ["EHN", "EH2", "BHN", "BH2", "HHN", "HH2"]:
        chan_ind = 1
    else:
        raise ValueError("Something is wrong with the channel code")

    return chan_ind
