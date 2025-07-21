import pandas as pd
import os
from datetime import datetime, timedelta, timezone
from seis_proc_db import services, tables
from seis_proc_db.database import Session


def insert_assoc_loc_db():
    assoc_method_name = "massociate"
    loc_method_name = "uLocator"
    loc_method_desc = "Real-time event locator by Ben Baker"
    basedir = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/process_ys_data/assoc_loc_io"
    yeardir = "2024_assoc06"
    start_date = "2024-01-01"
    end_date = "2025-01-01"

    dateformat = "%Y-%m-%d"
    curr_date = datetime.strptime(start_date, dateformat)
    last_date = datetime.strptime(end_date, dateformat)
    delta = timedelta(days=1)
    auth = "SPDL"

    indir = os.path.join(basedir, yeardir)

    with Session() as session:
        with session.begin():
            assoc_meth = services.get_assoc_method(session, assoc_method_name)
            assoc_method_id = assoc_meth.id
            loc_method = services.get_loc_method(session, loc_method_name)
            if loc_method is None:
                loc_method = services.insert_loc_method(
                    session, loc_method_name, loc_method_desc
                )
                session.flush()
            loc_method_id = loc_method.id

    events_to_insert = []
    while curr_date < last_date:
        print("Reading data for", curr_date)
        assoc_file = os.path.join(
            indir,
            "assoc_out",
            f"{curr_date.strftime(dateformat)}.associatedEvents.csv",
        )
        loc_file = os.path.join(
            indir,
            "loc_out",
            f"{curr_date.strftime(dateformat)}.locatedEvents.csv",
        )

        assoc_df = None
        if os.path.isfile(assoc_file):
            assoc_df = pd.read_csv(assoc_file)
        else:
            print("No assoc file found, skipping day...")
            curr_date += delta
            continue

        loc_df = None
        if os.path.isfile(loc_file):
            loc_df = pd.read_csv(loc_file)

        assoc_ev_df = assoc_df[
            [
                "event_identifier",
                "event_type",
                "origin_time",
                "event_latitude",
                "event_longitude",
                "event_depth",
                "event_type.1",
            ]
        ].drop_duplicates()

        for _, asor_row in assoc_ev_df.iterrows():
            is_trigger = False
            if asor_row["event_type.1"] == "trigger":
                is_trigger = True
            event = tables.Event(auth=auth, is_trigger=is_trigger)
            assoc_arr_df = assoc_df[
                assoc_df["event_identifier"] == asor_row["event_identifier"]
            ]
            asor = tables.AssocOrigin(
                assocm_id=assoc_method_id,
                lat=asor_row["event_latitude"],
                lon=asor_row["event_longitude"],
                depth=asor_row["event_depth"],
                ot=datetime.fromtimestamp(asor_row["origin_time"], tz=timezone.utc),
                narrs=len(assoc_arr_df),
            )
            for _, asarr_row in assoc_arr_df.iterrows():
                asarr = tables.AssocArrival(
                    pick_id=asarr_row["arrival_identifier"],
                    arrtime=datetime.fromtimestamp(
                        asarr_row["arrival_time"], tz=timezone.utc
                    ),
                    std_err=asarr_row["standard_error"],
                    aphase=asarr_row["phase"],
                    residual=asarr_row["residual"],
                    travel_time=asarr_row["travel_time"],
                )
                asor.assoc_arrs.append(asarr)

            locarr_df = None
            if loc_df is not None:
                locarr_df = loc_df[
                    loc_df["event_identifier"] == asor_row["event_identifier"]
                ]
            if locarr_df is not None and len(locarr_df) > 0:
                locor_row = locarr_df.iloc[0]
                locor = tables.Origin(
                    locm_id=loc_method_id,
                    lat=locor_row["latitude"],
                    lon=locor_row["longitude"],
                    depth=locor_row["depth"],
                    ot=locor_row["origin_time"],
                    weighted_rmse=locor_row["weightedRMSE"],
                    narrs=len(locarr_df),
                    min_dist=locarr_df["source_receiver_distance"].min(),
                )
                for _, locarr_row in locarr_df.iterrows():
                    locarr = tables.Arrival(
                        pick_id=locarr_row["arrival_identifier"],
                        arrtime=locarr_row["arrival_time"],
                        aphase=locarr_row["phase"],
                        std_err=locarr_row["standard_error"],
                        residual=locarr_row["residual"],
                        sr_dist=locarr_row["source_receiver_distance"],
                    )
                    locor.loc_arrs.append(locarr)

                locor.assoc_origin = asor
                event.origins.append(locor)

            event.assoc_origins.append(asor)
            events_to_insert.append(event)

        curr_date += delta

    print("Writing to db...")
    with Session() as session:
        with session.begin():
            session.add_all(events_to_insert)


def make_input_files():
    repicker_name = "MSWAG-Armstrong2023"
    cal_name = "Kuleshov2018-Armstrong2023"
    p_repicker_method = f"P-{repicker_name}"
    s_repicker_method = f"S-{repicker_name}"
    p_calibration_method = f"P-{cal_name}"
    s_calibration_method = f"S-{cal_name}"
    assoc_method_name = "massociate"
    assoc_method_desc = (
        "Migration-based association of differential pick times by Ben Baker."
    )
    start_date = "2024-01-01"
    end_date = "2025-01-01"
    p_max_width = 0.30
    s_max_width = 0.40
    p_min_width = 0.150
    s_min_width = 0.250
    ci_perc = 90
    base_outdir = "/uufs/chpc.utah.edu/common/home/koper-group3/alysha/process_ys_data/assoc_loc_io/"

    dateformat = "%Y-%m-%d"
    curr_date = datetime.strptime(start_date, dateformat)
    last_date = datetime.strptime(end_date, dateformat)
    print("Getting Picks for", curr_date, "to", last_date)
    delta = timedelta(days=1)
    slurm_array_list = []

    with Session() as session:
        with session.begin():
            assoc_meth = services.get_assoc_method(session, assoc_method_name)
            if assoc_meth is None:
                assoc_meth = services.insert_assoc_method(
                    session,
                    assoc_method_name,
                    assoc_method_desc,
                    p_min_width=p_min_width,
                    p_max_width=p_max_width,
                    s_min_width=s_min_width,
                    s_max_width=s_max_width,
                    repicker_name=repicker_name,
                    cal_name=cal_name,
                    ci_perc=ci_perc,
                )
                session.flush()
            elif (
                assoc_meth.details != assoc_method_desc
                or assoc_meth.p_min_ci_width != p_min_width
                or assoc_meth.p_max_ci_width != p_max_width
                or assoc_meth.s_min_ci_width != s_min_width
                or assoc_meth.s_max_ci_width != s_max_width
                or assoc_meth.ci_perc != ci_perc
                or assoc_meth.repicker_name != repicker_name
                or assoc_meth.cal_name != cal_name
            ):
                raise ValueError(
                    "AssocMethod exists already exists but the row values have changed..."
                )

            assoc_meth_id = assoc_meth.id

            while curr_date < last_date:
                curr_date_str = curr_date.strftime(dateformat)
                next_date_str = (curr_date + delta).strftime(dateformat)
                print("Working on", curr_date.strftime(dateformat))

                print("Gathering data from the database...")
                p_pick_df = services.make_pick_catalog_df(
                    session,
                    "P",
                    p_repicker_method,
                    p_calibration_method,
                    ci_perc,
                    start=curr_date_str,
                    end=next_date_str,
                    max_width=p_max_width,
                    min_width=p_min_width,
                )
                s_pick_df = services.make_pick_catalog_df(
                    session,
                    "S",
                    s_repicker_method,
                    s_calibration_method,
                    ci_perc,
                    start=curr_date_str,
                    end=next_date_str,
                    max_width=s_max_width,
                    min_width=s_min_width,
                )

                all_pick_df = (
                    pd.concat([p_pick_df, s_pick_df])
                    .sort_values("arrival_time")
                    .round({"uncertainty": 3})
                )

                print("Saving dfs...")
                outdir = f"{curr_date.year}_assoc{assoc_meth_id:02d}"
                pick_dir = os.path.join(base_outdir, outdir, "picks")
                stat_dir = os.path.join(base_outdir, outdir, "stations")
                assoc_dir = os.path.join(base_outdir, outdir, "assoc_out")
                loc_dir = os.path.join(base_outdir, outdir, "loc_out")

                if not os.path.exists(pick_dir):
                    os.makedirs(pick_dir)

                if not os.path.exists(stat_dir):
                    os.makedirs(stat_dir)

                if not os.path.exists(assoc_dir):
                    os.makedirs(assoc_dir)

                if not os.path.exists(loc_dir):
                    os.makedirs(loc_dir)

                day_pick_df = all_pick_df[
                    [
                        "pick_identifier",
                        "network",
                        "station",
                        "channel",
                        "location_code",
                        "phase_hint",
                        "arrival_time",
                        "uncertainty",
                    ]
                ]
                day_station_df = (
                    all_pick_df[
                        [
                            "network",
                            "station",
                            "latitude",
                            "longitude",
                            "elevation",
                        ]
                    ]
                    .drop_duplicates()
                    .sort_values(["network", "station"])
                )

                pick_file = os.path.join(
                    pick_dir, f"{curr_date.strftime(dateformat)}.picks.csv"
                )
                day_pick_df.to_csv(pick_file, index=False)
                stat_file = os.path.join(
                    stat_dir, f"{curr_date.strftime(dateformat)}.stations.csv"
                )
                day_station_df.to_csv(stat_file, index=False)

                assoc_file = os.path.join(
                    assoc_dir, f"{curr_date.strftime(dateformat)}.associatedEvents.csv"
                )
                loc_file = os.path.join(
                    loc_dir, f"{curr_date.strftime(dateformat)}.locatedEvents.csv"
                )

                slurm_array_list.append([pick_file, stat_file, assoc_file, loc_file])

                curr_date += delta

    slurm_array_df = pd.DataFrame(slurm_array_list)
    slurm_array_file = os.path.join(
        base_outdir, outdir, f"slurm_array_input_{start_date}_{end_date}.txt"
    )
    with open(slurm_array_file, "w") as f:
        f.write(str(len(slurm_array_list)) + "\n")

    slurm_array_df.to_csv(
        slurm_array_file, mode="a", index=False, header=False, sep=" "
    )


if __name__ == "__main__":
    # make_input_files()
    insert_assoc_loc_db()
