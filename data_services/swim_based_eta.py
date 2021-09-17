#!/usr/bin/env python

"""Common code for inferring SWIM-based arrival time prediction and example
"""

import sqlalchemy
import pickle
import logging

import pandas as pd

DEFAULT_PARMS = {
    "minimum_time_tracked":0,
    }

log = logging.getLogger(__name__)

def build_swim_eta(
        tfms_data: pd.DataFrame,
        tbfm_data: pd.DataFrame,
        first_position_data: pd.DataFrame,
        parms=DEFAULT_PARMS,
        ) -> pd.DataFrame:
    """
    This function will compute the best available ETA based only on the SWIM
    data provided for arrivals to one airport.
    It is expected that input data are only provided at various instants when
    messages were received from various SWIM systems. This will require users
    to use an approximate merge (e.g., pandas.merge_asof() to determine which
    best available ETA applied at any instant in a different (e.g., model
    training) dataset.

    So, for example, the inputs to this function may be at times 0, 10, and 20.
    If the training data for which lookaheads are being computed has rows at
    0, 8, 12, 16, and 23, then an approximate merge would use the predictions
    computed at 0, 0, 10, 10, and 20 for these rows, respectively. If, however,
    the ETA value at time 0 was 7, then at time 8 in the training dataset this
    best available time is now in the past. The reccommended practice in this
    case is to update the best available ETA to max(current_time, ETA value).
    Parameters
    ----------
    tfms_data : pd.DataFrame
        Data for TFMS ETA updates. Required columns are:
        gufi: flight identifier [object]
        timestamp: time at which updated ETA provided [timestamp]
        arrival_runway_estimated_data: ETA value [timestamp]
    tbfm_data : pd.DataFrame
        Consolidated view of all relevant TBFM data. Required columns are:
        gufi: flight identifier [object]
        timestamp: time at which update was provided [timestamp]
        arrival_runawy_sta: STA value [timestamp]
    first_position_data : pd.DataFrame
        Data about first time flight observed under surveillance. Required
        columns are:
        gufi: flight identifier [object]
        time_first_tracked: first time with surveillance data [timestamp]
    parms : dict
        Configuration parameters for the heuristic. Required keys are:
        minimum_time_tracked

    Returns
    -------
    res : pd.DataFrame
        Results indicating best estimate for ETA at various instants.
        Will not provide continuous coverage, so should use pd.merge_asof() to
        have estimate roll forward in other datasets. Columns are:
        gufi: flight identifier [object]
        timestamp: time at which prediction first applies [timestamp]
        arrival_runway_best_time: ETA value [timestamp]

    """

    # Prepare data
    tfms_data = (
        tfms_data
        .sort_values("timestamp")
        .rename(columns={
            "arrival_runway_estimated_time":"tfms_eta",
            })
        )
    tbfm_data = (
        tbfm_data
        .sort_values("timestamp")
        .rename(columns={
            "arrival_runway_eta":"tbfm_eta",
            "arrival_runway_sta":"tbfm_sta",
            })
        )

    # Add rows to ensure catching when TFMS ETA goes into the past
    tfms_data["next_timestamp"] = (
        tfms_data
        .groupby("gufi")["timestamp"]
        .shift(-1)
        )

    rows_to_add = (
        tfms_data
        .loc[
            (tfms_data["tfms_eta"] > tfms_data["timestamp"])
            & (tfms_data["tfms_eta"] < tfms_data["next_timestamp"]),
            :
            ]
        .copy()
        .drop(columns="next_timestamp")
        )
    rows_to_add["timestamp"] = rows_to_add["tfms_eta"]

    tfms_data = tfms_data.drop(columns="next_timestamp")

    # Build time series against which to join and drop duplicates
    base = (
        pd
        .concat([
            tfms_data.loc[:, ["gufi", "timestamp"]],
            tbfm_data.loc[:, ["gufi", "timestamp"]],
            rows_to_add.loc[:, ["gufi", "timestamp"]],
            ])
        .drop_duplicates()
        .sort_values("timestamp")
        )
    log.info(
        "After initial time series creation\n"
        + "\n".join([f"{k}\t{base.dtypes[k].name}" for k in base.dtypes.keys()])
        )

    # Merge TFMS data against main time series, carrying last value forward
    if tfms_data.shape[0] > 0:
        base = (
            pd
            .merge_asof(
                base,
                tfms_data,
                by="gufi",
                on="timestamp",
                allow_exact_matches=True,
                direction="backward",
                )
            )
    else:
        log.info("No TFMS ETA data found")
        base["tfms_eta"] = pd.NaT

    log.info(
        "After adding TFMS data\n"
        + "\n".join([f"{k}\t{base.dtypes[k].name}" for k in base.dtypes.keys()])
        )

    # Merge TBFM data against main time series, carrying last value forward
    if tbfm_data.shape[0] > 0:
        base = (
            pd
            .merge_asof(
                base,
                tbfm_data,
                by="gufi",
                on="timestamp",
                allow_exact_matches=True,
                direction="backward",
                )
            )
    else:
        log.info("No TBFM STA data found")
        base["tbfm_sta"] = pd.NaT

    log.info(
        "After adding TBFM data\n"
        + "\n".join([f"{k}\t{base.dtypes[k].name}" for k in base.dtypes.keys()])
        )

    # Merge first position times
    if first_position_data.shape[0] > 0:
        base = (
            pd
            .merge(
                base,
                first_position_data,
                on="gufi",
                how="left",
                )
            )
    else:
        log.info("No first time tracked data found")
        base["time_first_tracked"] = pd.NaT

    log.info(
        "After adding first position data\n"
        + "\n".join([f"{k}\t{base.dtypes[k].name}" for k in base.dtypes.keys()])
        )

    # Pre-compute some indicators
    base["time_since_first_tracked"] = (
        (base["timestamp"] - base["time_first_tracked"])
        .dt
        .total_seconds()
        )
    log.info(
        "After computing time since tracked\n"
        + "\n".join([f"{k}\t{base.dtypes[k].name}" for k in base.dtypes.keys()])
        )

    # Create a base value to overwrite, should better values be available
    base["arrival_runway_best_time"] = pd.NaT
    log.info(
        "After initializing arrival_runway_best_time\n"
        + "\n".join([f"{k}\t{base.dtypes[k].name}" for k in base.dtypes.keys()])
        )

    # Fill first with TFMS ETA
    idx1 = (
        base["tfms_eta"].notnull()
        )
    base.loc[idx1, "arrival_runway_best_time"] = base.loc[idx1, "tfms_eta"]
    log.info(
        "After step1\n"
        + "\n".join([f"{k}\t{base.dtypes[k].name}" for k in base.dtypes.keys()])
        )

    # Next, fill with TBFM STA, if conditions allow
    idx2 = (
        (base["time_since_first_tracked"] > parms["minimum_time_tracked"])
        & base["tbfm_sta"].notnull()
        & (base["tbfm_sta"] > base["timestamp"])
        )
    base.loc[idx2, "arrival_runway_best_time"] = base.loc[idx2, "tbfm_sta"]
    log.info(
        "After step2\n"
        + "\n".join([f"{k}\t{base.dtypes[k].name}" for k in base.dtypes.keys()])
        )

    # Next, fill with TFMS ETA, if it is in the past
    idx3 = (
        base["tfms_eta"].notnull()
        & (base["tfms_eta"] <= base["timestamp"])
        )
    base.loc[idx3, "arrival_runway_best_time"] = base.loc[idx3, "tfms_eta"]
    log.info(
        "After step3\n"
        + "\n".join([f"{k}\t{base.dtypes[k].name}" for k in base.dtypes.keys()])
        )

    # Select results and remove duplicates
    res = (
        base
        .loc[:,
             ["gufi",
              "timestamp",
              "arrival_runway_best_time",
              ]]
        .drop_duplicates()
        )
    log.info(
        "Final state\n"
        + "\n".join([f"{k}\t{res.dtypes[k].name}" for k in res.dtypes.keys()])
        )

    return res

def sample_usage():
    query_params = {
        "start_time":"2019-09-01 09:00",
        "end_time":"2019-09-02 09:00",
        "airport":"KIAH",
        }

    tfms_query = """
        select
          gufi,
          "timestamp",
          arrival_runway_estimated_time
        from matm_flight
        where "timestamp" between (timestamp :start_time - interval '12 hours') and :end_time
          and arrival_aerodrome_icao_name = :airport
          and last_update_source = 'TFM'
          and arrival_runway_estimated_time is not null
        """

    tbfm_query = """
        select
          tea.gufi,
          tea."timestamp",
          tea.arrival_runway_sta
        from tbfm_extension_all tea
        left join matm_flight_summary mfs on tea.gufi = mfs.gufi
        where tea."timestamp" between :start_time and :end_time
          and mfs.arrival_aerodrome_icao_name = :airport
        """

    first_position_query = """
        select
          gufi,
          min("timestamp") as time_first_tracked
        from matm_flight
        where "timestamp" between :start_time and :end_time
          and arrival_aerodrome_icao_name = :airport
          and position_latitude is not null
          and last_update_source in ('TFM', 'TMA')
        group by gufi
        """

    engine = sqlalchemy.create_engine(
        "postgresql://username1:password1@host1/dbname1"
        )
    with engine.connect() as conn:
        tfms_data = pd.read_sql_query(
            sqlalchemy.text(tfms_query),
            conn,
            params=query_params,
            )
        tbfm_data = pd.read_sql_query(
            sqlalchemy.text(tbfm_query),
            conn,
            params=query_params,
            )
        first_position_data = pd.read_sql_query(
            sqlalchemy.text(first_position_query),
            conn,
            params=query_params,
            )

    # # Add a simple cache for this data to avoid running the queries each time
    # sample_data = {
    #     "tfms_data":tfms_data,
    #     "tbfm_data":tbfm_data,
    #     "first_position_data":first_position_data,
    #     }
    # with open("swim_based_eta_example_data.pickle", "wb") as f:
    #     pickle.dump(sample_data, f)
        
    # with open("swim_based_eta_example_data.pickle", "rb") as f:
    #     raw_dat = pickle.load(f)
    # tfms_data = raw_dat["tfms_data"]
    # tbfm_data = raw_dat["tbfm_data"]
    # first_position_data = raw_dat["first_position_data"]

    best_eta = build_swim_eta(
        tfms_data,
        tbfm_data,
        first_position_data,
        )

    best_eta.to_csv("best_eta_sample.csv", index=False)
