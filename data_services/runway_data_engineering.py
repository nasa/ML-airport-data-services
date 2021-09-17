""" Common code for performing data engineering in arrival / departure models
"""

import logging
import os

import pandas as pd
import numpy as np

from typing import Dict, Any, Union, List
from datetime import date

from kedro.extras.datasets.pandas import CSVDataSet

log = logging.getLogger(__name__)

def df_join(
        data_0: pd.DataFrame,
        data_1: pd.DataFrame,
        join_kwargs: Dict[str, Any]={},
        ) -> pd.DataFrame:
    """
    Parameters
    ----------
    data_0 : pd.DataFrame
        DESCRIPTION.
    data_1 : pd.DataFrame
        DESCRIPTION.
    join_kwargs : Dict[str, Any], optional
        DESCRIPTION. The default is {}.

    Returns
    -------
    res : pd.DataFrame
        DESCRIPTION.

    """
    res = (
        pd
        .merge(
            data_0,
            data_1,
            **join_kwargs,
            )
        )

    return res

def infer_wake_categories(
        df: pd.DataFrame,
        aircraft_types_xml: pd.DataFrame,
        ) -> pd.DataFrame:
    # Note: this is done here instead of as an encoder because the data service
    #   feeding this model live will provide this data directly.

    aircraft_types_xml = (
        aircraft_types_xml
        .loc[:, ["type", "recat_weight_class"]]
        .rename(columns={
            "type":"aircraft_type",
            "recat_weight_class":"wake_turbulence_category",
            })
        )

    df = (
        pd
        .merge(
            df,
            aircraft_types_xml,
            on="aircraft_type",
            how="left",
            )
        )

    return df

def prep_config_data(
    configs: pd.DataFrame,
) -> pd.DataFrame:

    configs["arrival_runways"] = (
        configs["arrival_runways"]
        .fillna(method="ffill")
    )

    configs["departure_runways"] = (
        configs["departure_runways"]
        .fillna(method="ffill")
    )

    configs["airport_configuration_name"] = (
        "D_"
        + configs["departure_runways"].astype(str).str.replace(", ", "_")
        + "_A_"
        + configs["arrival_runways"].astype(str).str.replace(", ", "_")
    )

    configs = configs.sort_values("config_start_time")

    configs["grp"] = (
        (configs["airport_configuration_name"]!=configs["airport_configuration_name"].shift())
        .cumsum()
        .fillna(0)
    )

    configs = (
        configs
        .drop_duplicates(
            subset=[
                "grp",
            ],
            keep="first",
        )
        .drop(
            columns=["grp"],
        )
    )

    return configs

def sort_timestamp_merge_asof(
        data_0: pd.DataFrame,
        data_1: pd.DataFrame,
        merge_asof_kwargs: Dict[str, Any]={
            "by": "gufi",
            "on": "timestamp",
            "allow_exact_matches": True,
            "direction": "backward",
        },
        ) -> pd.DataFrame:
    """
    Parameters
    ----------
    data_0 : pd.DataFrame
        DESCRIPTION.
    data_1 : pd.DataFrame
        DESCRIPTION.
    merge_asof_kwargs : Dict[str, Any], optional
        DESCRIPTION. The default is {"by":"gufi","on":"timestamp","allow_exact_matches":True,"direction":"backward"}.

    Returns
    -------
    data : pd.DataFrame
        DESCRIPTION.

    """
    data_0 = data_0.sort_values(merge_asof_kwargs["on" if "on" in merge_asof_kwargs else "left_on"])
    data_1 = data_1.sort_values(merge_asof_kwargs["on" if "on" in merge_asof_kwargs else "right_on"])

    data = pd.merge_asof(
        data_0,
        data_1,
        **merge_asof_kwargs,
        )

    return data

def add_difference_columns(
        data: pd.DataFrame,
        diff_cols: Dict[str, Any],
        ) -> pd.DataFrame:
    """
    Parameters
    ----------
    data : pd.DataFrame
        DESCRIPTION.
    diff_cols : Dict[str, Any]
        DESCRIPTION.

    Returns
    -------
    data : pd.DataFrame
        DESCRIPTION.

    """

    for col in diff_cols:
        log.info(f"Computing difference between {diff_cols[col][0]} ({data.dtypes[diff_cols[col][0]].name}) and {diff_cols[col][1]} ({data.dtypes[diff_cols[col][1]].name})")
        data[col] = (
            (data[diff_cols[col][0]] - data[diff_cols[col][1]])
            .dt.total_seconds()
            )

    return data

def select_tv_train_samples(
        data: pd.DataFrame,
        test_fraction: float,
        tv_timestep_fraction_train: float,
        random_seed: int,
        ) -> pd.DataFrame:
    """
    Parameters
    ----------
    data : pd.DataFrame
        Full dataset
    test_fraction: float
        Fraction of flights that we want in the test dataset, all others will
        be available for the training data.
    tv_timestep_fraction_train : float
        Fraction of training dataset to use for training. This is configurable,
        but should likely be less than 1.0 because of the huge number of rows.
    random_seed : int
        Random seed, if desired

    Returns
    -------
    data_grouped : pd.DataFrame
        Data with test / train group indicators

    """

    # Split all gufis into either test / train
    np.random.seed(random_seed)
    gufi_split = pd.DataFrame({
        "gufi": data["gufi"].unique(),
        })
    gufi_split["rand"] = np.random.uniform(size=gufi_split.shape[0])
    gufi_split["group"] = "train"
    gufi_split.loc[
        gufi_split["rand"] < test_fraction,
        "group"] = "test"

    # Join group indicator back
    data_grouped = pd.merge(
        data,
        gufi_split[["gufi", "group"]],
        on="gufi",
        how="left",
        )

    # Subsample in train group to reduce number of observations
    data_grouped["rand"] = np.random.uniform(size=data_grouped.shape[0])

    # Create indicator columns
    data_grouped["test_sample"] = (
        data_grouped["group"] == "test"
        )
    data_grouped["train_sample"] = (
        (data_grouped["group"] == "train")
        & (data_grouped["rand"] < tv_timestep_fraction_train)
        )

    data_grouped = data_grouped.drop(columns=["group", "rand"])

    return data_grouped

def drop_cols_not_in_inputs(
        data: pd.DataFrame,
        inputs: Dict[str, Any],
        target: Dict[str, Any],
        ) -> pd.DataFrame:
    """
    Parameters
    ----------
    data : pd.DataFrame
        DESCRIPTION.
    inputs : Dict[str, Any]
        DESCRIPTION.

    Returns
    -------
    data : pd.DataFrame
        DESCRIPTION.

    """
    keep_cols = ["gufi"] + [*inputs.keys()] + [target["name"]]
    col_list = [*data.columns]

    data = data.drop(
        columns=[
            col for col in col_list
            if col not in keep_cols
        ],
    )

    # Reorder per input specification
    data = data[keep_cols]

    return data

def drop_rows_with_missing_target(
        data: pd.DataFrame,
        target: Dict[str, Any],
        ) -> pd.DataFrame:
    idx = (
        data[target["name"]].notnull()
        & (data[target["name"]] != "")
    )
    log.info(f"Removed {data.shape[0] - idx.sum()} rows for missing target value")
    return data.loc[
        idx,
        :]

def apply_live_filtering(
    dat: pd.DataFrame,
    inputs: Dict[str, Any],
) -> pd.DataFrame:
    """
    Parameters
    ----------
    dat : pd.DataFrame
        Problem data
    inputs : Dict[str, Any]
        Input data specification from parameters.yml

    Returns
    -------
    dat : pd.DataFrame
        Data with live filtering rules applied

    This function is intended to mimic the filtering rules in the live system
    that are applied _before_ sending any data to the FilterPipeline model.
    For example, the live system doesn't even send flights outside the correct
    lookahead range to the model for results.
    This is a different kind of filtering than is applied inside the
    FilterPipeline, where the rules (e.g., to handle core features) are
    implemented.

    """
    keep_idx = pd.Series(
        data=True,
        index=dat.index,
        )

    # Remove lookahead outside range
    if ("lookahead" in dat.columns):
        if (("lookahead" in inputs)
            and ("constraints" in inputs["lookahead"])
            ):
            if ("min" in inputs["lookahead"]["constraints"]):
                keep_idx = (
                    keep_idx
                    & (dat["lookahead"] >= inputs["lookahead"]["constraints"]["min"])
                )
            else:
                log.info("Missing min keyword for lookahead")

            if ("max" in inputs["lookahead"]["constraints"]):
                keep_idx = (
                    keep_idx
                    & (dat["lookahead"] <= inputs["lookahead"]["constraints"]["max"])
                )
            else:
                log.info("Missing max keyword for lookahead")
        else:
            log.info("Malformed lookahead input spec")
    else:
        log.info("Missing lookahead column in data")

    return dat[keep_idx]

def clean_runway_names(
        dat: pd.DataFrame,
        col: Union[str, List[str]],
        ) -> pd.DataFrame:

    if isinstance(col, str):
        col = [col]

    for c in col:
        log.info(f"Cleaning runway names in column {c}")
        dat[c] = dat[c].astype(str).str.lstrip("0").str.rstrip(".0")

    return dat


def de_save(
    data: pd.DataFrame,
    params_globals: str,
    data_folder: str = './data/05_model_input/',
) -> None:

    if 'batch_mode' in params_globals:
        log.info("Batch mode detected")

        # Delete previous runs batch files for airport_icao
        if params_globals['start_time'] == params_globals['batch_mode']['run_start_time']:
            log.info("Found batch files from previous run, removing")
            files = os.listdir(data_folder)
            files = [f for f in files if
                     f[0:len(params_globals['airport_icao']) + 1] == params_globals['airport_icao'] + '_']
            for f in files:
                os.remove(data_folder + f)

        # Save current batch
        filepath=data_folder + params_globals['airport_icao'] + '_' + str(params_globals['start_time']) \
                     + '_' + str(params_globals['end_time']) + ".de_data_set.csv"
        log.info("Saving data to {}".format(filepath))
        data_set = CSVDataSet(filepath)
        data_set.save(data)

        # Concatenate all data in single file in last iteration, ds pipeline expecting single file
        if params_globals['end_time'] >= params_globals['batch_mode']['run_end_time']:
            log.info("Concatenating all data files")

            files = os.listdir(data_folder)
            files = [f for f in files if f[0:len(params_globals['airport_icao']) + 1] == params_globals['airport_icao'] + '_']

            log.info("Found {} files to load".format(len(files)))

            file_start_dates = [date.fromisoformat(f.split('_')[1]) for f in files]
            idx_sorted = np.argsort(file_start_dates)

            # Load data ordered
            de_data = []
            for idx in idx_sorted:
                with open(data_folder + files[idx], "rb") as f:
                    log.info("Loading data in {}".format(files[idx]))
                    de_data.append(pd.read_csv(f, low_memory=False, dtype="str"))

            log.info("Combining data from {} files".format(len(de_data)))

            # Concatenate all data, keep order and remove duplicates
            de_data = pd.concat(de_data, sort=False)
            log.info("Combined data has {} rows".format(de_data.shape[0]))

            # For duplicates, keep "last", the "first" duplicates from previous batch may not include all data due to
            # the end of batch
            de_data = de_data[de_data[['gufi','lookahead']].duplicated(keep='last') == False]
            log.info("After de-dupe, combined data has {} rows".format(de_data.shape[0]))

            # Save data
            data_set = CSVDataSet(
                filepath=data_folder + params_globals['airport_icao'] + ".de_data_set.csv")
            data_set.save(de_data)
    else:

        data_set = CSVDataSet(
            filepath=data_folder + params_globals['airport_icao'] + '_' + str(params_globals['start_time']) \
                     + '_' + str(params_globals['end_time']) + ".de_data_set.csv")
        data_set.save(data)

        # Save off a version without dates that serves as a starting point for DS pipeline
        data_set_no_dates = CSVDataSet(
            filepath=data_folder + params_globals['airport_icao'] + ".de_data_set.csv")
        data_set_no_dates.save(data)
