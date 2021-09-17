#!/usr/bin/env python

"""Common functions for sampling in time series
"""

import pandas as pd

from typing import List

def _count_in_ts(
        current_time: pd.Series,
        event_time: pd.Series,
        ts_data: pd.DataFrame,
        field: str,
        windows: List[pd.Timedelta],
        actuals=None,
        ) -> pd.DataFrame:
    """
    Compute a time series representation of expected counts, given frequently-
        updated predictions of some future event.

    Parameters
    ----------
    current_time : pd.Series
        The time at which the computation for each flight is being performed
        Indexing should match event_time
    event_time : pd.Series
        The time at which the flight is expected to operate, used to determine
        when to sample into the time series of events
        Indexing should match current_time
    ts_data : pd.DataFrame
        Time series data on which count will be performed. Required fields are
        gufi: flight with which prediction is associated
        timestamp: time at which prediction was provided
        (field): prediction time
    field : str
        Name of field in ts_data to use
    windows : list(pd.Timedelta)
        Time windows over which to count. Positive values count predictions
        after the sampling time, while negative values count predictions before
        the sampling time
    actuals : pd.Series (optional)
        Actual operation times, if available. If provided, these will be
        combined with ts_data so that counts for windows overlapping 'now' will
        include both predicted and actual events

    Returns
    -------
    res : pd.DataFrame
        Results of counting computation, with number of fields matching length
        of window input. Fields named as 'count_X' where X is a value from
        window.
        Indexing will match current_time
    """

    # Determine dates that must be processed to support pre-computation
    if (len(current_time) != len(event_time)):
        raise(Exception("Length of inputs inconsistent"))

    sampling_ins = pd.DataFrame(
        data={
            "current_time":current_time.values,
            "event_time":event_time.values,
            },
        index=current_time.index,
        )
    date_in_sampling_ins = sampling_ins["current_time"].dt.date

    # Sort all predictions ahead of time
    ts_data = ts_data.sort_values(by=[
        "gufi",
        "timestamp",
        field,
        ])

    # Helper function to get latest predictions and count number in windows
    def _compute_counts(row, dat, field, windows):
        sub_dat2 = (
            dat
            .loc[
                (dat["timestamp"] <= row["current_time"]),
                :
                ]
            .groupby("gufi")
            .tail(1)
            )

        vals = list()
        for w in windows:
            if (w.total_seconds() > 0):
                left = row["event_time"]
                right = row["event_time"] + w
            else:
                left = row["event_time"] + w
                right = row["event_time"]

            val = (
                sub_dat2[field]
                .between(left, right)
                .sum()
                )
            vals.append(val)

        return vals

    # Loop over all dates present in input data
    rev_buffer = pd.to_timedelta("6H")
    fwd_buffer = pd.to_timedelta("30H")
    vals = list()
    for d in date_in_sampling_ins.unique():
        # Get subset of entire time series to work on
        sub_ts_data = ts_data.loc[
            ts_data["timestamp"].between(
                pd.to_datetime(d) - rev_buffer,
                pd.to_datetime(d) + fwd_buffer
                ),
            :]

        # Get subset of sampling instructions to work on
        sub_sampling_ins = sampling_ins[date_in_sampling_ins == d]

        # Execute computation
        # Results will be indexed as sub_sampling_ins were, which will match
        #   how the original input data were indexed
        res = sub_sampling_ins.apply(
            _compute_counts,
            axis=1,
            args=(sub_ts_data, field, windows,),
            result_type="expand",
            )
        vals.append(res)

    # Combine results of each day's computation and assign sensible column
    #   names
    all_res = pd.concat(vals)
    all_res.columns = [
        "count_{:.0f}".format(w.total_seconds()) for w in windows
        ]

    return all_res

def _sample_forecast(
        current_time: pd.Series,
        event_time: pd.Series,
        forecast_data: pd.DataFrame,
        max_lookback=pd.to_timedelta("4 hour"),
        ) -> pd.DataFrame:
    """
    Lookup the most recently provided forecast value applicable for some event

    This function will return the values specific to each event, e.g., in
        effect for each flight.

    Parameters
    ----------
    current_time : pd.Series
        The time at which the lookup for each flight is being performed, used
        to determine which forecast was most-recently provided.
        Indexing should match event_time
    event_time : pd.Series
        The time at which the flight is expected to operate, used to determine
        which future time period in the forecast applies.
        Indexing should match current_time
    forecast_data : pd.DataFrame
        Weather data into which lookup is being performed. Required columns are
        posted_timestamp: the time at which the forecast was published
        forecast_timestamp: the time at which the forecast applies
    max_lookback : pd.Timedelta (optional)
        Maximum time window to look backward from current time to find a
        forecast in effect
    Returns
    -------
    res : pd.DataFrame
        Results of performing lookup. Will be indexed to match current_time and
        event_time and include all fields in forecast_data
    """

    # Split up the weather data into smaller DataFrames so that the entire
    #   thing isn't scanned with each lookup, and pre-sort everything
    one_day = pd.to_timedelta("1 day")
    prepped_data = dict()
    for d in current_time.dt.date.unique():
        prepped_data[d] = (
            forecast_data
            .loc[
                forecast_data["posted_timestamp"].between(
                    pd.to_datetime(d) - max_lookback,
                    pd.to_datetime(d) + one_day
                    ),
                :
                ]
            .sort_values(
                by=["posted_timestamp", "forecast_timestamp"],
                )
            )

    # Create a DataFrame over which to iterate when performing lookup
    combined_data = pd.DataFrame({
        "current_time":current_time,
        "event_time":event_time,
        })

    # Helper function to perform lookup of forecast data
    def _do_lookup(row, dat):
        # Get pre-computed data from main dictionary
        sub_dat = dat[row["current_time"].date()]

        # Get last row from pre-computed data that was provided before lookup
        #   time that is applicable for event time
        # Append the empty row in case no valid rows are identified in lookup
        #   data
        sub_dat2 = (
            sub_dat
            .loc[
                (sub_dat["posted_timestamp"] <= row["current_time"])
                & (sub_dat["forecast_timestamp"] <= row["event_time"]),
                :
                ]
            .tail(1)
            .append(pd.Series(dtype=float), ignore_index=True)
            .iloc[0]
            )

        return sub_dat2

    # Actually perform the computation
    res = combined_data.apply(
        _do_lookup,
        args=(prepped_data,),
        axis=1,
        )

    # Appending the empty row to protect against missing lookup data breaks
    #   data types, so apply the data types originally associated with each
    #   column in the input data
    for col in forecast_data.columns:
        res[col] = res[col].astype(forecast_data[col].dtype)

    return res


def _predicted_lookahead_equals_value(
    data: pd.DataFrame,
    predicted_time_col_name: str,
    lookahead_values: List[pd.Timedelta],
    column_name_prefix='predicted_lookahead_eq_',
    max_lookback: pd.Timedelta=pd.to_timedelta('24 hour'),
) -> pd.DataFrame:
    """
    Find rows where the predicted lookahead time to some
    predicted future event time equals a certain value.
    Assumption is that as time progresses in the data input,
    new updates of the predicted future time will be provided.
    Such rows will be marked with a new Boolean column.

    Parameters
    ----------
    data : pd.DataFrame
        Main data set. Assumed to have these columns
        gufi: flight with which times and prediction are associated
        predicted_time_col_name: string provided in next parameter; this
        contains the predicted time of the future event
        timestamp: time at which the predicted future time was predicted
    predicted_time_col_name : str
        Name of the column with the datetime prediction of the
        future event time
    lookahead_values : List[pd.Timedelta]
        A list of lookahead values.
        Function will generate one new Boolean column for each.
        These are assumed to be at least one second apart from each other.
    column_name_prefix : str (default provided)
        The prefix that will be given to the new added columns.
        Suffix will be the pd.Timedelta in seconds
    max_lookback : pd.Timedelta (optional)
        Maximum time that a prediction of the future time remains "valid"
        after it is predicted.
    Returns
    -------
    data : pd.DataFrame
        Input data, but with a new Boolean column per lookahead_value.
        If the new column is True, this means that this was a prediction
        that led to the predicted lookahead equaling the corresponding value
        (possibly not at the instant the prediction was generated,
        but it was the valid prediction when the predicted lookahead hit
        lookahead_value and at that moment it was produced within
        max_lookback time).
    """
    # Filter down to only rows with a predicted_time
    data_filtered = data[data[predicted_time_col_name].notnull()].copy()

    # Add last timestamp per GUFI column to use in sorting
    max_timestamp_per_gufi = data_filtered[['gufi', 'timestamp']]\
        .groupby('gufi')\
        .max()\
        .rename(columns={'timestamp': 'gufi_max_timestamp'})
    data_filtered = data_filtered.join(
        max_timestamp_per_gufi,
        on='gufi',
    )

    # Sort by last timestamp (keep GUFIs ordered based on time), GUFI,
    # and then timestamp
    data_filtered = data_filtered.sort_values(
        [
            'gufi_max_timestamp',
            'gufi',
            'timestamp',
        ],
    )

    # Add column with seconds to predicted_time
    data_filtered['seconds_to_{}'.format(predicted_time_col_name)] = (
        (data_filtered[predicted_time_col_name] - data_filtered['timestamp'])
        .dt.total_seconds()
    )

    # Now add column for next update's (next row's) timestamp
    # seconds to predicted_time_col_name (if same GUFI)
    data_filtered['next_timestamp'] = (
        data_filtered
        .groupby("gufi")["timestamp"]
        .shift(periods=-1)
    )

    # Drop last timestamp per GUFI column
    data_filtered.drop(columns=['gufi_max_timestamp'], inplace=True)

    # Add column with seconds to next timestamp
    data_filtered[
        'seconds_to_{}_at_next_timestamp'.format(predicted_time_col_name)
    ] = (
        (
            data_filtered[predicted_time_col_name] -
            data_filtered['next_timestamp']
        ).dt.total_seconds()
    )

    # Then can find all rows where the seconds to best time will hit the
    # particular predicted lookahead
    # between when the prediction is made and when the next one is made
    # Add Boolean indicator column for such rows
    for lookahead_value in lookahead_values:
        lookahead_value_total_secs = lookahead_value.total_seconds()
        lookahead_value_col_name = column_name_prefix +\
            '{:.0f}'.format(lookahead_value_total_secs)
        # Default is False
        data_filtered[lookahead_value_col_name] = False
        # If seconds to predicted_time will hit the lookahead_value
        # before the next update,
        # and not be "stale" (more than max_lookback old) when it does
        # flag this row as True
        data_filtered.loc[
            (
                data_filtered['seconds_to_{}'.format(predicted_time_col_name)]
                >= lookahead_value_total_secs
            ) &
            (
                data_filtered[
                    'seconds_to_{}_at_next_timestamp'.format(
                        predicted_time_col_name
                    )
                ]
                < lookahead_value_total_secs
            ) &
            (
                (
                    data_filtered[
                        'seconds_to_{}'.format(predicted_time_col_name)
                    ] -
                    lookahead_value_total_secs
                )
                <= max_lookback.total_seconds()
            ),
            lookahead_value_col_name
        ] = True

    # Join results from filtered data set back into the input data
    # index in data_filtered should be the same as that in data
    data = data.join(
        data_filtered[[
            column_name_prefix + '{:.0f}'.format(
                lookahead_value.total_seconds()
            )
            for lookahead_value in lookahead_values
        ]]
    )

    # Fill in missings in new columns with False
    for lookahead_value in lookahead_values:
        data[[
            column_name_prefix + '{:.0f}'.format(
                lookahead_value.total_seconds()
            )
        ]] = data[[
            column_name_prefix + '{:.0f}'.format(
                lookahead_value.total_seconds()
            )
        ]].fillna(
            value=False,
        )

    return data


def _actual_lookahead_equals_value(
    data: pd.DataFrame,
    actual_time_col_name: str,
    lookahead_values: List[pd.Timedelta],
    column_name_prefix='actual_lookahead_eq_',
    max_lookback: pd.Timedelta=pd.to_timedelta('24 hour'),
) -> pd.DataFrame:
    """
    Find rows from a time series where the actual lookahead time to some
    future event time equals a certain value.
    Such rows will be marked with a new Boolean column.

    Parameters
    ----------
    data : pd.DataFrame
        Main data set. Assumed to have these columns
        gufi: flight with which times and other rows are associated
        actual_time_col_name: string provided in next parameter; this
        contains the actual time of the future event
        timestamp: time at which the row was produced (assumed valid
        until next row's timestamp)
    actual_time_col_name : str
        Name of the column with the datetime of the
        future event time
    lookahead_values : List[pd.Timedelta]
        A list of lookahead values.
        Function will generate one new Boolean column for each.
        These are assumed to be at least one second apart from each other.
    column_name_prefix : str (default provided)
        The prefix that will be given to the new added columns.
        Suffix will be the pd.Timedelta in seconds
    max_lookback : pd.Timedelta (optional)
        Maximum time that a prediction of the future time remains "valid"
        after it is predicted.
    Returns
    -------
    data : pd.DataFrame
        Input data, but with a new Boolean column per lookahead_value.
        If the new column is True, this means that this was the row
        valid lookahead equaling the corresponding value
        (possibly not at the instant the row was generated,
        but it was the valid row when the actual lookahead hit
        lookahead_value and at that moment it was produced within
        max_lookback time).
    """
    # Filter down to only rows with an actual_time
    data_filtered = data[data[actual_time_col_name].notnull()].copy()

    # Add last timestamp per GUFI column to use in sorting
    max_timestamp_per_gufi = data_filtered[['gufi', 'timestamp']]\
        .groupby('gufi')\
        .max()\
        .rename(columns={'timestamp': 'gufi_max_timestamp'})
    data_filtered = data_filtered.join(
        max_timestamp_per_gufi,
        on='gufi',
    )

    # Sort by last timestamp (keep GUFIs ordered based on time), GUFI,
    # and then timestamp
    data_filtered = data_filtered.sort_values(
        [
            'gufi_max_timestamp',
            'gufi',
            'timestamp',
        ],
    )

    # Add column with seconds to predicted_time
    data_filtered['seconds_to_{}'.format(actual_time_col_name)] = (
        (data_filtered[actual_time_col_name] - data_filtered['timestamp'])
        .dt.total_seconds()
    )

    # Now add column for next update's (next row's) timestamp
    # seconds to predicted_time_col_name (if same GUFI)
    data_filtered['next_timestamp'] = (
        data_filtered
        .groupby('gufi')['timestamp']
        .shift(periods=-1)
    )

    # Drop last timestamp per GUFI column
    data_filtered.drop(columns=['gufi_max_timestamp'], inplace=True)

    # Add column with seconds to next timestamp
    data_filtered[
        'seconds_to_{}_at_next_timestamp'.format(actual_time_col_name)
    ] = (
        (
            data_filtered[actual_time_col_name] -
            data_filtered['next_timestamp']
        ).dt.total_seconds()
    )

    # Then can find all rows where the seconds to best time will hit the
    # particular predicted lookahead
    # between when the prediction is made and when the next one is made
    # Add Boolean indicator column for such rows
    for lookahead_value in lookahead_values:
        lookahead_value_total_secs = lookahead_value.total_seconds()
        lookahead_value_col_name = column_name_prefix +\
            '{:.0f}'.format(lookahead_value_total_secs)
        # Default is False
        data_filtered[lookahead_value_col_name] = False
        # If seconds to predicted_time will hit the lookahead_value
        # before the next update,
        # and not be "stale" (more than max_lookback old) when it does
        # flag this row as True
        data_filtered.loc[
            (
                data_filtered['seconds_to_{}'.format(actual_time_col_name)]
                >= lookahead_value_total_secs
            ) &
            (
                data_filtered[
                    'seconds_to_{}_at_next_timestamp'.format(
                        actual_time_col_name
                    )
                ]
                < lookahead_value_total_secs
            ) &
            (
                (
                    data_filtered[
                        'seconds_to_{}'.format(actual_time_col_name)
                    ] -
                    lookahead_value_total_secs
                )
                <= max_lookback.total_seconds()
            ),
            lookahead_value_col_name
        ] = True

    # Join results from filtered data set back into the input data
    # index in data_filtered should be the same as that in data
    data = data.join(
        data_filtered[[
            column_name_prefix + '{:.0f}'.format(
                lookahead_value.total_seconds()
            )
            for lookahead_value in lookahead_values
        ]]
    )

    # Fill in missings in new columns with False
    for lookahead_value in lookahead_values:
        data[[
            column_name_prefix + '{:.0f}'.format(
                lookahead_value.total_seconds()
            )
        ]] = data[[
            column_name_prefix + '{:.0f}'.format(
                lookahead_value.total_seconds()
            )
        ]].fillna(
            value=False,
        )

    return data

