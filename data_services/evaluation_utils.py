"""
Functions that are helpful for evaluating models.

"""
from typing import Dict, Any, List

import pandas as pd
import numpy as np
from . import error_metrics


def evaluate_predictions(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    metrics_dict: Dict[str, Any] = {
        'mean_absolute_error':
        error_metrics.mean_absolute_error
    },
) -> pd.DataFrame:
    """
    Evaluate predictions and return evaluation results data frame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the 'group' column.
        Each group is evaluated separately
        and becomes a row in the output data frame.
        Same index as y_true and y_pred.
    y_true : pd.Series
        Ground truth labels.
        Same index as df and y_pred.
    y_pred : pd.Series
        Predictions. Okay if contains null.
        Same index as y_true and df.
    metrics_dict : Dict[str, Any], default only has MAE
        Dictionary of metric names & functions to use for evaluation.
        These become columns in output data frame.

    Returns
    -------
    pd.DataFrame with evaluation results.
    """
    evaluation_df = pd.DataFrame(
        index=df.group.unique(),
    )

    for metric_name, metric_func in metrics_dict.items():
        if metric_name == 'percent_within_n':
            continue  # Handled by separate function

        evaluation_df[metric_name] = None

        for group in df.group.unique():
            evaluation_df.loc[group, metric_name] =\
                metric_func(
                    df.loc[df.group == group, y_true],
                    df.loc[df.group == group, y_pred],
                )

    return evaluation_df


def calc_percent_within_n_df(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    ns: List[int] = [10, 30, 60],
) -> pd.DataFrame:
    """
    Calculate percent within N metrics for regression model.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the 'group' column.
        Each group is evaluated separately
        and becomes a row in the output data frame.
        Same index as y_true and y_pred.
    y_true : pd.Series
        Ground truth labels.
        Same index as df and y_pred.
    y_pred : pd.Series
        Predictions. Okay if contains null.
        Same index as y_true and df.
    ns : List[int], default [10, 30, 60]
        list of thresholds n to use

    Returns
    -------
    pd.DataFrame
        Evaluation results.
        Row per group.
        Column per threshold.
    """
    evaluation_df = pd.DataFrame(
        index=df.group.unique(),
    )

    for t in ns:
        evaluation_df['percent_within_{}'.format(t)] = 0.0
        for group in df.group.unique():
            evaluation_df.loc[group, 'percent_within_{}'.format(t)] =\
                error_metrics.percent_within_n(
                    df.loc[df.group == group, y_true],
                    df.loc[df.group == group, y_pred],
                    t
                )

    return evaluation_df


def residual_distribution_summary_lookahead(
    data: pd.DataFrame,
    target: str,
    prediction: str,
    lookahead_column: str,
    lookahead_bins_seconds: np.ndarray = 60*np.array(
        [-1, 0, 5, 10, 15, 30, 60, 90, 120, 240, 360]
    ),
    quantiles: np.ndarray = np.array([0.05, 0.25, 0.5, 0.75, 0.95]),
):
    residual_distribution_summary_lookahead_bins_list = []

    for bin_lower, bin_upper in zip(
        lookahead_bins_seconds[:-1],
        lookahead_bins_seconds[1:]
    ):
        summary_lookahead_bin = pd.DataFrame(
            index=[0],
            columns=(
                ['bin_lower', 'bin_middle', 'bin_upper'] +
                ['q{}'.format(q) for q in quantiles] +
                ['num_samples', 'median_absolute_error']
            ),
        )
        summary_lookahead_bin.loc[0, 'bin_lower'] = bin_lower
        summary_lookahead_bin.loc[0, 'bin_upper'] = bin_upper
        summary_lookahead_bin.loc[0, 'bin_middle'] = (bin_upper + bin_lower)/2

        data_bin = data.loc[
            (
                (data[lookahead_column] > bin_lower) &
                (data[lookahead_column] <= bin_upper)
            ),
            :
        ]
        summary_lookahead_bin.loc[0, 'num_samples'] = data_bin.shape[0]
        errors_bin = (data_bin[target] - data_bin[prediction])
        summary_lookahead_bin.loc[0, 'median_absolute_error'] = (
            np.nanmedian(np.abs(errors_bin))
        )
        summary_lookahead_bin.loc[
            0,
            ['q{}'.format(q) for q in quantiles],
        ] = np.nanquantile(
            errors_bin,
            quantiles,
        )

        residual_distribution_summary_lookahead_bins_list.append(
            summary_lookahead_bin
        )

    return pd.concat(residual_distribution_summary_lookahead_bins_list)\
        .reset_index(drop=True)\
        .astype(float)
