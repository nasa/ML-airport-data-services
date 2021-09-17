"""
Error metric functions.

This file contains functions that compute error metrics.
It handles null predicted values.
"""

import numpy as np
import pandas as pd


def mean_squared_error(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Mean squared error.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth labels
    y_pred : pd.Series
        Predictions.
        Same size as y_true.
        Can contain null.

    Returns
    -------
    float : mean squared error of non-null predictions
    """
    valid_idx = (y_pred.notnull())

    return np.mean(np.power(y_true[valid_idx] - y_pred[valid_idx], 2))


def mean_absolute_error(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Mean absolute error.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth labels
    y_pred : pd.Series
        Predictions.
        Same size as y_true.
        Can contain null.

    Returns
    -------
    float : mean absolute error of non-null predictions
    """
    valid_idx = (y_pred.notnull())

    return np.mean(np.abs(y_true[valid_idx] - y_pred[valid_idx]))


def median_absolute_error(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Median absolute error.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth labels
    y_pred : pd.Series
        Predictions.
        Same size as y_true.
        Can contain null.

    Returns
    -------
    float : median absolute error of non-null predictions
    """
    valid_idx = (y_pred.notnull())

    return np.median(np.abs(y_true[valid_idx] - y_pred[valid_idx]))


def mean_absolute_percentage_error(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Mean absolute percentage error.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth labels
    y_pred : pd.Series
        Predictions.
        Same size as y_true.
        Can contain null.

    Returns
    -------
    float :
        mean absolute percentage error of non-null predictions
        and where y_true is non-zero
    """
    valid_idx = ((y_true != 0) & y_pred.notnull())

    return np.mean(
        np.abs(
            (y_true[valid_idx] - y_pred[valid_idx]) /
            y_true[valid_idx]
        )
    ) * 100


def median_absolute_percentage_error(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Median absolute percentage error.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth labels
    y_pred : pd.Series
        Predictions.
        Same size as y_true.
        Can contain null.

    Returns
    -------
    float :
        median absolute percentage error of non-null predictions
        and where y_true is non-zero
    """
    valid_idx = ((y_true != 0) & y_pred.notnull())

    return np.median(
        np.abs(
            (
                y_true[valid_idx] -
                y_pred[valid_idx]
            ) /
            y_true[valid_idx]
        )
    ) * 100


def percent_within_n(
    y_true: pd.Series,
    y_pred: pd.Series,
    n: int = 1,
) -> float:
    """
    Percent of prediction errors less than a threshold n.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth labels
    y_pred : pd.Series
        Predictions.
        Same size as y_true.
        Can contain null.
    n : int, default 1
        Threshold for prediction errors

    Returns
    -------
    float :
        percent of non-null predictions with error less than threshold
    """
    valid_idx = (y_pred.notnull())

    return np.mean(np.abs(y_true[valid_idx]-y_pred[valid_idx]) < n) * 100


def rmse(
    y_true,
    y_pred,
):
    """
    Root mean squared error.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth labels
    y_pred : pd.Series
        Predictions.
        Same size as y_true.
        Can contain null.

    Returns
    -------
    float :
        RMSE of non-null predictions
    """
    valid_idx = (y_pred.notnull())

    return np.sqrt(mean_squared_error(y_true[valid_idx], y_pred[valid_idx]))


def tilted_loss(
    y_true: pd.Series,
    y_pred: pd.Series,
    quantile: float = 0.2,
):
    """
    Tilted loss.

    Basis of quantile regressors.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth labels
    y_pred : pd.Series
        Predictions.
        Same size as y_true.
        Can contain null.
    quantile : float, default 0.2
        Quantile used in tilted loss calculation

    Returns
    -------
    float :
        tilted loss of non-null predictions
    """
    valid_idx = (y_pred.notnull())

    return np.mean(
        np.max(
            np.vstack(
                (
                    quantile*(y_true[valid_idx]-y_pred[valid_idx]).values,
                    (quantile-1)*(y_true[valid_idx]-y_pred[valid_idx]).values
                )
            ),
            axis=0
        )
    )


def fraction_less_than_actual(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Fraction less than actual (true).

    Useful for quantile regressor evaluation.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth labels
    y_pred : pd.Series
        Predictions.
        Same size as y_true.
        Can contain null.

    Returns
    -------
    float :
        fraction of non-null predictions less than corresponding true values
    """
    valid_idx = (y_pred.notnull())

    return np.mean(y_pred[valid_idx] < y_true[valid_idx])


def percent_valid_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """
    Fraction non-null ("valid") predictions.

    Useful for evaluation outputs of FilterPipeline, which can contain null.

    Parameters
    ----------
    y_true : pd.Series
        Ground truth labels.
        Not used here, but included for uniformity.
    y_pred : pd.Series
        Predictions.
        Same size as y_true.
        Can contain null.

    Returns
    -------
    float :
        fraction of predictions that are null.
    """
    return y_pred.notnull().mean() * 100


# This is used to translate from metric names provided in parameters
# to actual metric functions in this file
METRIC_NAME_TO_FUNCTION_DICT = {
    'mean_absolute_error': mean_absolute_error,
    'median_absolute_error': median_absolute_error,
    'mean_absolute_percentage_error': mean_absolute_percentage_error,
    'median_absolute_percentage_error': median_absolute_percentage_error,
    'percent_within_n': percent_within_n,
    'mean_squared_error': mean_squared_error,
    'rmse': rmse,
    'tilted_loss': tilted_loss,
    'fraction_less_than_actual': fraction_less_than_actual,
    'percent_valid_predictions': percent_valid_predictions,
}
