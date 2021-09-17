import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline as sklearn_Pipeline
from data_services.FilterPipeline import FilterPipeline
from typing import Any, Dict, Callable


class TimeDeltaTotalSecondsTransformer(BaseEstimator, TransformerMixin):
    """Compute a timedelta difference feature from two timestamp fields."""

    def __init__(
        self,
        timedelta_start_col_name: str,
        timedelta_end_col_name: str,
        new_timedelta_col_name: str = 'timedelta',
    ):
        """
        Compute a timedelta difference feature from two timestamp fields.

        Parameters
        ----------
        timedelta_start_col_name : str
            start of time delta feature
        timedelta_end_col_name : str
            end of time delta feature
        new_timedelta_col_name : str, default 'timedelta'
            name of computed feature

        Returns
        -------
        None.
        """
        self.timedelta_start_col_name = timedelta_start_col_name
        self.timedelta_end_col_name = timedelta_end_col_name
        self.new_timedelta_col_name = new_timedelta_col_name

    def fit(
        self,
        data: pd.DataFrame,
        y=None,
    ):
        """
        Fit.

        Checks that timedelta start and end columns are in the data columns.

        Parameters
        ----------
        X : pd.DataFrame
            features data set
        y : default None
            target data set

        Returns
        -------
        self : TimeDeltaTotalSecondsTransformer class
        """
        if self.timedelta_start_col_name not in data.columns.tolist():
            raise(ValueError(
                'timedelta start column name {} not in data'.format(
                    self.timedelta_end_col_name
                )
            ))

        if self.timedelta_end_col_name not in data.columns.tolist():
            raise(ValueError(
                'timedelta end column name {} not in data'.format(
                    self.tiemdelta_end_col_name
                )
            ))

        return self

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform by computing the timedelta feature.

        Parameters
        ----------
        data : pd.DataFrame
            data with start and end time columns

        Returns
        -------
        data :  pd.DataFrame
            same as what was passed in, but with new timedelta feature.
            computed in seconds
        """
        transformed = data.copy(deep=True)
        transformed[self.new_timedelta_col_name] = (
            data[self.timedelta_end_col_name] -
            data[self.timedelta_start_col_name]
        ).dt.total_seconds()

        return transformed


class AddSecondsToTimestampTransformer(BaseEstimator, TransformerMixin):
    """Compute timestamp feature by adding timedelta to another timestsamp."""

    def __init__(
        self,
        base_timestamp_col_name: str,
        timedelta_seconds_col_name: str,
        transformed_timestamp_col_name: str = 'transformed_timestamp',
    ):
        """
        Compute timestamp feature by adding timedelta to another timestsamp.

        Parameters
        ----------
        base_timestamp_col_name : str
            start timestamp column
        timedelta_seconds_col_name : str
            column with timedelta to add to base timestamp
        transformed_timestamp_col_name : str, default 'transformed_timestamp'
            name of new computed timestamp feature column

        Returns
        -------
        None.
        """
        self.base_timestamp_col_name = base_timestamp_col_name
        self.timedelta_seconds_col_name = timedelta_seconds_col_name
        self.transformed_timestamp_col_name = transformed_timestamp_col_name

    def fit(
        self,
        data: pd.DataFrame,
        y=None,
    ):
        """
        Fit.

        Checks that timedelta start and timedelta are in the data columns.

        Parameters
        ----------
        data : pd.DataFrame
            features data set
        y : default None
            target data set

        Returns
        -------
        self : TimeDeltaTotalSecondsTransformer class
        """
        if self.base_timestamp_col_name not in data.columns.tolist():
            raise(ValueError(
                'base timestamp column name {} not in data'.format(
                    self.timedelta_end_col_name
                )
            ))

        if self.timedelta_seconds_col_name not in data.columns.tolist():
            raise(ValueError(
                'timedelta seconds column name {} not in data'.format(
                    self.tiemdelta_end_col_name
                )
            ))

        return self

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform by computing the new timestamp feature.

        Parameters
        ----------
        data : pd.DataFrame
            data with start and end time columns

        Returns
        -------
        data :  pd.DataFrame
            same as what was passed in, but with new timestamp feature
        """
        transformed = pd.DataFrame(
            index=data.index,
            columns=data.columns + [self.transformed_timestamp_col_name],
        )
        transformed[self.transformed_timestamp_col_name] = (
            data[self.base_timestamp_col_name] +
            pd.to_timedelta(data[self.timedelta_seconds_col_name], unit='s')
        )

        return transformed


class AddDurationToTimestampPipeline(sklearn_Pipeline):
    """
    Wrapper around FilterPipeline to transform prediction from
    timedelta duration to a timestamp by adding it to a timestamp in the input.
    """
    def __init__(
        self,
        core_pipeline: FilterPipeline,
        base_timestamp_col_name: str = 'timestamp',
        default_response: object = np.datetime64('NaT'),  # needed?
        core_prediction_units: str = 'seconds',
    ):
        """
        Wrapper class to go around a FilterPipeline and transform time duration
        predictions to timestamps, by adding them to a base timestamp in
        the input features data.

        Parameters
        ----------
        core_pipeline : FilterPipeline
            Itself a wrapper around sklearn 'estimator'
            scikit-learn estimator or Pipeline object at the center of this
                predictive model that implements .fit() and .predict()
        base_timestamp_col_name : str
            name of the base timestamp column to which predicted durations
            are added to get the output predicted timestamps
        default_response : object
            Default response (may not need this)
        core_prediction_units: str
            Units for predicted durations

        Returns
        -------
        None.
        """
        self.core_pipeline = core_pipeline
        self.base_timestamp_col_name = base_timestamp_col_name
        self.default_response = default_response
        self.core_prediction_units = core_prediction_units

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> object:
        """
        Confirm that base_timestamp_col_name is in columns of X and then
        fit by calling core_pipeline.fit().

        Parameters
        ----------
        X : pd.DataFrame
            features data set
        y : pd.Series
            target data set

        Returns
        -------
        self : AddDurationToTimestampPipeline class with core pipeline fit
        """
        if self.base_timestamp_col_name not in X.columns:
            raise(ValueError(
                'base timestamp column name {} not in data'.format(
                    self.timedelta_end_col_name
                )
            ))

        self.core_pipeline.fit(X, y)

        return self

    def predict_core(
        self,
        X: pd.DataFrame,
    ):
        """
        Get predictions from core pipeline, without any conversion.

        Parameters
        ----------
        X : pd.DataFrame
            data for which predictions are to be provided by core pipeline

        Results
        -------
        predictions from core pipeline
        """
        return self.core_pipeline.predict(X)

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Get predictions, converted from timedelta to timestamp.

        Parameters
        ----------
        X : pd.DataFrame
            data for which predictions are to be provided

        Returns
        -------
        np.ndarray : np.ndarray of predictions converted to np.datetime64
        """
        # get predictions, in core units
        predictions_core = self.core_pipeline.predict(X)

        # Convert to datetime
        predictions_datetime = X[self.base_timestamp_col_name] +\
            pd.to_timedelta(
                predictions_core,
                unit=self.core_prediction_units,
            )

        return predictions_datetime.to_numpy(dtype=np.datetime64)

    def predict_df_core(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Get predictions and errors from core FilterPipeline .predict_df(),
        with no additional conversions.

        Parameters
        ----------
        X : pd.DataFrame
            data for which predictions are to be provided

        Returns
        -------
        pd.DataFrame : predictions data frame from core
        """
        return self.core_pipeline.predict_df(X)

    def predict_df(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Get predictions and errors like from core FilterPipeline .predict_df(),
        with predictions converted to timestamp.

        Parameters
        ----------
        X : pd.DataFrame
            data for which predictions are to be provided

        Returns
        -------
        pd.DataFrame : predictions data frame like from FilterPipeline,
            but with predictions converted to timestamp
        """
        # get predictions dataframe in core units
        predictions_df = self.core_pipeline.predict_df(X)

        # convert to datetime
        # rename the preds column because will overwrite it
        predictions_df = predictions_df.rename(
            columns={'pred': 'pred_core'},
        )
        # Join in the base timestamp col from X
        predictions_df = predictions_df.join(
            X[self.base_timestamp_col_name]
        )
        # Now compute preds as datetime
        predictions_df['pred'] = (
            predictions_df[self.base_timestamp_col_name] +
            pd.to_timedelta(
                predictions_df['pred_core'],
                unit=self.core_prediction_units,
            )
        )
        # drop columns no longer need
        predictions_df = predictions_df.drop(
            columns=[
                'pred_core',
                self.base_timestamp_col_name,
            ],
        )

        return predictions_df
