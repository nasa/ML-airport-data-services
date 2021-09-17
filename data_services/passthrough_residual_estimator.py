import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.pipeline import Pipeline as sklearn_Pipeline
from data_services.FilterPipeline import FilterPipeline
from typing import Any, Dict, Callable


class PassthroughResidualEstimator(BaseEstimator, TransformerMixin):
    """
    Wrapper around sklearn BaseEstimator to enable prediction of the target via
    prediction of the residual of using an input as the prediction.
    """
    def __init__(
        self,
        core_model: BaseEstimator,
        predict_residual_of_input: str,
    ):
        """
        Wrapper class to go around an sklearn Pipeline enable prediction of
        the target via prediction of the residual of using an input as
        the prediction.

        Parameters
        ----------
        core_model : BaseEstimator
            scikit-learn estimator at the center of this
                predictive model that implements .fit() and .predict()
        predict_residual_of_input : str
            the input to use as the baseline prediction;
                model will learn to predict the residual of this prediction;
                final prediction will be this baseline prediction plus the
                predicted residual

        Returns
        -------
        None.
        """
        self.core_model = core_model
        self.predict_residual_of_input = predict_residual_of_input

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ):
        """
        Confirm that predict_residual_of_input is in the columns of X
        Create new target with it.
        Fit with core model.

        Parameters
        ----------
        X : pd.DataFrame
            features data set
        y : pd.Series
            target data set

        Returns
        -------
        self : PredictResidualPipeline with core pipeline fit
        """
        if self.predict_residual_of_input not in X.columns:
            raise(ValueError(
                'base prediction column name {} not in data'.format(
                    self.predict_residual_of_input
                )
            ))

        residual_target = y.values - X[self.predict_residual_of_input].values

        self.core_model.fit(X, residual_target)

        return self

    def predict_core(
        self,
        X: pd.DataFrame,
    ):
        """
        Get predictions from core pipeline, which is fitted to predict
        the residual, without any conversion.

        Parameters
        ----------
        X : pd.DataFrame
            data for which predictions are to be provided

        Results
        -------
        predictions of residuals from core pipeline
        """
        return self.core_model.predict(X)

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Get predictions, coverted back to predictions of target
        rather than predictions of residuals.

        Parameters
        ----------
        X : pd.DataFrame
            data for which predictions are to be provided

        Results
        -------
        np.ndarray : np.ndarray of predictions of target produced
            via adding predicted residuals to predict_residual_of_input
        """
        predicted_residuals = self.core_model.predict(X)

        return X[self.predict_residual_of_input].values + predicted_residuals
