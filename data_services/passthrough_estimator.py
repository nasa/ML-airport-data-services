from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class PassthroughEstimator(BaseEstimator, TransformerMixin):
    """Estimate or predict by just passing through an input."""

    def __init__(
        self,
        passthrough_feature: str,
    ):
        """
        Estimate or predict by just passing through an input.

        Parameters
        ----------
        passthrough_feature : str,
            model input (feature) to be returned as the estimate/prediction

        Returns
        -------
        None.
        """
        self.passthrough_feature = passthrough_feature

    def fit(
        self,
        data: pd.DataFrame,
        y: pd.Series,
    ):
        """
        Fit.

        Checks that passthrough_feature is in the data.
        Checks that the passthrough_feature data type matches up
            with target type.

        Parameters
        ----------
        data : pd.DataFrame
            features data set
        y : pd.Series
            target data set

        Returns
        -------
        self : PassthroughEstimator class
        """
        if self.passthrough_feature not in data.columns:
            raise TypeError(
                'passthrough feature {} not in data columns'.format(
                    self.passthrough_feature
                )
            )

        if not isinstance(
            data[self.passthrough_feature].values[0],
            type(y.iloc[0])
        ):
            raise ValueError(
                'passthrough feature {} values are of type {}, '.format(
                    self.passthrough_feature,
                    type(data[self.passthrough_feature].values[0])
                ) +
                'but target is of different type {}.'.format(
                    type(y.iloc[0])
                )
            )

        return self

    def predict(
        self,
        data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Predict by passing through the passthrough_feature.

        Parameters
        ----------
        data : pd.DataFrame
            The features data set to be predicted

        Returns
        -------
        np.ndarray : prediction
        """
        return data[self.passthrough_feature].values
