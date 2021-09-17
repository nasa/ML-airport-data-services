from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from typing import List


class FormatMissingData(BaseEstimator, TransformerMixin):
    """Modify missing data values as required by model sklearn Pipeline."""

    def __init__(
        self,
        skip_inputs: List[str] = [],
    ):
        """
        Modify missing data values as required by model sklearn Pipeline.

        Parameters
        ----------
        skip_inputs : List[str], default []
            list of features whose missing values will
            not be adjusted
            if [], then *all* inputs are formatted

        Returns
        -------
        None.
        """
        self.skip_inputs = skip_inputs

    def fit(
        self,
        X=None,
        y=None,
    ):
        """
        Fit.

        Does nothing.

        Parameters
        ----------
        X : default None
            features data set
        y : default None
            target data set

        Returns
        -------
        self : FormatMissingData class
        """
        return self

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform.

        Convert None or '' to np.nan.

        Parameters
        ----------
        data : pd.DataFrame
            data with missing values to be adjusted

        Returns
        -------
        data : pd.DataFrame
            same as what was passed in, but with missing data adjustments
        """
        for feature in data.columns:
            if feature not in self.skip_inputs:
                if np.issubdtype(data[feature].dtype, np.number):
                    data[feature].replace([None], [np.nan], inplace=True)
                else:
                    idx = data[feature].isin(['', None])
                    data.loc[idx, feature] = None

        return data
