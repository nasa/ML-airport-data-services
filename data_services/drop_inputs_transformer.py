from typing import List

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

"""Drop inputs from DataFrame.
Useful if raw inputs are used to compute features,
but not to be used as features themselves.
"""


class DropInputsTransfomer(BaseEstimator, TransformerMixin):
    """
    Drop columns.

    Can be useful is some inputs are used to compute features,
    then no longer needed in an sklearn Pipeline.
    """

    def __init__(
        self,
        drop_cols: List[str],
    ):
        """
        Drop columns.

        Parameters
        ----------
        drop_cols : List[str]
            List of names of columns to drop.
        """
        self.drop_cols = drop_cols

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
        self : DropInputsTransfomer class
        """
        return self

    def transform(
        self,
        X: pd.DataFrame,
    ):
        """
        Drop columns, then return data.

        Parameters
        ----------
        X : pd.DataFrame
            data set with columns to be dropped

        Returns
        -------
        data : pd.DataFrame
            data set with any columns in drop_cols dropped
        """
        X = X.drop(
            columns=self.drop_cols
        )

        return X
