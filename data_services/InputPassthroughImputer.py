from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np



class InputPassthroughImputer(BaseEstimator, TransformerMixin):
    """Impute a column by passing through the value in another column."""

    def __init__(
        self,
        pass_from_field: str,
        pass_to_field: str,
    ):
        """
        Impute a column by passing through the value in another column.

        Parameters
        ----------
        pass_from_field : str
            Field whose values will fill in the null values.
        pass_to_field : str
            Field to be imputed (whose null values will be filled in)

        Returns
        -------
        None.
        """
        self.pass_from_field = pass_from_field
        self.pass_to_field = pass_to_field

    def fit(
        self,
        X: pd.DataFrame,
        y=None,
    ):
        """
        Fit imputer.

        Just checks for existence of columns in data set.

        Parameters
        ----------
        X : pd.DataFrame
            data set that will be transformed via imputation
        y : default None
            target data set

        Returns
        -------
        self : DataInspector class
        """
        if self.pass_from_field not in X.columns:
            raise(
                ValueError,
                'pass from input {} not in data columns'.format(
                    self.pass_from_field
                )
            )

        if self.pass_to_field not in X.columns:
            raise(
                ValueError,
                'pass to input {} not in data columns'.format(
                    self.pass_to_field
                )
            )

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ):
        """
        Impute.

        Parameters
        ----------
        data : pd.DataFrame
            data to be imputed

        Returns
        -------
        data : pd.DataFrame
            data after imputation by passing over from other field
        """
        X.loc[:, self.pass_to_field] = X[self.pass_to_field].fillna(
            X[self.pass_from_field].copy(deep=True)
        )

        return X
