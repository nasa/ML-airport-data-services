from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class FillEmptyStrings(BaseEstimator, TransformerMixin):
    # Replaces all '' string values with value provided in initialization

    def __init__(self, value):
        self.value = value

    def fit(
        self,
        data: pd.DataFrame,
        y=None
    ):

        return self

    def transform(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        data[data == ''] = self.value

        return data