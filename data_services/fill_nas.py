from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class FillNAs(BaseEstimator, TransformerMixin):
    # Replaces all None and NA values with value provided in initialization

    def __init__(self, value):
        self.value = value

    def fit(
        self,
        data,
        y=None
    ):

        return self

    def transform(
        self,
        data: np.ndarray,
    ) -> np.ndarray:

        for i in range(len(data)):
            for j in range(len(data[i])):
                if data[i][j] is None or (
                    type(data[i][j]) != str and
                    np.isnan(data[i][j])
                ):
                    data[i][j] = self.value

        return data
