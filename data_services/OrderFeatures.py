from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class OrderFeatures(BaseEstimator, TransformerMixin):
    # Reorders columns per training order
    def fit(
            self,
            data: pd.DataFrame,
            y=None
    ):
        self.features_ordered = data.columns
        return self

    def transform(
            self,
            data : pd.DataFrame,
    ) -> pd.DataFrame:
        return data[self.features_ordered].copy()
