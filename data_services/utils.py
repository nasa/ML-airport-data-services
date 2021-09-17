import numpy as np
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

def keep_positive_round(
    x: np.ndarray,
) -> np.ndarray:
    x = np.round(x)
    x[x < 0] = np.nan
    return (x)


def scp_artifacts_to_NTX_casa_wrapper(
    model_pipeline: Pipeline,
):
    pass


class FormatMissingData(BaseEstimator, TransformerMixin):
    # Modifies missing data values as required by model pipeline
    def __init__(self, features_core):
        self.features_core = features_core

    def fit(
            self,
            X=None,
            y=None
    ):
        return self

    def transform(
            self,
            data: pd.DataFrame,
    ) -> pd.DataFrame:

        for feature in data.columns:
            if feature not in self.features_core:
                if np.issubdtype(data[feature].dtype, np.number):
                    data[feature].replace([None], [np.nan], inplace=True)
                else:
                    idx = data[feature].isin(['', np.nan])
                    data.loc[idx, feature] = None

        return data


class TimeOfDay(BaseEstimator, TransformerMixin):

    def __init__(
            self, fields
    ):
        self.fields = fields

    def fit(
            self,
            X=None,
            y=None
    ):
        return self

    def transform(
            self,
            data,
    ) -> pd.DataFrame:
        for field in self.fields:
            data[field + '_hour'] = data[field].dt.hour
            data = data.drop(columns=['timestamp'])

        return data

    def get_feature_names(self):
        return self.fields