from pandas import DataFrame
import math
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict

import pandas as pd


class AvgRunwayAMATaxiEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
    ):
        self.runway_ama_avg_taxi_time = None

    def fit(
        self,
        data,
        y,
    ):
        data['actual_arrival_ama_taxi_time']=y
        self.runway_ama_avg_taxi_time = data.groupby(["arrival_runway_actual"])['actual_arrival_ama_taxi_time'].mean()
        return self

    def transform(
        self,
        data,
    ) -> pd.DataFrame:

        transformed = pd.DataFrame(
            index=data.index,
            columns=[
                'runway_ama_avg_taxi_time'
            ],
        )
        transformed['runway_ama_avg_taxi_time'] = data.apply(self.get_avg_taxi_time, axis=1)

        return transformed

    def get_avg_taxi_time(
        self,
        row
    ):
        runway = row['arrival_runway_actual']

        return self.runway_ama_avg_taxi_time.loc[runway] if runway in self.runway_ama_avg_taxi_time else self.runway_ama_avg_taxi_time.mean()
    
    def get_feature_names(self):
        return [self.runway_ama_avg_taxi_time.name]
