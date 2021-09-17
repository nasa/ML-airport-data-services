from pandas import DataFrame
import math
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict

import pandas as pd


class AvgStandAMATaxiEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self
    ):
        self.stand_avg_ama_time = None   # lookup table to be created in fit()
        self.overall_avg = None


    def fit(
            self,
            data,
            y,
    ):

        data['actual_arrival_ama_taxi_time']=y
        self.overall_avg = data['actual_arrival_ama_taxi_time'].mean()
        self.stand_avg_ama_time = data.groupby(["arrival_runway_actual",
                                                          "arrival_stand_actual"])['actual_arrival_ama_taxi_time'].mean()


        return self



    def transform(
            self,
            data,
    ) -> pd.DataFrame:

        transformed = pd.DataFrame(
            index=data.index,
            columns=[
                'stand_avg_ama_time'
            ],
        )
        transformed['stand_avg_ama_time'] = data.apply(self.get_stand_avg_ama_time, axis=1)

        return transformed


    def get_stand_avg_ama_time(
            self,
            row
    ):
        runway = row['arrival_runway_actual']
        stand = row['arrival_stand_actual']

        if runway not in self.stand_avg_ama_time or stand not in self.stand_avg_ama_time[runway]:
            return self.overall_avg

        return self.stand_avg_ama_time[runway][stand]

    def get_feature_names(self):
        return [self.stand_avg_ama_time.name]