
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import statistics



def get_feature_names(self):
    return self.fields


class AirportConfigurationEncoder(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            configurations,
            fields
    ):
        self.configurations = configurations
        self.configurations_values_map = None
        self.fields = fields

    def fit(
            self,
            X=None,
            y=None
    ):
        self.configurations_values_map = {k: self.compute_values(k) for k in self.configurations}
        return self

    def transform(
            self,
            data,
    ) -> pd.DataFrame:

        arrival_avg = 'arrival_avg'
        arrival_count = 'arrival_count'
        departure_avg = 'departure_avg'
        departure_count = 'departure_count'

        data[arrival_avg] = 0
        data[arrival_count] = 0
        data[departure_avg] = 0
        data[departure_count] = 0

        for k, v in self.configurations_values_map.items():
            idx = (data['airport_configuration_name_current'] == k)
            data.loc[idx, arrival_avg] = v[0]
            data.loc[idx, arrival_count] = v[1]
            data.loc[idx, departure_avg] = v[2]
            data.loc[idx, departure_count] = v[3]

        data = data.drop(columns=['airport_configuration_name_current'])
        return data

    def compute_values(self, configuration):
        split_configs = configuration.split('_A_')

        arrival_runways = [int(r.split('/')[0][:-1]) for r in configuration.split('_A_')[1].split('_')]
        arrival_avg = statistics.mean(arrival_runways)
        arrival_count = len(arrival_runways)

        departure_runways = [int(r.split('/')[0][:-1]) for r in configuration.split('_A_')[0].split('_')[1:]]
        departure_avg = statistics.mean(departure_runways)
        departure_count = len(departure_runways)

        return [departure_avg, departure_count, arrival_avg, arrival_count]

