
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin
import statistics



class AirportConfigurationEncoder(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            configurations
    ):
        self.configurations = configurations
        self.configurations_values_map = None
        self.new_colnames = ['dep_avg_head','dep_num_rwy','arr_avg_head','arr_num_rwy']

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


        for k in self.new_colnames :
            data[k] = 0

        for k, v in self.configurations_values_map.items():
            idx = (data['airport_configuration_name_current'] == k)
            for col_i in self.new_colnames :
                data.loc[idx, col_i] = v[col_i]

        data = data.drop(columns=['airport_configuration_name_current'])
        return data

    def compute_values(self, configuration):
        """
        Calculate the average heading and the number of active runways for departures and arrivals
        for a given configuration

        """
        values = {}
        split_configs = configuration.split('_A_')
        
        arrival_runways = [int(r) for r in re.findall(r'\d+', split_configs[1])]
        values['arr_avg_head'] = statistics.mean(arrival_runways)
        values['arr_num_rwy'] = len(arrival_runways)

        departure_runways = [int(r) for r in re.findall(r'\d+', split_configs[0])]
        values['dep_avg_head'] = statistics.mean(departure_runways)
        values['dep_num_rwy'] = len(departure_runways)

        return values

    def get_feature_names(self):
        return self.new_colnames


