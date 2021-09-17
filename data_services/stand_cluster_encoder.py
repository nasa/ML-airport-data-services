"""Encoder class that extend scikit-learn encoder

Requires a pre-defined mapping from stands to stand clusters.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict

import pandas as pd
import re

class StandEncoder(BaseEstimator, TransformerMixin):

    def fit(
            self,
            data,
            y=None
    ):
        # Gate names are encoded as a vector with the gate number for each inferred terminal as an independent element
        # E.g for an airport with terminals A, B and C,  B12-> [0,12,0], C30-> [0,0,30]
        if ((len(data.columns) != 1) | (['arrival_stand_actual','departure_stand_actual'].count(data.columns[0]) == 0)) :
            raise(TypeError("StandEncoder fit expects 1 column dataframe with specific column name\n"+\
                            "Given DataFrame with columns : {}".format(data.columns)))

        colnames = []
        df_gates = data.drop_duplicates()
        df_gates = df_gates[df_gates.iloc[:,0].isna() == False]

        for i in range(len(df_gates.index)):
            v = df_gates.values[i,0]
            m = re.search(r"\d+", v)
            if (m is not None):  # and (m.start()>0)
                colname = 'group_' + v[0:m.start()]

                if colname not in list(df_gates):
                    df_gates[colname] = 0
                    colnames.append(colname)

                df_gates[colname].values[i] = int(m.group())
            else:
                # One hot encode single gate if note numbered
                colname = 'group_' + v
                if colname not in list(df_gates):
                    df_gates[colname] = 0
                    colnames.append(colname)
                df_gates[colname].values[i] = 1

        self.df_gates_encoded = df_gates.reset_index(drop=True)
        self.colnames = colnames
        return self

    def transform(
            self,
            data,
    ) -> pd.DataFrame:
        
        if ((len(data.columns) != 1) | (['arrival_stand_actual','departure_stand_actual'].count(data.columns[0]) == 0)) :
            raise(TypeError("StandEncoder transform expects 1 column dataframe with specific column name\n"+\
                            "Given DataFrame with columns : {}".format(data.columns)))

        
        key_to_merge = data.columns[0]
        data = pd.merge(data, self.df_gates_encoded, sort=False, on=key_to_merge, how='left')
        data.fillna(0, inplace=True)

        return data[self.colnames]

    def get_feature_names(self):
            return self.colnames
