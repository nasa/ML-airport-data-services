"""Encoder class that extend scikit-learn encoder

Requires a pre-defined mapping from stands to stand clusters.
"""

from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict

import pandas as pd
import re


# 

class TerminalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, col_to_transf):

        # if only a string, transform to 1 element list
        if (type(col_to_transf) == str) :
            col_to_transf = [col_to_transf]
        if ((len(col_to_transf) != 1) | (['arrival_stand_actual','departure_stand_actual'].count(col_to_transf[0]) == 0)) :
            raise(TypeError("TerminalEncoder expects 1 column dataframe with specific column name\n"+\
                            "Given DataFrame with columns : {}".format(col_to_transf)))

        self.col_to_transf = col_to_transf

    def fit(
            self,
            data,
            y=None
    ):
        
        colnames = []
        df_gates = data[self.col_to_transf].drop_duplicates()
        df_gates = df_gates[df_gates.isna() == False]
        # Add split information on Terminal and Gate number
        df_gates = df_gates.join(df_gates.iloc[:,0].str.extract('(?P<terminal>[A-Z]+)(?P<gatenum>[0-9]+)'))

        # Check there is at least 1 Terminal Name in the DataFrame as defined by the previous formula
        if (len(df_gates.terminal.dropna()) == 0) :
            raise(TypeError("TerminalEncoder did not find any Terminal in the dataframe"))
        
        self.df_gates_encoded = df_gates
        self.colnames = df_gates['terminal'].dropna().unique()
        return self

    
    def transform(
            self,
            data,
    ) -> pd.DataFrame:

        key_to_merge = self.col_to_transf
        data = pd.merge(data, self.df_gates_encoded, sort=False, on=key_to_merge, how='left')
        return data

    def get_feature_names(self):
            return self.colnames

        
