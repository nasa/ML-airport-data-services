#!/usr/bin/env python

from __future__ import annotations

import re

import pandas as pd

from typing import List
from sklearn.base import BaseEstimator, TransformerMixin

class AirportConfigEncoder(BaseEstimator, TransformerMixin):
    def __init__(
            self,
            known_runways: List[str],
            ):
        self.known_runways = known_runways

        column_names = [
            "{}_{}".format(
                token,
                runway,
                )
            for token in ["arr", "dep"]
            for runway in known_runways
            ]
        column_names.sort()

        self.column_names = column_names

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series=None,
            ) -> AirportConfigEncoder:
        return self

    def transform(
            self,
            X: pd.DataFrame,
            ) -> pd.DataFrame:
        res = list()
        for config in X["airport_configuration_name"].unique():
            this_mapping = self._encode_config(config)
            res.append(this_mapping)

        config_mapping = pd.concat(
            res,
            ignore_index=False,
            )

        X_trans = (
            X
            .merge(
                config_mapping,
                on="airport_configuration_name",
                how="left",
            )
        )

        return X_trans[self.column_names]

    def _encode_config(
        self,
        config: str,
    ) -> pd.DataFrame:
        patterns = {
            "arr":'A_(?P<arr>.*)',
            "dep":'D_(?P<dep>.*)_A',
        }

        row = pd.DataFrame(
            columns=self.column_names,
            index=[0],
        )
        row.loc[0, "airport_configuration_name"] = config

        for token, pattern in patterns.items():
            payload = re.findall(pattern, config)

            if len(payload) > 0:
                runways = payload[0].split("_")

                if len(runways) > 0:
                    columns_to_set = [f"{token}_{r}" for r in runways]
                    row.loc[0, columns_to_set] = True

        row = row.fillna(False)
        return row

    def get_feature_names(
            self,
            ) -> List[str]:
        return self.column_names
