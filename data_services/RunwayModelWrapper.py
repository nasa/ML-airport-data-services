#!/usr/bin/env python

from __future__ import annotations

import pandas as pd
import numpy as np

from typing import Callable, Union, List, Tuple

from data_services.FilterPipeline import FilterPipeline

class RunwayModelWrapper(FilterPipeline):
    def __init__(
        self,
        operation: str,
        **kwargs,
    ):
        if operation not in ["dep", "arr"]:
            raise ValueError(f"Invalid operation type {operation} provided")
        self.operation = operation

        super().__init__(**kwargs)

    def predict_df(
        self,
        X_raw: pd.DataFrame,
    ) -> pd.DataFrame:

        X = X_raw.reset_index(drop=True)
        preds = super().predict_df(X)

        patterns = {
            "arr":'A_(?P<arr>.*)',
            "dep":'D_(?P<dep>.*)_A',
        }

        dat = (
            X
            .merge(
                preds,
                left_index=True,
                right_index=True,
                suffixes=[
                    None,
                    "_pred",
                ]
            )
        )

        dat["runways"] = (
            dat["airport_configuration_name"]
            .str
            .extract(patterns[self.operation])[self.operation]
        )

        dat["in_config"] = dat.apply(
            lambda row: row["pred"] in row["runways"]
                        if pd.notnull(row["runways"])
                        else None,
            axis=1,
        )

        if (
            (dat["in_config"].dtype == bool)
            and (~dat["in_config"]).any()
        ):
            rep_dat = dat.loc[dat["in_config"] == False, :]
            probs = self.predict_proba(rep_dat[X.columns])

            for c in probs.columns:
                rep_dat[c] = probs[c].values

            RUNWAY_NAMES = probs.columns
            def _max_prob_in_config(
                row: pd.DataFrame,
            ) -> pd.Series:
                for c in RUNWAY_NAMES:
                    if c not in row["runways"]:
                        row[c] = -1.0
                return RUNWAY_NAMES[np.argmax(row[RUNWAY_NAMES])]

            rep_vals = rep_dat.apply(
                _max_prob_in_config,
                axis=1,
            )

            dat.loc[rep_vals.index, "pred"] = rep_vals.values
            dat.loc[rep_vals.index, "error_msg"] = "pred from max prob"

        return dat[preds.columns]
