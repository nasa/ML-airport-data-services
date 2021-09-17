#!/usr/bin/env python

"""Simple predictive model that returns the most common value for some input
"""

from sklearn.base import BaseEstimator

class MostCommonValue(BaseEstimator):
    def __init__(
            self,
            group_by_fields:list,
            ):
        if ("," in group_by_fields):
            group_by_fields = group_by_fields.split(",")

        if not isinstance(group_by_fields, list):
            group_by_fields = [group_by_fields]

        self.group_by_fields = group_by_fields
        self.prediction_table = None
        self.most_common_value = None

    def fit(self, X, y):
        X0 = X.copy()
        X0["res"] = y
        lookup_table = (
            X0
            .groupby(self.group_by_fields)
            .agg({
                "res":lambda x: x.value_counts().index[0],
                })
            .reset_index()
            )
        self.prediction_table = lookup_table
        self.most_common_value = X0["res"].value_counts().index[0]
        return self

    def predict(self, X):
        preds = (
            X
            .loc[:, self.group_by_fields]
            .merge(
                self.prediction_table,
                how="left",
                on=self.group_by_fields,
                )
            .loc[:, "res"]
            .fillna(self.most_common_value)
            .to_numpy()
            )
        return preds

    def score(self, X, y):
        preds = self.predict(X)
        score = sum(preds == y.to_numpy()) / len(y)
        return score
