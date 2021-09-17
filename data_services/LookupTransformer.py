#!/usr/bin/env python

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class LookupTransformer(TransformerMixin, BaseEstimator):

    def __init__(
            self,
            lookup_data: dict
            ):
        self.lookup_data = lookup_data

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (
            pd
            .Series(
                X.flatten()
                )
            .map(self.lookup_data)
            .values
            .reshape(-1,1)
            )

    def inverse_transform(self, X):
        return X
