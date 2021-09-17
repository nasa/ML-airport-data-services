
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Any, Dict

import pandas as pd


class AircraftClassEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        aircraft_categories: Dict[str, str],
    ):
        self.aircraft_categories = aircraft_categories

    def fit(
        self,
        aircraft_class,
        y=None,
    ):
        self.category = list(set(self.aircraft_categories.values()))
        self.category = [x for x in self.category if pd.notnull(x)]
        self.category.sort()
        return self

    def transform(
        self,
        aircraft_type,
    ) -> pd.DataFrame:
        class2categories = aircraft_type['aircraft_type']\
            .map(self.aircraft_categories)
        class2categories = class2categories.to_frame()

        # Hand-rolled one-hot encoder
        transformed = pd.DataFrame(
            index=class2categories.index,
            columns=[
                'aircraft_class_category_{}'.format(c)
                for c in self.category
            ],
        )
        transformed.fillna(0, inplace=True)
        # Approach here will ignore (leave all as 0s) any
        # unknown aircraft type (stand not in aircraft_categories dictionary)
        # or any time aircraft type is missing
        for c in self.category:
            transformed.loc[
                class2categories.aircraft_type == c,
                'aircraft_class_category_{}'.format(c)
            ] = 1

        return transformed
    
    def get_feature_names(self):
        features = []
        for c in self.category:
            features.append('aircraft_class_category_{}'.format(c))
        return features
