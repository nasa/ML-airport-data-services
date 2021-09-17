from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as sklearn_Pipeline
from sklearn.preprocessing import OneHotEncoder
from data_services.fill_nas import FillNAs
from data_services.OrderFeatures import OrderFeatures
from data_services.compass_speed import CompassSpeed
from data_services.utils import TimeOfDay
from data_services.format_missing_data import FormatMissingData
import xgboost as xgb
import pandas as pd
from typing import Dict, Any
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from data_services.apc_flow_encoder import AirportConfigurationEncoder

def has_encoder(encoder, features):
    return features['name'][features['encoder'].str.contains(r'^\s*{0}\s*,|,\s*{0}\s*,|,\s*{0}\s*$|^\s*{0}\s*$'.format(encoder))]

def airport_configuration_pipeline(
        data: pd.DataFrame,
        features: pd.DataFrame,
        model: Dict[str, Any]
) -> sklearn_Pipeline:

    # Wind speed/direction encoding
    lookaheads = np.unique(features['lat'][features['encoder'] == 'compass_speed'])
    compass_speed = CompassSpeed(lookaheads)

    configurations= pd.unique(data.airport_configuration_name_current)
    airport_configuration_encoder = AirportConfigurationEncoder(configurations,  has_encoder('airport_configuration_encoder', features))

    # Build one-hot encoder for general categorical features
    oh_enc = OneHotEncoder(
        categories=[
            data[feature].unique()
            for feature in has_encoder('onehot', features)
        ],
        sparse=False,
        handle_unknown="ignore",
    )

    time_of_day = TimeOfDay(has_encoder('timeofday', features))
    ordinal_encoder=OrdinalEncoder()


    # Make column transformer
    col_transformer = ColumnTransformer([
        ('one_hot_encoder', oh_enc, has_encoder('onehot', features)),
        ('airport_configuration_encoder', airport_configuration_encoder, has_encoder('airport_configuration_encoder', features)),
        ('ordinal_encoder', ordinal_encoder, has_encoder('ordinal', features)),
        ('timeofday_encoder', time_of_day, has_encoder('timeofday', features)),

    ],
        remainder='passthrough',
    )

    # Orders feature columns
    order_features = OrderFeatures()

    # Replaces miscellaneous missing values with expected values
    format_missing_data = FormatMissingData()

    # Replace NAs with provided value, only affecting numeric/bool since categorical are handled at the encoder level
    fill_nas = FillNAs(0)

    model= xgb.XGBClassifier(**model['model_params'])

    # Make pipeline
    pipeline = sklearn_Pipeline(
        steps=[
            ('order_features', order_features),
            ('format_missing_data', format_missing_data),
            ('col_transformer', col_transformer),
            ('fill_nas', fill_nas),
            ('imp_model', model),
        ]
    )

    return pipeline