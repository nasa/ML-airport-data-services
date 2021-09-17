from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class CompassSpeed(BaseEstimator, TransformerMixin):
    def __init__(
            self, lookaheads
    ):
        self.lookaheads = lookaheads

    def fit(
            self,
            X=None,
            y=None
    ):
        return self

    def transform(
            self,
            data,
    ) -> pd.DataFrame:
        input_fields = []
        for lookahead in self.lookaheads:
            lookahead_str = '_' + str(lookahead)
            wind_direction_col = 'wind_direction' + lookahead_str
            wind_speed_col = 'wind_speed' + lookahead_str
            compass_1_col = 'compass_1' + lookahead_str
            compass_2_col = 'compass_2' + lookahead_str
            compass_3_col = 'compass_3' + lookahead_str
            compass_4_col = 'compass_4' + lookahead_str

            data[compass_1_col] = 0
            data[compass_2_col] = 0
            data[compass_3_col] = 0
            data[compass_4_col] = 0

            idx = (data[wind_direction_col] >= 0) & (data[wind_direction_col] <= 45)
            data.loc[idx, compass_1_col] = data.loc[idx, wind_speed_col]

            idx = (data[wind_direction_col] >= 46) & (data[wind_direction_col] <= 90)
            data.loc[idx, compass_2_col] = data.loc[idx, wind_speed_col]

            idx = (data[wind_direction_col] >= 91) & (data[wind_direction_col] <= 135)
            data.loc[idx, compass_3_col] = data.loc[idx, wind_speed_col]

            idx = (data[wind_direction_col] >= 136) & (data[wind_direction_col] <= 180)
            data.loc[idx, compass_4_col] = data.loc[idx, wind_speed_col]

            idx = (data[wind_direction_col] >= 181) & (data[wind_direction_col] <= 225)
            data.loc[idx, compass_1_col] = -data.loc[idx, wind_speed_col]

            idx = (data[wind_direction_col] >= 226) & (data[wind_direction_col] <= 270)
            data.loc[idx, compass_2_col] = -data.loc[idx, wind_speed_col]

            idx = (data[wind_direction_col] >= 271) & (data[wind_direction_col] <= 315)
            data.loc[idx, compass_3_col] = -data.loc[idx, wind_speed_col]

            idx = (data[wind_direction_col] >= 316) & (data[wind_direction_col] <= 360)
            data.loc[idx, compass_4_col] = -data.loc[idx, wind_speed_col]

            input_fields.append(wind_direction_col)
            input_fields.append(wind_speed_col)

        data = data.drop(columns=input_fields)

        return data

    def get_feature_names(self):

        feature_names = []
        for lookahead in self.lookaheads:
            lookahead_str = '_' + str(lookahead)
            feature_names.append('compass_1' + lookahead_str)
            feature_names.append('compass_2' + lookahead_str)
            feature_names.append( 'compass_3' + lookahead_str)
            feature_names.append( 'compass_4' + lookahead_str)

        return feature_names