#!/usr/bin/env python

import unittest

import pandas as pd

from data_services.airport_config_encoder import AirportConfigEncoder

class Test(unittest.TestCase):
    def test_complete_data(self):
        X = pd.DataFrame({
            "airport_configuration_name":[
                "D_35L_36R_A_31R_35C_35R",
            ]
        })

        ace = AirportConfigEncoder(["35L","36R","31L","31R","35C","35R"])

        Xt = ace.transform(X)

        self.assertEqual(
            Xt.shape[0],
            1,
        )
        self.assertEqual(
            Xt.shape[1],
            12,
        )

        correct_answers = [
            "dep_35L",
            "dep_36R",
            "arr_31R",
            "arr_35C",
            "arr_35R",
        ]
        for c in Xt.columns:
            self.assertEqual(
                Xt.loc[0,c],
                True if c in correct_answers else False,
            )

    def test_missing_arrs(self):
        X = pd.DataFrame({
            "airport_configuration_name":[
                "D_35L_36R_A",
            ]
        })

        ace = AirportConfigEncoder(["35L","36R","31L","31R","35C","35R"])

        Xt = ace.transform(X)

        self.assertEqual(
            Xt.shape[0],
            1,
        )
        self.assertEqual(
            Xt.shape[1],
            12,
        )

        correct_answers = [
            "dep_35L",
            "dep_36R",
        ]
        for c in Xt.columns:
            self.assertEqual(
                Xt.loc[0,c],
                True if c in correct_answers else False,
            )

    def test_missing_deps(self):
        X = pd.DataFrame({
            "airport_configuration_name":[
                "DA_31R_35C_35R",
            ]
        })

        ace = AirportConfigEncoder(["35L","36R","31L","31R","35C","35R"])

        Xt = ace.transform(X)

        self.assertEqual(
            Xt.shape[0],
            1,
        )
        self.assertEqual(
            Xt.shape[1],
            12,
        )

        correct_answers = [
            "arr_31R",
            "arr_35C",
            "arr_35R",
        ]
        for c in Xt.columns:
            self.assertEqual(
                Xt.loc[0,c],
                True if c in correct_answers else False,
            )

    def test_missing_all(self):
        X = pd.DataFrame({
            "airport_configuration_name":[
                "DA",
            ]
        })

        ace = AirportConfigEncoder(["35L","36R","31L","31R","35C","35R"])

        Xt = ace.transform(X)

        self.assertEqual(
            Xt.shape[0],
            1,
        )
        self.assertEqual(
            Xt.shape[1],
            12,
        )

        correct_answers = []
        for c in Xt.columns:
            self.assertEqual(
                Xt.loc[0,c],
                True if c in correct_answers else False,
            )

    def test_multiple_complete_rows(self):
        X = pd.DataFrame({
            "airport_configuration_name":[
                "D_35L_36R_A_31R_35C_35R",
                "D_36R_A_31L",
            ]
        })

        ace = AirportConfigEncoder(["35L","36R","31L","31R","35C","35R"])

        Xt = ace.transform(X)

        self.assertEqual(
            Xt.shape[0],
            2,
        )
        self.assertEqual(
            Xt.shape[1],
            12,
        )

        correct_answers_0 = [
            "dep_35L",
            "dep_36R",
            "arr_31R",
            "arr_35C",
            "arr_35R",
        ]
        for c in Xt.columns:
            self.assertEqual(
                Xt.loc[0,c],
                True if c in correct_answers_0 else False,
            )

        correct_answers_1 = [
            "dep_36R",
            "arr_31L",
        ]
        for c in Xt.columns:
            self.assertEqual(
                Xt.loc[1,c],
                True if c in correct_answers_1 else False,
            )

if __name__ == '__main__':
    unittest.main()
