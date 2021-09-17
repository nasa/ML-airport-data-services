#!/usr/bin/env python

import unittest

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from data_services.FilterPipeline import FilterPipeline
from data_services.RunwayModelWrapper import RunwayModelWrapper

X = pd.DataFrame({
    "airport_configuration_name":[
        "D_17L_A_18R",
        "D_17L_A_18R",
        "D_17L_A_18R",
        "D_17L_A_18R",
    ],
    })

class Test(unittest.TestCase):

    def test_no_subs(self):
        rmw = RunwayModelWrapper(
            core_pipeline=Pipeline(
                steps=[
                    ("model", DummyClassifier(strategy="constant", constant="17L")),
                ],
            ),
            default_response="17R",
            operation="dep",
        )

        this_y = pd.Series([
            "17L",
            "17L",
            "17L",
            "17L",
        ])
        
        rmw.fit(X, this_y)
    
        res = rmw.predict_df(X)
        self.assertEqual(
            len(res),
            4,
        )
        self.assertTrue(
            (res["pred"]=="17L").all()
        )

    def test_all_subs(self):
        rmw = RunwayModelWrapper(
            core_pipeline=Pipeline(
                steps=[
                    ("model", DummyClassifier(strategy="constant", constant="17R")),
                ],
            ),
            default_response="17R",
            operation="dep",
        )

        this_y = pd.Series([
            "17R",
            "17R",
            "17R",
            "17L",
        ])
        
        rmw.fit(X, this_y)
    
        res = rmw.predict_df(X)
        self.assertEqual(
            len(res),
            4,
        )
        self.assertTrue(
            (res["pred"]=="17L").all()
        )

    def test_bad_config(self):
        rmw = RunwayModelWrapper(
            core_pipeline=Pipeline(
                steps=[
                    ("model", DummyClassifier(strategy="constant", constant="17R")),
                ],
            ),
            default_response="17R",
            operation="arr",
        )

        this_X = pd.DataFrame({
            "airport_configuration_name":[
                "D_31_A",
                "D_31_A",
                "D_31_A",
                "D_31_A",
            ],
        })

        this_y = pd.Series([
            "17R",
            "17R",
            "17R",
            "17L",
        ])
        
        rmw.fit(this_X, this_y)
    
        res = rmw.predict_df(this_X)
        self.assertEqual(
            len(res),
            4,
        )
        self.assertTrue(
            (res["pred"]=="17R").all()
        )

if __name__ == '__main__':
    unittest.main()
