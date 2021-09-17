#!/usr/bin/env python

import unittest

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from data_services.FilterPipeline import FilterPipeline

X = pd.DataFrame({
    "col1":["a", "b", "c", "a", None],
    "col2":[1, None, 3, 4, 5],
    })
y = pd.Series([
    "fork",
    "spoon",
    "knife",
    "fork",
    "ladle",
    ])
model_args = {
    "strategy":"most_frequent",
    }
X_test = pd.DataFrame({
    "col1":["a"],
    "col2":[None],
    })
y_test = pd.Series([
    "fork"],
    )

class Test(unittest.TestCase):

    def test_no_rules(self):
        p = Pipeline(
            steps=[
                ("model", DummyClassifier(**model_args)),
                ],
            )
        fp = FilterPipeline(
            core_pipeline=p,
            default_response="fork",
            )
        feat_keep, feat_error_msg, target_keep, target_error_msg = fp.filter(
            X=X,
            y=y,
            )
        self.assertEqual(
            feat_keep.sum(),
            5,
            )
        self.assertEqual(
            target_keep.sum(),
            5,
            )

        fp.fit(X,y)
        res = fp.predict(X_test)
        self.assertEqual(
            res,
            "fork",
            )

    def test_add_include_rule_feat_cat(self):
        p = Pipeline(
            steps=[
                ("model", DummyClassifier(**model_args)),
                ],
            )
        fp = FilterPipeline(
            core_pipeline=p,
            default_response="fork",
            )
        fp.add_include_rule(
            "col1",
            ["a", "b",],
            )

        self.assertEqual(
            len(fp.rules),
            1,
            )

        feat_keep, feat_error_msg, target_keep, target_error_msg = fp.filter(
            X=X,
            y=y,
            )
        self.assertEqual(
            feat_keep.sum(),
            3,
            )
        self.assertEqual(
            target_keep.sum(),
            5,
            )

        fp.fit(X,y)
        res = fp.predict(X_test)
        self.assertEqual(
            res,
            "fork",
            )

    def test_add_include_rule_feat_cont(self):
        p = Pipeline(
            steps=[
                ("model", DummyClassifier(**model_args)),
                ],
            )
        fp = FilterPipeline(
            core_pipeline=p,
            default_response="fork",
            )
        fp.add_include_rule(
            "col2",
            lambda x: x>=4,
            )

        self.assertEqual(
            len(fp.rules),
            1,
            )

        feat_keep, feat_error_msg, target_keep, target_error_msg = fp.filter(
            X=X,
            y=y,
            )
        self.assertEqual(
            feat_keep.sum(),
            2,
            )
        self.assertEqual(
            target_keep.sum(),
            5,
            )

        fp.fit(X,y)
        res = fp.predict(X_test)
        self.assertEqual(
            res,
            "fork",
            )

    def test_add_include_rule_target_cat(self):
        p = Pipeline(
            steps=[
                ("model", DummyClassifier(**model_args)),
                ],
            )
        fp = FilterPipeline(
            core_pipeline=p,
            default_response="fork",
            )
        fp.add_include_rule(
            "col1",
            ["fork", "spoon",],
            rule_type="include_preds",
            )

        self.assertEqual(
            len(fp.rules),
            1,
            )

        feat_keep, feat_error_msg, target_keep, target_error_msg = fp.filter(
            X=X,
            y=y,
            )
        self.assertEqual(
            feat_keep.sum(),
            5,
            )
        self.assertEqual(
            target_keep.sum(),
            3,
            )

        fp.fit(X,y)
        res = fp.predict(X_test)
        self.assertEqual(
            res,
            "fork",
            )

    def test_scoring(self):
        p = Pipeline(
            steps=[
                ("model", DummyClassifier(**model_args)),
                ],
            )
        fp = FilterPipeline(
            core_pipeline=p,
            default_response="fork",
            )

        fp.fit(X,y)
        res = fp.predict(X_test)
        self.assertEqual(
            res,
            "fork",
            )

        score = fp.score(
            X_test,
            y_test,
            )
        self.assertEqual(
            score,
            1.0,
            )
        
if __name__ == '__main__':
    unittest.main()
