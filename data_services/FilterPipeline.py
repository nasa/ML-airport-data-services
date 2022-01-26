#!/usr/bin/env python

from __future__ import annotations

import pandas as pd
import numpy as np

from typing import Callable, Union, Any, List, Tuple

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

class FilterPipeline(Pipeline):
    def __init__(
            self,
            core_pipeline: Union[Pipeline, BaseEstimator]=Pipeline(steps=[
                    ('dummy',FunctionTransformer(validate=False)),
                    ]),
            default_response: Any=None,
            default_score_behavior: str="all",
            print_debug: bool=False,
            identifying_column: str="gufi",
            rules = pd.DataFrame(columns=["type","field","vals","error_msg",],dtype=str),
            ):
        """
        Wrapper class to apply consistent filtering criteria for rows for model
        fitting and prediction, and error tracking for filtered rows not
        satisfying all rules

        Parameters
        ----------
        core_pipeline : sklearn Pipeline or BaseEstimator
            scikit-learn (or equivalent) object that implements .fit() and
                .predict()
        default_response : Any
            Default response to return when a row does not satisfy all
                filtering rules, with type consistent with output of
                core_pipeline or function called with default_response(X)
        default_score_behavior : str, optional, default "all"
            Default mode that .score() method should employ.
            Under mode "all", include all rows in the score, including those
                that return the default value.
            Under mode "model", only include rows that satisfy the filtering
                rules in the score, ignoring prediction rules
        print_debug : bool, optional, default False
            Indicator whether various debug messages should be printed to
                stdout
        identifying_column : str, optional, default "gufi"
            Column from input data to include in output from .predict_df()
            Only really works if core_pipeline configured to ignore this data
                and not try to use it in making the prediction, and if this
                identifying data is not already stored as an index

        Returns
        -------
        None.
        """
        self.core_pipeline = core_pipeline
        self.default_response = default_response
        self.default_score_behavior = default_score_behavior
        self.print_debug = print_debug
        self.identifying_column = identifying_column
        if hasattr(self.core_pipeline, 'steps'):
            self.steps = self.core_pipeline.steps

        self.rules = rules

        
    def _default_response(self, features) :
        if callable(self.default_response) :
            return self.default_response(features)
        else :
            return self.default_response
       

    def _add_rule(
            self,
            field:str,
            vals: Union[List, Callable],
            error_msg: str,
            rule_type: str,
            ):

        self.rules = self.rules.append(
            {
                'type': rule_type,
                'field': field,
                'vals': vals,
                'error_msg': error_msg,
                },
            ignore_index=True,
            )

    def add_include_rule(
            self,
            field: str,
            vals: Union[List, Callable],
            error_msg: str='include rule failed',
            rule_type: str='include_feature',
            ):
        """
        Method to add another inclusion filtering rule. Will only utilize rows
        in subsequent processing that have any of the values in 'vals' in
        column 'field'.

        Parameters
        ----------
        field : str, optional
            Name of input data column on which to apply filtering rule
        vals : list or callable, optional, None
            List of allowable values for field, or function that will evaluate
                to True if row is to be kept
        error_msg : str, optional, default ''
            Text describring rules not satisfied
        rule_type : str, optional, default include_feature
            Argument to differentiate between rules applied to columns in the
                input data versus the target variable

        Returns
        -------
        None.
        """

        self._add_rule(
            field,
            vals,
            error_msg,
            rule_type,
            )

    def add_exclude_rule(
            self,
            field: str,
            vals: Union[List, Callable],
            error_msg: str='exclude rule failed',
            rule_type: str='exclude_feature',
            ):
        """
        Method to add another exclusion filtering rule. Will only utilize rows
        in subsequent processing that have none of the values in 'vals' in
        column 'field'.

        Parameters
        ----------
        field : str, optional, default ''
            Name of input data column on which to apply filtering rule
        vals : list or callable, optional, None
            List of allowable values for field, or function that will evaluate
                to True if row is to be removed
        error_msg : str, optional, default ''
            Text describring rules not satisfied
        rule_type : str, optional, default include_feature
            Argument to differentiate between rules applied to columns in the
                input data versus the target variable

        Returns
        -------
        None.
        """

        self._add_rule(
            field,
            vals,
            error_msg,
            rule_type,
            )

    def add_include_rule_preds(
            self,
            func: Union[List, Callable],
            error_msg: str='',
            ):
        """
        Convenience method for backward compatability to add include rules on
            target values
        """

        self._add_rule(
            "",
            func,
            error_msg,
            "include_preds",
            )

    def add_exclude_rule_preds(
            self,
            func: Union[List, Callable],
            error_msg: str = ''
            ):
        """
        Convenience method for backward compatability to add exclude rules on
            target values
        """

        self._add_rule(
            "",
            func,
            error_msg,
            "exclude_preds",
            )

    def _filter(
            self,
            X: pd.DataFrame,
            ) -> Tuple[pd.Series, pd.Series]:
        """
        Filtering method to apply inclusion / exclusion rules to some matrix
        of features, returning a boolean vector indicating whether each row
        satisfied all rules.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing data to be filtered

        Returns
        -------
        idx_keep : pd.Series
            Boolean Series indicating which rows are to be used
        idx_error_msg : pd.Series
            String Series containing an error message for rules not satisfied
        """

        idx_remove = pd.Series(
            [False] * len(X.index),
            index=X.index,
        )

        idx_error_msg = pd.Series(
            [''] * len(X.index),
            index=X.index,
        )

        include_rules = self.rules[self.rules["type"] == 'include_feature']
        for _, row in include_rules.iterrows():
            if (row.field in X.columns):

                if isinstance(row.vals, Callable):
                    idx_good_vals = pd.Series(row.vals(X[row.field]), index=X.index)
                else:
                    idx_good_vals = X[row.field].isin(row.vals)

                idx_remove = idx_remove | (~idx_good_vals)
                idx_error_msg[~idx_good_vals] = (
                    idx_error_msg[~idx_good_vals]
                    .str
                    .cat(
                        np.repeat(row.error_msg, (~idx_good_vals).sum()),
                        sep=", ",
                        )
                    )

                if self.print_debug:
                    print("Removed {} rows in field {} with error {}".format(
                        (~idx_good_vals).sum(),
                        row.field,
                        row.error_msg,
                        ))
            else:
                if self.print_debug:
                    print("Field {} not found for include rule".format(
                        row.field,
                        ))

        exclude_rules = self.rules[self.rules.type == 'exclude_feature']
        for _, row in exclude_rules.iterrows():
            if (row.field in X.columns):

                if isinstance(row.vals, Callable):
                    idx_bad_vals = pd.Series(row.vals(X[row.field]), index=X.index)
                else:
                    idx_bad_vals = X[row.field].isin(row.vals)

                idx_remove = idx_remove | idx_bad_vals
                idx_error_msg[idx_bad_vals] = (
                    idx_error_msg[idx_bad_vals]
                    .str
                    .cat(
                        np.repeat(row.error_msg, idx_bad_vals.sum()),
                        sep=", ",
                        )
                    )

                if self.print_debug:
                    print("Removed {} rows in field {} with error {}".format(
                        idx_bad_vals.sum(),
                        row.field,
                        row.error_msg,
                        ))
            else:
                if self.print_debug:
                    print("Field {} not found for exclude rule".format(
                        row.field,
                        ))

        if self.print_debug:
            print("Removed {} rows in total".format(
                idx_remove.sum(),
            ))

        idx_keep = ~idx_remove

        idx_error_msg = idx_error_msg.str.lstrip(', ')
        idx_error_msg[idx_error_msg == ''] = None

        return idx_keep, idx_error_msg

    def _filter_preds(
            self,
            preds: Union[pd.Series, np.ndarray],
            ) -> Tuple[pd.Series, pd.Series]:
        """
        Filtering method to apply inclusion / exclusion rules to predicted
        values, returning a boolean vector indicating whether predictions
        satisfied all rules.

        Parameters
        ----------
        X : pd.Series or np.ndarray
            Series containing data to be filtered

        Returns
        -------
        idx_keep : pd.Series
            Boolean Series indicating which rows are to be used
        idx_error_msg : pd.Series
            String Series containing an error message for rules not satisfied
        """

        if isinstance(preds, np.ndarray):
            preds = pd.Series(preds)

        idx_remove = pd.Series(
            [False] * len(preds),
            index=preds.index,
        )

        idx_error_msg = pd.Series(
            [''] * len(preds),
            index=preds.index,
        )

        include_rules = self.rules[self.rules.type == 'include_preds']
        for _, row in include_rules.iterrows():

            if isinstance(row.vals, Callable):
                idx_good_vals = pd.Series(row.vals(preds), index=preds.index)
            else:
                idx_good_vals = preds.isin(row.vals)

            idx_remove = idx_remove | (~idx_good_vals)
            idx_error_msg[~idx_good_vals] = (
                idx_error_msg[~idx_good_vals]
                .str
                .cat(
                    np.repeat(row.error_msg, (~idx_good_vals).sum()),
                    sep=", ",
                    )
                )

            if self.print_debug:
                print("Removed {} rows due to {}, ".format(
                    (~idx_good_vals).sum(),
                    row.error_msg,
                    ))

        exclude_rules = self.rules[self.rules.type == 'exclude_preds']
        for _, row in exclude_rules.iterrows():

            if isinstance(row.vals, Callable):
                idx_bad_vals = pd.Series(row.vals(preds), index=preds.index)
            else:
                idx_bad_vals = preds.isin(row.vals)

            idx_remove = idx_remove | idx_bad_vals
            idx_error_msg[idx_bad_vals] = (
                idx_error_msg[idx_bad_vals]
                .str
                .cat(
                    np.repeat(row.error_msg, idx_bad_vals.sum()),
                    sep=", ",
                    )
                )

            if self.print_debug:
                print("Removed {} rows due to {}".format(
                    idx_bad_vals.sum(),
                    row.error_msg,
                    ))

        if self.print_debug:
            print("Removed {} rows in total".format(
                idx_remove.sum(),
                ))

        idx_keep = ~idx_remove

        idx_error_msg = idx_error_msg.str.lstrip(', ')
        idx_error_msg[idx_error_msg == ''] = None

        return idx_keep, idx_error_msg

    def filter(
            self,
            X: pd.DataFrame=None,
            y: pd.Series=None,
            ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Public filtering method to handle both features and target

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame containing data to be filtered
        y : pd.Series
            Series containing data to be filtered

        Returns
        -------
        feat_keep : pd.Series
            DESCRIPTION.
        feat_error_msg : pd.Series
            DESCRIPTION.
        target_keep : pd.Series
            DESCRIPTION.
        target_error_msg : pd.Series
            DESCRIPTION.

        """

        feat_keep = None
        feat_error_msg = None
        if (X is not None):
            if isinstance(X, pd.DataFrame):
                feat_keep, feat_error_msg = self._filter(X)
            else:
                raise(TypeError("Unknown type {} for feature matrix".format(
                    type(X),
                    )))

        target_keep = None
        target_error_msg = None
        if (y is not None):
            if isinstance(y, pd.Series):
                target_keep, target_error_msg = self._filter_preds(y)
            else:
                raise(TypeError("Unknown type {} for target".format(
                    type(y),
                    )))

        return feat_keep, feat_error_msg, target_keep, target_error_msg

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            ) -> FilterPipeline:
        """
        Generic fit method. Incorporates logic to apply rule-based filtering
            before performing the fit on the underlying model, so that only
            rows satisfying all rules are passed to the model. Rows breaking
            any rule are simply discarded.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Truth response values

        Returns
        -------
        obj : FilterPipeline
            Trained model object
        """

        if (len(X) != len(y)):
            raise ValueError("Length of X and Y inconsistent")

        # This convoluted approach required in case indexing is not consistent
        #   between these two objects
        X_new_index = X.reset_index(drop=True)
        y_new_index = y.reset_index(drop=True)

        # Only use rows satisfying all filtering rules for model training
        [good_obs, error_msgs] = self._filter(X_new_index)
        [good_y, error_msgs_y] = self._filter_preds(y_new_index)

        if self.print_debug:
            print("training data has {} allowable values".format(
                X_new_index[good_obs & good_y].shape[0]
                ))

        self.core_pipeline.fit(X_new_index[good_obs & good_y],
                               y_new_index[good_obs & good_y])
        return self

    def _predict(
            self,
            X: pd.DataFrame,
            ) -> np.ndarray:
        """
        Generic predict method. Incorporates logic to apply rule-based
        filtering before performing the prediction, so that only rows
        satisfying all rules are passed to the model. Rows violating any rule
        are returned with a default value. The same row order as was input is
        used for the output.

        Parameters
        ----------
        X : pd.DataFrame
            Input features

        Returns
        -------
        preds : np.ndarray
            Predicted values
        error_msgs : pd.Series
            error codes assigned to returned preds
        """

        # This convoluted approach required to accomodate non-unique indices
        #   coming in with the input data. Building our own new, unique index
        #   will guarantee that we're able to return results of the same size
        #   and in the same order.
        X_new_index = X.reset_index(drop=True)
        [good_obs, error_msgs] = self._filter(X_new_index)
        empty_serie = pd.Series([self._default_response(X_new_index.iloc[0:1])]).iloc[0:0]

        # Generate predictions for rows satisfying all filtering rules
        if any(good_obs):
            real_preds = pd.Series(
                self.core_pipeline.predict(X_new_index[good_obs]),
                index=X_new_index[good_obs].index,
            )
        else:
            real_preds = empty_serie

        # Add back rows previously filtered out, and fill them in with the
        #   default value, need np.ravel in case just 1 element of dict type
        if (good_obs.all()) :
            not_real_preds = empty_serie
        else :
            # _default_response return a single instance if it is a constant
            # or if a function called when only 1 bad values
            default_values = self._default_response(X_new_index[~good_obs])
            n_values = (~good_obs).sum()
            # duplicate values is not always necessary but could be if the prediction is not scalar
            if ( (n_values == 1) or not(callable(self.default_response))) :
                default_values = [default_values]*n_values
            not_real_preds = pd.Series(default_values, index = X_new_index[~good_obs].index)


        preds = pd.concat((real_preds,
                           not_real_preds)
                          ).sort_index().values

        # Check rules on preds and replace with default value as needed
        good_preds, error_msgs_preds = self._filter_preds(preds)
        idx_bad_real_preds = good_obs & (~good_preds)

        preds[idx_bad_real_preds] = self._default_response(X_new_index[idx_bad_real_preds])
        error_msgs[idx_bad_real_preds] = error_msgs_preds[idx_bad_real_preds]

        return preds, error_msgs.values

    def predict(
            self,
            X: pd.DataFrame,
            ) -> np.ndarray:
        """
        Generic predict method. Returns a numpy array, consistent with typical
            scikit-learn behavior.

        Parameters
        ----------
        X : pd.DataFrame
            Input features

        Returns
        -------
        preds : np.ndarray
            Predicted values
        """

        preds, _ = self._predict(X)

        return preds

    def predict_df(
            self,
            X: pd.DataFrame,
            ) -> pd.DataFrame:
        """
        Predict method returning predictions, error messages for rows not
            satisfying all rules and index of input data frame, if available.

        Parameters
        ----------
        X : pd.DataFrame
            Input features

        Returns
        -------
        preds : pd.DataFrame
            Data frame including predictions, error messages and index, if
                available
        """

        preds, error_msg = self._predict(X)

        preds_df = pd.DataFrame(
            data={
                'pred': preds,
                'error_msg': error_msg
                },
            index=X.index,
            )

        if self.identifying_column in X.columns:
            preds_df[self.identifying_column] = (
                X[self.identifying_column].values
                )

        return preds_df

    def predict_proba(
            self,
            X: pd.DataFrame,
            ) -> pd.DataFrame:

        # This convoluted approach required to accomodate non-unique indices
        #   coming in with the input data. Building our own new, unique index
        #   will guarantee that we're able to return results of the same size
        #   and in the same order.
        X_new_index = X.reset_index(drop=True)
        [good_obs, error_msgs] = self._filter(X_new_index)

        # Generate predictions for rows satisfying all filtering rules
        if any(good_obs):
            real_preds = pd.DataFrame(
                self.core_pipeline.predict_proba(X_new_index[good_obs]),
                index=X_new_index[good_obs].index,
                columns=self.core_pipeline.classes_,
            )
        else:
            real_preds = pd.DataFrame(
                dtype=float,
                index=X_new_index[good_obs].index,
                columns=self.core_pipeline.classes_,
            )

        # Add back rows previously filtered out, and fill them in with the
        #   default value
        preds = (
            real_preds
            .reindex(
                index=X_new_index.index,
                fill_value=0.0,
                )
            )

        not_real_preds = pd.Series(self._default_response(X_new_index[~good_obs]),
                                   index=X_new_index[~good_obs].index)
        if (not_real_preds.isin(preds.columns).all()) :
            stack_preds = preds.stack()
            stack_preds.loc[zip(not_real_preds.index,
                                not_real_preds
                                )] = 1.0
            preds = stack_preds.unstack()
        else:
            print("WARNING: FilterPipeline default_response not in set of known targets")

        return preds

    def score(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            ) -> float:
        """
        Generic score method. Can be used to evaluate performance on entire
        sample ("all"), or only on those samples that are passed through the
        predictive model ("model"). In the former (default) case, some samples
        may use default values because filtering rules preclude their
        evaluation by the model.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Truth response values

        Returns
        -------
        score : float
            Accuracy score
        """


        if (self.default_score_behavior.lower() in ["all", "model"]):
            self.default_score_behavior = self.default_score_behavior.lower()
        else:
            raise ValueError("Unknown scoring behavior {} provided".format(
                self.default_score_behavior,
            ))

        
        if (self.default_score_behavior == "model"):
            return self.score_model(X, y)
        else:
            return self.score_all(X, y)

    def score_all(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            ) -> float:
        return self._score(X, y, "all")

    def score_model(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            ) -> float:
        return self._score(X, y, "model")

    def _score(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            score_behavior: str,
            ) -> float:

        if (len(X) != len(y)):
            raise ValueError("Length of X and Y inconsistent")

        X_new_index = X.reset_index(drop=True)
        y_new_index = y.reset_index(drop=True)

        preds, error_msgs = self._predict(X_new_index)
        preds_new_index = pd.Series(
            preds,
            index=y_new_index.index,
            )

        # Determine which rows to include in scoring
        if (score_behavior == "model"):
            obs_to_score = error_msgs.isnull()
        elif (score_behavior == "all"):
            obs_to_score = pd.Series([True]*len(preds))
        else:
            raise(ValueError("Unknown scoring approach {} found".format(
                score_behavior,
                )))

        res = (
            (y_new_index[obs_to_score] == preds_new_index[obs_to_score]).sum()
            / obs_to_score.sum()
            )

        return res
