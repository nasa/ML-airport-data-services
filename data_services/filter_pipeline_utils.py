"""Utilities for dealing with FilterPipelines and their rules
"""

import logging

import pandas as pd
import numpy as np
import numbers

from copy import deepcopy
from pprint import pformat
from typing import Dict, Any, Callable, List

from .FilterPipeline import FilterPipeline

"""
Utility functions for creating FilterPipeline rules,
as specified in appropriately-formatted parameters.
"""


log = logging.getLogger(__name__)


def create_exclude_func_max(
    max_val,
) -> Callable:
    """
    Return function for checking whether an input is greater than a value.

    Makes a deep copy of maximum value so pass-by-value, not reference.

    Parameters
    ----------
    max_val
        maximum value

    Returns
    -------
    Callable : function that returns True if its input is greater than max_val
    """
    max_val = deepcopy(max_val)

    return lambda x: [
            pd.notnull(v) and
            isinstance(v, numbers.Number) and
            (v > max_val) for v in x
    ]


def create_exclude_func_min(
    min_val,
) -> Callable:
    """
    Return function for checking whether an input is less than a value.

    Makes a deep copy of minimum value so pass-by-value, not reference.

    Parameters
    ----------
    min_val
        minimum value

    Returns
    -------
    Callable : function that returns True if its input is less than min_val
    """
    min_val = deepcopy(min_val)

    return lambda x: [
            pd.notnull(v) and
            isinstance(v, numbers.Number) and
            (v < min_val) for v in x
    ]


def create_exclude_func_not_numeric(
) -> Callable:
    """
    Return function for checking numeric type.

    Returns
    -------
    Callable : function that returns True if type its not numeric
    """
    return lambda x: [(isinstance(v, numbers.Number) == False) or
                            (type(v) == bool) for v in x]


def create_exclude_func_not_datetime(
) -> Callable:
    """
    Return function for checking datetime type.

    Returns
    -------
    Callable : function that returns True if type its not datetime
    """
    return lambda x: [(isinstance(v,np.datetime64) or
                        isinstance(v,pd._libs.tslibs.timestamps.Timestamp)) == False for v in x]


def create_exclude_func_not_bool(
) -> Callable:
    """
    Return function for checking bool type.

    Returns
    -------
    Callable : function that returns True if type its not bool
    """
    return lambda x: [ type(v) != bool for v in x]


def create_exclude_func_not_categorical(
) -> Callable:
    """
    Return function for checking categorical type.

    Returns
    -------
    Callable : function that returns True if type its not categorical
    """
    return lambda x: [ type(v) != str for v in x]


def add_rules_to_filter_pipeline(
    fp: FilterPipeline,
    inputs_params: Dict[str, Any],
    target_params: Dict[str, Any],
    categories: Dict[str, List[Any]]=None,
) -> FilterPipeline:
    """
    Add rules to FilterPipeline based on input and target parameters.

    Parameters
    ----------
    fp : FilterPipeline
        the FilterPipeline instance to add rules to
    inputs_params : Dict[str, Any]
        dictionary with input column names as keys
        and values that specify 'constraints' that become pipeline rules.
    target_params : Dict[str, Any]
        dictionary with 'constraints' that become pipeline rules
        for predictions.

    Returns
    -------
    FilterPipeline : with rules added
    """
    for i in inputs_params:
        if inputs_params[i]['core']:
            log.info(
                'Adding FilterPipeline inclusion rule for column {} '
                'to ensure it is non-null because it is a core input'.format(i)
            )
            fp.add_include_rule(
                i,
                lambda x: pd.notnull(x),
                '{} is core but has null value'.format(i),
            )

        if 'constraints' in inputs_params[i]:
            for c in inputs_params[i]['constraints']:
                # Include values
                if c == 'include_vals':
                    log.info(
                        'Adding FilterPipeline inclusion rule for column {} '
                        'with categories {}'.format(
                            i,
                            ','.join(inputs_params[i]['constraints'][c]),
                        )
                    )
                    include_vals = deepcopy(
                        inputs_params[i]['constraints'][c]
                    )
                    # Also put empty values here for non-core
                    if not inputs_params[i]['core']:
                        include_vals.extend([
                            np.nan,
                            '',
                            None,
                        ])
                    fp.add_include_rule(
                        i,
                        include_vals,
                        '{} not in set of valid values'.format(i),
                    )

                # Include values, learned from data
                elif c == 'include_learned':
                    if categories is not None:
                        if i in categories:
                            log.info(
                                'Adding FilterPipeline inclusion rule for column {} '
                                'with categories {}'.format(
                                    i,
                                    ','.join(categories[i]),
                                )
                            )
                            include_vals = deepcopy(
                                categories[i]
                            )
                            # Also put empty values here for non-core
                            if not inputs_params[i]['core']:
                                include_vals.extend([
                                    np.nan,
                                    '',
                                    None,
                                ])
                            fp.add_include_rule(
                                i,
                                include_vals,
                                '{} not in set of valid values'.format(i),
                            )
                        else:
                            raise(KeyError(f"No categories provided for {i}"))

                    # If categories is null, then try to rely on allowable vals
                    else:
                        if 'allowable_vals' in inputs_params[i]['constraints']:
                            allowed = inputs_params[i]['constraints']['allowable_vals']

                            log.info(
                                'Adding FilterPipeline inclusion rule for column {} '
                                'with categories {}'.format(
                                    i,
                                    ','.join(allowed),
                                )
                            )
                            include_vals = deepcopy(
                                allowed
                            )
                            # Also put empty values here for non-core
                            if not inputs_params[i]['core']:
                                include_vals.extend([
                                    np.nan,
                                    '',
                                    None,
                                ])
                            fp.add_include_rule(
                                i,
                                include_vals,
                                '{} not in set of valid values'.format(i),
                            )

                        # Otherwise, add no constraints
                        else:
                            pass

                # Exclude values
                elif c == 'exclude_vals':
                    log.info(
                        'Adding FilterPipeline exclusion rule for column {} '
                        'with categories {}'.format(
                            i,
                            ','.join(inputs_params[i]['constraints'][c]),
                        )
                    )
                    fp.add_exclude_rule(
                        i,
                        deepcopy(inputs_params[i]['constraints'][c]),
                        '{} in set of excluded values'.format(i),
                    )

                # Min
                elif c == 'min':
                    log.info(
                        'Adding FilterPipeline exclusion rule for column {} '
                        'for minimum value {}'.format(
                            i,
                            inputs_params[i]['constraints'][c],
                        )
                    )

                    exclude_func = create_exclude_func_min(
                        inputs_params[i]['constraints'][c]
                    )

                    fp.add_exclude_rule(
                        i,
                        exclude_func,
                        '{} below minimum value {}'.format(
                            i,
                            inputs_params[i]['constraints'][c]
                        ),
                    )

                # Max
                elif c == 'max':
                    log.info(
                        'Adding FilterPipeline exclusion rule for column {} '
                        'for maximum value {}'.format(
                            i,
                            inputs_params[i]['constraints'][c],
                        )
                    )

                    exclude_func = create_exclude_func_max(
                        inputs_params[i]['constraints'][c]
                    )

                    fp.add_exclude_rule(
                        i,
                        exclude_func,
                        '{} above maximum value {}'.format(
                            i,
                            inputs_params[i]['constraints'][c]
                        ),
                    )

                # List of allowable values
                elif c == 'allowable_vals':
                    # Take no action, as this list is referenced along with the
                    #   learned values
                    pass

                else:
                    raise(ValueError(
                        'Unknown constraint type {} specified'.format(c)
                    ))

    # target rules
    if 'constraints' in target_params:
        for c in target_params['constraints']:
            if c == 'min':
                log.info(
                    'Adding FilterPipeline exclusion rule for target '
                    'for minimum value {}'.format(
                        target_params['constraints'][c],
                    )
                )

                exclude_func = create_exclude_func_min(
                    target_params['constraints'][c],
                )

                fp.add_exclude_rule(
                    '',
                    exclude_func,
                    'target {} below minimum value {}'.format(
                        target_params['name'],
                        target_params['constraints'][c],
                    ),
                    'exclude_preds',
                )

            elif c == 'max':
                log.info(
                    'Adding FilterPipeline exclusion rule for target '
                    'for maximum value {}'.format(
                        target_params['constraints'][c],
                    )
                )

                exclude_func = create_exclude_func_max(
                    target_params['constraints'][c],
                )

                fp.add_exclude_rule(
                    '',
                    exclude_func,
                    'target {} above maximum value {}'.format(
                        target_params['name'],
                        target_params['constraints'][c],
                    ),
                    'exclude_preds',
                )

            elif c == 'include_vals':
                log.info(
                    'Adding FilterPipeline inclusion rule for target '
                    'with categories {}'.format(
                        ','.join(target_params['constraints'][c]),
                    )
                )

                include_vals = deepcopy(
                    target_params['constraints'][c]
                )

                fp.add_include_rule(
                    '',
                    include_vals,
                    'target {} not in set of valid values'.format(
                        target_params['name'],
                    ),
                    'include_preds',
                )

            # Include values, learned from data
            elif c == 'include_learned':
                if categories is not None:
                    if "target" in categories:
                        log.info(
                            'Adding FilterPipeline inclusion rule for target '
                            'with categories {}'.format(
                                ','.join(categories["target"]),
                            )
                        )
                        include_vals = deepcopy(
                            categories["target"]
                        )
                        fp.add_include_rule(
                            "",
                            include_vals,
                            'target {} not in set of valid values'.format(
                                target_params['name'],
                            ),
                            "include_preds",
                        )
                    else:
                        raise(KeyError("No categories provided for target"))
                else:
                    log.info("ALERT: category list is null")

            else:
                raise(ValueError(
                    'Unknown constraint type {} specified for target'.format(c)
                ))

    return fp

def get_feature_categories(
        dat: pd.DataFrame,
        inputs: Dict[str, Any],
        target: Dict[str, Any],
        groups: List[str] = ['train'],
        sample_column: str = "train_sample",
        ) -> Dict[str, List[Any]]:
    """
    Parameters
    ----------
    dat : pd.DataFrame
        Full test/train dataset
    inputs : Dict[str, Any]
        Standard input specification
    target : Dict[str, Any]
        Standard target specification

    Returns
    -------
    categories : Dict[str, List[Any]]
        List of allowable values for categorical variables
    """

    log.info("Identifying feature categories")

    if ("group" in dat.columns):
        dat = dat[dat["group"].isin(groups)]

    if (sample_column in dat.columns):
        dat = dat[dat[sample_column] == True]

    fp = add_rules_to_filter_pipeline(
        FilterPipeline(),
        inputs,
        target,
        )

    X = dat[inputs.keys()].copy()
    y = dat[target["name"]].copy()

    X_train_keep_idx, _, y_train_keep_idx, _ = fp.filter(X=X, y=y)
    good_idx = X_train_keep_idx & y_train_keep_idx
    X_train_filtered = X[good_idx]
    y_train_filtered = y[good_idx]

    # Build lists of values for each feature with OneHotEncoder
    categories = dict()
    for feat in inputs:
        if (("encoder" in inputs[feat])
            and (inputs[feat]["encoder"] == "OneHotEncoder")):
            vals = X_train_filtered[feat].dropna().unique()
            vals = [x for x in vals
                    if (x!="")]
            categories[feat] = vals

    # Compute known categories for target, if applicable
    if (target["type"] == "str"):
        categories["target"] = y_train_filtered.unique().tolist()

    log.info("Identified the following categories")
    log.info("\n{}".format(pformat(categories)))

    return categories
