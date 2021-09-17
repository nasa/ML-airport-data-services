"""Utility fuctions for building sklearn pipelines using ATD-2 input standards
"""

import logging

import pandas as pd

from typing import Any, Dict, List, Tuple

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from .sklearn_wrapper import ColumnTransformer
from .InputPassthroughImputer import InputPassthroughImputer
from .aircraft_class_encoder import AircraftClassEncoder
from .timedelta_total_seconds_transformer import TimeDeltaTotalSecondsTransformer
from .data_inspector import append_data_inspector
from .airport_config_encoder import AirportConfigEncoder

"""
Utility functions for assembling sklearn Pipeline steps for
imputing, encoding, and transforming inputs,
as specified in appropriately-formatted parameters.
"""


log = logging.getLogger(__name__)


VERBOSITY_THRESHOLD = 3


def assemble_impute_steps(
    inputs: Dict[str, Any],
    pipeline_inspect_data_verbosity: int = 0,
    data_inspector_verbosity: int = 0,
) -> List[Tuple]:
    """
    Assemble list of sklearn Pipeline steps for imputing missing inputs.

    Parameters
    ----------
    inputs : Dict[str, Any]
        dictionary with input column names as keys
        and values that specify how to impute them
    pipeline_inspect_data_verbosity : int, default 0
        if > VERBOSITY_THRESHOLD, put a DataInspector between all steps
    data_inspector_verbosity : int, default 0
        verbosity to use in the DataInspector

    Returns
    -------
    List[Tuple] : list of 2-dim tuples of (name, imputer) per input
    """
    sklearn_pipeline_impute_steps = []

    for i in inputs:
        if 'imputer' in inputs[i]:
            log.info('Adding imputer {} for {}'.format(
                inputs[i]['imputer'],
                i
            ))

            sklearn_pipeline_impute_steps_input = []

            if (inputs[i]['imputer'] == 'SimpleImputer'):
                imp = SimpleImputer(**inputs[i]['imputer_params'])
                sklearn_pipeline_impute_steps_input.append((
                    'impute_' + i,
                    ColumnTransformer(
                        transformers=[(
                            i,
                            imp,
                            [i]
                        )],
                        remainder='passthrough',
                        sparse_threshold=0,
                    )
                ))

            elif (inputs[i]['imputer'] == 'InputPassthroughImputer'):
                imputer_params = inputs[i]['imputer_params']
                imputer_params['pass_to_field'] = i
                imp = InputPassthroughImputer(**imputer_params)

                sklearn_pipeline_impute_steps_input.append((
                    'impute_' + i,
                    imp
                ))

            else:
                raise(ValueError('Unknown imputer type provided'))

            sklearn_pipeline_impute_steps.extend(
                sklearn_pipeline_impute_steps_input
            )

            if pipeline_inspect_data_verbosity > VERBOSITY_THRESHOLD:
                log.info('Adding data inspector after imputing {}'.format(i))
                sklearn_pipeline_impute_steps = append_data_inspector(
                    sklearn_pipeline_impute_steps,
                    data_inspector_verbosity,
                    'data after imputer for input {}'.format(i),
                    'data_inspector_after_impute_{}'.format(i),
                )

    return sklearn_pipeline_impute_steps


def assemble_encode_steps(
    dat: pd.DataFrame,
    inputs: Dict[str, Any],
    aircraft_categories: Dict[str, str] = None,
    pipeline_inspect_data_verbosity: int = 0,
    data_inspector_verbosity: int = 0,
    groups: List[str] = ['train'],
    all_categories: Dict[str, List[Any]] = None,
    known_runways: List[str] = None,
) -> List[Tuple]:
    """
    Assemble list of sklearn Pipeline steps for encoding inputs.

    Parameters
    ----------
    dat : pd.DataFrame
        Data to be imputed.
        Used here just to identify categories for one-hot encoding,
        and similar tasks.
    inputs : Dict[str, Any]
        dictionary with input column names as keys
        and values that specify how to encode them
    aircraft_categories : Dict[str, str], default None
        Aircraft details to use for AircraftClassEncoder
    pipeline_inspect_data_verbosity : int, default 0
        if > VERBOSITY_THRESHOLD, put a DataInspector between all steps
    data_inspector_verbosity : int, default 0
        verbosity to use in the DataInspector
    groups : List[str], default ['train']
        which groups in the data should be used when building encoders
    all_categories : Dict[str, List[Any]], default None
        Pre-specified one-hot encoder categories to use instead of inferring
    known_runways : List[str], default None
        Runways to use for AirportConfigEncoder

    Returns
    -------
    List[Tuple] : list of 2-dim tuples of (name, encoder) per input
    """
    if ("group" in dat.columns):
        dat = dat[dat.group.isin(groups)]

    sklearn_pipeline_encode_steps = []

    for i in inputs:
        if 'encoder' in inputs[i]:
            log.info('Adding encoder {} for {}'.format(
                inputs[i]['encoder'],
                i
            ))

            sklearn_pipeline_encode_steps_input = []

            if inputs[i]['encoder'] == 'OneHotEncoder':
                # If categories already pre-defined in parameters file, then
                #   rely on those
                if 'categories' in inputs[i]['encoder_params']:
                    log.info(f'For encoder for {i}, using categories from parameters file')
                    categories = inputs[i]['encoder_params']['categories']
                # If categories for these fields already determined, re-use
                #   that data here to ensure consistency across each piece of
                #   the model
                elif ((all_categories is not None)
                    and (i in all_categories)):
                    log.info(f'For encoder for {i}, using categories determined in prior step')
                    categories = list(all_categories[i])
                # Otherwise, just infer these values from the data
                else:
                    log.info(f'For encoder for {i}, forced to infer categories')
                    categories = list(
                        dat
                        .loc[
                            dat[i].notnull(),
                            i,
                            ]
                        .unique()
                    )

                log.info(
                    'Categories in pre-impution training data: {}'.format(
                        ','.join([str(x) for x in categories])
                    )
                )

                # Drop values, if provided
                if 'drop' in inputs[i]['encoder_params']:
                    categories = [
                        c for c in categories
                        if c not in inputs[i]['encoder_params']['drop']
                        ]
                    log.info('Dropped categories {}'.format(
                        ','.join(inputs[i]['encoder_params']['drop']),
                    ))

                # If imputer might add a placeholder fill value, make sure it 
                #   is included in the possible values
                if (("imputer_params" in inputs[i])
                    and ("fill_value" in inputs[i]["imputer_params"])
                    ):
                    categories.extend([
                        inputs[i]["imputer_params"]["fill_value"]
                        ])

                # Remove categories keyword from encoder_params, if applicable
                enc_params = {
                    k:v for k, v in inputs[i]['encoder_params'].items()
                    if k != "categories"
                }

                enc = OneHotEncoder(
                    **enc_params,
                    categories=[categories],
                )


                sklearn_pipeline_encode_steps_input.append((
                    'encode_' + i,
                    ColumnTransformer(
                        transformers=[(
                            i,
                            enc,
                            [i]
                        )],
                        remainder='passthrough',
                        sparse_threshold=0,
                    )
                ))

            elif inputs[i]['encoder'] == 'AircraftClassEncoder':
                enc = AircraftClassEncoder(aircraft_categories)

                sklearn_pipeline_encode_steps_input.append((
                    'encode_' + i,
                    ColumnTransformer(
                        transformers=[(
                            i,
                            enc,
                            [i]
                        )],
                        remainder='passthrough',
                        sparse_threshold=0,
                    )
                ))

            elif (inputs[i]["encoder"] == "AirportConfigEncoder"):
                enc = AirportConfigEncoder(known_runways=known_runways)

                sklearn_pipeline_encode_steps_input.append((
                    'encode_' + i,
                    ColumnTransformer(
                        transformers=[(
                            i,
                            enc,
                            [i]
                        )],
                        remainder='passthrough',
                        sparse_threshold=0,
                    )
                ))

            else:
                raise(ValueError('Unknown encoder type provided'))

            sklearn_pipeline_encode_steps.extend(
                sklearn_pipeline_encode_steps_input
            )

            if pipeline_inspect_data_verbosity > VERBOSITY_THRESHOLD:
                log.info('Adding data inspector after encoding {}'.format(i))
                sklearn_pipeline_encode_steps = append_data_inspector(
                    sklearn_pipeline_encode_steps,
                    data_inspector_verbosity,
                    'data after encoder for input {}'.format(i),
                    'data_inspector_after_encode_{}'.format(i),
                )

    return sklearn_pipeline_encode_steps


def assemble_transform_steps(
    inputs: Dict[str, Any],
    pipeline_inspect_data_verbosity: int = 0,
    data_inspector_verbosity: int = 0,
) -> List[Tuple]:
    """
    Assemble list of sklearn Pipeline steps for transforming inputs.

    Parameters
    ----------
    inputs : Dict[str, Any]
        dictionary with input column names as keys
        and values that specify how to transform them
    pipeline_inspect_data_verbosity : int, default 0
        if > VERBOSITY_THRESHOLD, put a DataInspector between all steps
    data_inspector_verbosity : int, default 0
        verbosity to use in the DataInspector

    Returns
    -------
    List[Tuple] : list of 2-dim tuples of (name, transformer) per input
    """
    sklearn_pipeline_transform_steps = []

    for i in inputs:
        if 'transformer' in inputs[i]:
            log.info('Adding transformer {} for {}'.format(
                inputs[i]['transformer'],
                i
            ))

            sklearn_pipeline_transform_steps_input = []

            if (
                inputs[i]['transformer'] ==
                'TimeDeltaTotalSecondsTransformer'
            ):
                tx = TimeDeltaTotalSecondsTransformer(
                    **inputs[i]['transformer_params'],
                )

                sklearn_pipeline_transform_steps_input.append((
                    'transform_' + i,
                    tx
                ))

            else:
                raise(ValueError('Unknown transformer type provided'))

            sklearn_pipeline_transform_steps.extend(
                sklearn_pipeline_transform_steps_input
            )

            if pipeline_inspect_data_verbosity > VERBOSITY_THRESHOLD:
                log.info('Adding data inspector after transforming {}'.format(
                    i
                ))
                sklearn_pipeline_transform_steps = append_data_inspector(
                    sklearn_pipeline_transform_steps,
                    data_inspector_verbosity,
                    'data after transformer for input {}'.format(i),
                    'data_inspector_after_transform_{}'.format(i),
                )

    return sklearn_pipeline_transform_steps
