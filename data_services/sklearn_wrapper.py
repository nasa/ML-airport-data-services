from functools import wraps
from itertools import chain

import pandas as pd
import numpy as np
from sklearn import preprocessing, compose, feature_selection, decomposition
from sklearn.compose._column_transformer import _get_transformer_list
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder
from .aircraft_class_encoder import AircraftClassEncoder
from .airport_config_encoder import AirportConfigEncoder

modules = (preprocessing, feature_selection, decomposition)

"""
Wrapper around sklearn.preprocessing module so that it returns pd.DataFrame.

Based on this github comment:
https://github.com/scikit-learn/scikit-learn/issues/5523#issuecomment-643351354

Special work done on implementation of wrapper for ColumnTransformer.
Other parts unchanged from github comment.

this:
from sklearn.compose import ColumnTransformer
becomes:
from sklearn_wrapper import ColumnTransformer
"""


def base_wrapper(Parent):
    class Wrapper(Parent):

        def transform(self, X, **kwargs):
            result = super().transform(X, **kwargs)
            check = self.check_out(X, result)
            return check if check is not None else result

        def fit_transform(self, X, y=None, **kwargs):
            result = super().fit_transform(X, y, **kwargs)
            check = self.check_out(X, result)
            return check if check is not None else result

        def check_out(self, X, result):
            if isinstance(X, pd.DataFrame):
                result = pd.DataFrame(result, index=X.index, columns=X.columns)
                result = result.astype(X.dtypes.to_dict())
            return result

        def __repr__(self):
            name = Parent.__name__
            tmp = super().__repr__().split('(')[1]
            return f'{name}({tmp}'

    Wrapper.__name__ = Parent.__name__
    Wrapper.__qualname__ = Parent.__name__

    return Wrapper


def base_pca_wrapper(Parent):
    Parent = base_wrapper(Parent)

    class Wrapper(Parent):
        @wraps(Parent)
        def __init__(self, *args, **kwargs):
            self._prefix_ = kwargs.pop('prefix', 'PCA')
            super().__init__(*args, **kwargs)

        def check_out(self, X, result):
            if isinstance(X, pd.DataFrame):
                columns = [f'{self._prefix_}_{i}' for i in range(1, (self.n_components or X.shape[1]) + 1)]
                result = pd.DataFrame(result, index=X.index, columns=columns)
            return result

    return Wrapper


class ColumnTransformer(base_wrapper(compose.ColumnTransformer)):
    """
    Wraps sklearn ColumnTransformer so that it returns a data frame.

    Special handling of one-hot encoders and AircraftClassEncoder.
    """

    def check_out(
        self,
        X,
        result,
    ):
        """
        Convert result of column transformer to data frame, if possible.

        Parameters
        ----------
        X :
            Input to sklearn ColumnTransformer.
            Typically will be pd.DataFrame, but does not have to be.
        result :
            Output from sklearn ColumnTransformer.
            Typically np.ndarray.

        Returns
        -------
        pd.DataFrame, if input X is a pd.DataFrame.
            Fills in column names as best it can.
            Special handling for simple ColumnTransformers that just do a
            one-hot encoding or AircraftClassEncoder.
            Rely on .categories or .category in these cases to name the
            output data frame columns appropriately.
        """
        if isinstance(X, pd.DataFrame):
            # Collect some information about the remainder, if needed
            if self._remainder[1] != 'drop':
                remainder_col_names = [*X.columns[self._remainder[-1]]]
                remainder_col_dtypes = X.dtypes.iloc[
                    self._remainder[-1]
                ].to_dict()

            # Get columns transformed
            cols_transformed = []
            for cols_list in self._columns:
                cols_transformed.extend(cols_list)
            cols_transformed = list(set(cols_transformed))

            # Get number of columns created by transfomer
            num_cols_transfomer_out = len(result[0]) -\
                (X.shape[1] - len(cols_transformed))

            # if multiple transformers or multiple input columns,
            # then only numbers for transformed column names
            # at least preserve passed through column names
            if (
                (len(self.transformers) > 1)
                or (callable(self.transformers[0][2])
                    and len(self.transformers[0][2](X)) > 1)
                or (len(self.transformers[0][2]) > 1)
            ):
                if self._remainder[1] == 'drop':
                    result_df = pd.DataFrame(
                        result,
                        index=X.index,
                        columns=[
                            str(i) for i in range(num_cols_transfomer_out)
                        ],
                    )
                else:
                    result_df = pd.DataFrame(
                        result,
                        index=X.index,
                        columns=(
                            [str(i) for i in range(num_cols_transfomer_out)] +
                            remainder_col_names
                        ),
                    )
                    result_df.loc[:, remainder_col_names] = (
                        result_df[remainder_col_names].astype(
                            X.dtypes.iloc[
                                self._remainder[-1]
                            ].to_dict()
                        )
                    )

            # In very simple single-col case, can give output column names
            # indicating which input column they came from
            # also, at least pass on the column names for passed through cols
            else:
                assert len(cols_transformed) == 1,\
                    'Unexpected number of columns (more than 1)'

                if num_cols_transfomer_out > 1:
                    if (
                        isinstance(
                            self.transformers[0][1],
                            OneHotEncoder
                        ) and
                        isinstance(
                            self.transformers[0][1].categories,
                            list
                        )
                    ):
                        # Rely on .categories,
                        # which I think needs to be specified
                        # during initialization of the encoder for this to work
                        X = X.reset_index(drop=True)

                        transformed_col_names = [
                            '{}_{}'.format(
                                cols_transformed[0],
                                c,
                            )
                            for c in self.transformers[0][1].categories[0]
                        ]

                        transformed_col_dtypes = {
                            name: np.float64
                            for name in transformed_col_names
                        }

                    elif (isinstance(
                        self.transformers[0][1],
                        AircraftClassEncoder
                    )):
                        X = X.reset_index(drop=True)

                        # Fit a dummy aircraft class encoder
                        # so can get at the .category attribute
                        tmp_ac_enc = AircraftClassEncoder(
                            self.transformers[0][1].aircraft_categories
                        )

                        tmp_ac_enc.fit(None)

                        transformed_col_names = [
                            '{}_{}'.format(
                                cols_transformed[0],
                                c,
                            )
                            for c in tmp_ac_enc.category
                        ]

                        transformed_col_dtypes = {
                            name: np.float64
                            for name in transformed_col_names
                        }

                    elif (isinstance(
                        self.transformers[0][1],
                        AirportConfigEncoder
                    )):
                        transformed_col_names = (
                            self
                            .transformers[0][1]
                            .column_names
                            )

                        transformed_col_dtypes = {
                            name: np.float64
                            for name in transformed_col_names
                        }

                    else:
                        transformed_col_names = [
                            '{}_{}'.format(cols_transformed[0], i)
                            for i in range(num_cols_transfomer_out)
                        ]
                else:
                    transformed_col_names = cols_transformed

                if self._remainder[1] == 'drop':
                    result_df = pd.DataFrame(
                        result,
                        index=X.index,
                        columns=transformed_col_names,
                    )
                else:
                    result_df = pd.DataFrame(
                        result,
                        index=X.index,
                        columns=transformed_col_names + remainder_col_names,
                    )
                    result_df.loc[:, remainder_col_names] = (
                        result_df[remainder_col_names].astype(
                            X.dtypes.iloc[
                                self._remainder[-1]
                            ].to_dict()
                        )
                    )
                    try:
                        result_df.loc[:, transformed_col_names] = (
                            result_df[transformed_col_names].astype(
                                transformed_col_dtypes
                            )
                        )
                    # when transformed_col_dtypes not defined
                    except NameError:
                        pass

            return result_df


class SelectKBest(base_wrapper(feature_selection.SelectKBest)):

    def check_out(self, X, result):
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(result, index=X.index, columns=X.columns[self.get_support()]). \
                astype(X.dtypes[self.get_support()].to_dict())


def make_column_transformer(*transformers, **kwargs):
    n_jobs = kwargs.pop('n_jobs', None)
    remainder = kwargs.pop('remainder', 'drop')
    sparse_threshold = kwargs.pop('sparse_threshold', 0.3)
    verbose = kwargs.pop('verbose', False)
    if kwargs:
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(kwargs.keys())[0]))
    transformer_list = _get_transformer_list(transformers)
    return ColumnTransformer(transformer_list, n_jobs=n_jobs,
                             remainder=remainder,
                             sparse_threshold=sparse_threshold,
                             verbose=verbose)


def __getattr__(name):
    if name not in __all__:
        return

    for module in modules:
        Parent = getattr(module, name, None)
        if Parent is not None:
            break

    if Parent is None:
        return

    if module is decomposition:
        Wrapper = base_pca_wrapper(Parent)
    else:
        Wrapper = base_wrapper(Parent)

    return Wrapper


__all__ = [*[c for c in preprocessing.__all__ if c[0].istitle()],
           *[c for c in decomposition.__all__ if c[0].istitle()],
           'SelectKBest']


def __dir__():
    tmp = dir()
    tmp.extend(__all__)
    return tmp