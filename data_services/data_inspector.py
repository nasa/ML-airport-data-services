from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import logging
import pprint
from typing import List, Tuple


class DataInspector(BaseEstimator, TransformerMixin):
    """
    Inspect data, but do not manipulate data.

    Use logging to output inspection results.
    """

    def __init__(
        self,
        verbosity: int = 1,
        prefix: str = '',
        max_rows: int = None,
    ):
        """
        Inspect data, but do not manipulate data.

        Use logging to output inspection results.

        Parameters
        ----------
        verbosity : int, default 1
            Higher verbosity levels output more inspection results
            0 - output nothing
            1 - data type and shape, and column names for DataFrame
            2 - same as 1 but with dtypes for columns
            3 - same as 2 but also with fraction null per column
        prefix : str, default ''
            String prefix to put at start of logged inspecion outputs
        max_rows : int, default None
            Number of rows to show when printing Pandas items
            Default None is to display all

        Returns
        -------
        None.
        """
        self.verbosity = verbosity
        self.prefix = prefix
        self.max_rows = None

    def fit(
        self,
        X=None,
        y=None
    ):
        """
        Fit inspector.

        Does nothing.

        Parameters
        ----------
        X : default None
            features data set
        y : default None
            target data set

        Returns
        -------
        self : DataInspector class
        """
        return self

    def transform(
        self,
        data,
    ):
        """
        Inspect data, then return it.

        Parameters
        ----------
        data :
            data to be inspected

        Returns
        -------
        data :
            same as what was passed in
        """
        log = logging.getLogger(__name__)

        pp = pprint.PrettyPrinter(indent=4)

        pd.set_option('display.max_rows', self.max_rows)

        if self.verbosity > 0:
            if isinstance(data, np.ndarray):
                log.info('{} data type: np.array'.format(self.prefix))
                log.info('{} data shape: {}'.format(
                    self.prefix,
                    data.shape,
                ))

            elif isinstance(data, pd.DataFrame):
                log.info('{} data type: pd.DataFrame'.format(self.prefix))
                log.info('{} data shape: {}'.format(
                    self.prefix,
                    data.shape,
                ))
                if self.verbosity == 1:
                    log.info('{} data column names:\n{}'.format(
                        self.prefix,
                        pp.pformat([*data.columns]),
                    ))
                elif self.verbosity > 1:
                    log.info('{} data dtypes:\n{}'.format(
                        self.prefix,
                        pp.pformat(data.dtypes),
                    ))
                if self.verbosity > 2:
                    log.info('{} data columns fraction isnull:\n{}'.format(
                        self.prefix,
                        pp.pformat(data.isnull().mean())
                    ))
                if self.verbosity > 9:
                    log.info('{} sample data:\n{}'.format(
                        self.prefix,
                        pp.pformat(data.iloc[0]),
                    ))

            else:
                log.info('{} data type: unknown'.format(self.prefix))

        return data


def append_data_inspector(
    sklearn_Pipeline_steps: List[Tuple],
    data_inspector_verbosity: int = 0,
    data_inspector_prefix: str = '',
    data_inspector_name: str = 'data_inspector',
):
    """
    Append a data inspector step
    at the end of a list of sklearn Pipeline steps.
    """
    data_inspector = DataInspector(
        verbosity=data_inspector_verbosity,
        prefix=data_inspector_prefix,
    )
    sklearn_Pipeline_steps.append((
        data_inspector_name,
        data_inspector
    ))

    return sklearn_Pipeline_steps
