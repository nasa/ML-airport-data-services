"""SQLQueryFileDataSet loads and saves data to a local Excel file. The
underlying functionality is supported by pandas, so it supports all
allowed pandas options for loading and saving Excel files.
"""
from typing import Any, Dict

from copy import copy

import pandas as pd
import sqlalchemy
from datetime import timedelta

import logging

from kedro.io import AbstractDataSet
from kedro.extras.datasets.pandas.sql_dataset import _get_missing_module_error
from kedro.extras.datasets.pandas.sql_dataset import _get_sql_alchemy_missing_error


class SQLQueryFileDataSet(AbstractDataSet):
    """``SQLQueryFileDataSet`` loads data using a SQL query specified in a local
    file. It uses ``pandas.DataFrame`` internally, so it supports all allowed
    pandas options on ``read_sql_query``. Since Pandas uses SQLAlchemy behind
    the scenes, when instantiating ``SQLQueryDataSet`` one needs to pass
    a compatible connection string either in ``credentials`` (see the example
    code snippet below) or in ``load_args``. Connection string formats supported
    by SQLAlchemy can be found here:
    https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls
    Example:
    ::
        >>> from src.kedro_extensions.io.sqlfile_dataset import SQLQueryFileDataSet
        >>> import pandas as pd
        >>>
        >>> sqlfilepath = 'query.sql'  # query.sql contains "SELECT * FROM table_a"
        >>> credentials = {
        >>>     "con": "postgresql://username1:password1@host1/test"
        >>> }
        >>> data_set = SQLQueryFileDataSet(sqlfilepath=sqlfilepath,
        >>>                                credentials=credentials)
        >>>
        >>> sql_data = data_set.load()
        >>>
    """

    def __init__(
        self,
        sqlfilepath: str,
        credentials: Dict[str, Any],
        load_args: Dict[str, Any] = None,
        layer: str = None,
    ) -> None:
        """Creates a new ``SQLQueryFileDataSet``.
        Args:
            sqlfilepath: Path to file with sql query statement.
            credentials: A dictionary with a ``SQLAlchemy`` connection string.
                Users are supposed to provide the connection string 'con'
                through credentials. It overwrites `con` parameter in
                ``load_args`` and ``save_args`` in case it is provided. To find
                all supported connection string formats, see here:
                https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls
            load_args: Provided to underlying pandas ``read_sql_query``
                function along with the connection string.
                To find all supported arguments, see here:
                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_query.html
                To find all supported connection string formats, see here:
                https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls
            layer: The data layer according to the data engineering convention:
                https://kedro.readthedocs.io/en/stable/06_resources/01_faq.html#what-is-data-engineering-convention
        Raises:
            DataSetError: When either ``sql`` or ``con`` parameters is emtpy.
        """

        if not sqlfilepath:
            raise Exception(
                "`sqlfilepath` argument cannot be empty. Please provide a sql query file"
            )

        if not (credentials and "con" in credentials and credentials["con"]):
            raise Exception(
                "`con` argument cannot be empty. Please "
                "provide a SQLAlchemy connection string."
            )

        default_load_args = {}  # type: Dict[str, Any]

        self._load_args = (
            {**default_load_args, **load_args}
            if load_args is not None
            else default_load_args
        )

        self._sqlfilepath = sqlfilepath
        self._layer = layer
        self._load_args["sqlfilepath"] = sqlfilepath
        self._load_args["con"] = credentials["con"]

    def _describe(self) -> Dict[str, Any]:
        load_args = self._load_args.copy()
        del load_args["sqlfilepath"]
        del load_args["con"]
        return dict(sqlfilepath=self._load_args["sqlfilepath"], load_args=load_args, layer=self._layer)

    def _load(self) -> pd.DataFrame:
        try:
            with open(self._sqlfilepath, 'r') as query_file_h:
                sql = query_file_h.read()
            load_args = self._load_args.copy()
            del load_args["sqlfilepath"]
            return pd.read_sql_query(sqlalchemy.text(sql), **load_args)
        except ImportError as import_error:
            raise _get_missing_module_error(import_error)
        except ModuleNotFoundError:
            raise _get_sql_alchemy_missing_error()

    def _save(self, data: pd.DataFrame) -> None:
        raise Exception("`save` is not supported on SQLQueryDataSet")

class SQLQueryFileChunkedDataSet(AbstractDataSet):
    """``SQLQueryFileChunkedDataSet`` loads data using a SQL query specified in
    a local file. It uses ``pandas.DataFrame`` internally, so it supports all
    allowed pandas options on ``read_sql_query``. Since Pandas uses SQLAlchemy
    behind the scenes, when instantiating ``SQLQueryDataSet`` one needs to pass
    a compatible connection string either in ``credentials`` (see the example
    code snippet below) or in ``load_args``. Connection string formats
    supported by SQLAlchemy can be found here:
    https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls
    Example:
    ::
        >>> from src.kedro_extensions.io.sqlfile_dataset import SQLQueryFileChunkedDataSet
        >>> import pandas as pd
        >>>
        >>> sqlfilepath = 'query.sql'  # query.sql contains "SELECT * FROM table_a"
        >>> credentials = {
        >>>     "con": "postgresql://username1:password1@host1/test"
        >>> }
        >>> data_set = SQLQueryFileChunkedDataSet(sqlfilepath=sqlfilepath,
        >>>                                       credentials=credentials,
        >>>                                       chunk_size="1D")
        >>>
        >>> sql_data = data_set.load()
        >>>
    """

    def __init__(
        self,
        sqlfilepath: str,
        credentials: Dict[str, Any],
        load_args: Dict[str, Any],
        layer: str = None,
    ) -> None:
        """Creates a new ``SQLQueryFileDataSet``.
        Args:
            sqlfilepath: Path to file with sql query statement.
            credentials: A dictionary with a ``SQLAlchemy`` connection string.
                Users are supposed to provide the connection string 'con'
                through credentials. It overwrites `con` parameter in
                ``load_args`` and ``save_args`` in case it is provided. To find
                all supported connection string formats, see here:
                https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls
            load_args: Provided to underlying pandas ``read_sql_query``
                function along with the connection string.
                To find all supported arguments, see here:
                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_query.html
                To find all supported connection string formats, see here:
                https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls
            layer: The data layer according to the data engineering convention:
                https://kedro.readthedocs.io/en/stable/06_resources/01_faq.html#what-is-data-engineering-convention
        Raises:
            DataSetError: When either ``sql`` or ``con`` parameters is emtpy.
        """

        if not sqlfilepath:
            raise Exception(
                "`sqlfilepath` argument cannot be empty. Please provide a sql query file"
            )

        if not (credentials and "con" in credentials and credentials["con"]):
            raise Exception(
                "`con` argument cannot be empty. Please "
                "provide a SQLAlchemy connection string."
            )

        # Optionally can have params for other SQL query parameters
        # Optionally can provide chunk_size_days
        default_load_args = {'params': dict()}  # type: Dict[str, Any]

        self._load_args = (
            {**default_load_args, **load_args}
            if load_args is not None
            else default_load_args
        )

        self._sqlfilepath = sqlfilepath
        self._layer = layer
        self._load_args["sqlfilepath"] = sqlfilepath
        self._load_args["con"] = credentials["con"]

    def _describe(self) -> Dict[str, Any]:
        load_args = self._load_args.copy()
        del load_args["sqlfilepath"]
        del load_args["con"]
        return dict(
            sqlfilepath=self._load_args["sqlfilepath"],
            load_args=load_args,
            layer=self._layer
        )

    def _load(self) -> pd.DataFrame:
        try:
            with open(self._sqlfilepath, 'r') as query_file_h:
                sql = query_file_h.read()
            load_args = self._load_args.copy()

            log = logging.getLogger(__name__)
            
            log.info('SQLQueryFileChunkedDataSet: starting {}-day chunked queries with {} from {} to {}'.format(
                load_args["chunk_size_days"],
                self._sqlfilepath,
                load_args["params"]["start_time"].strftime("%Y%m%d-%H%M"),
                load_args["params"]["end_time"].strftime("%Y%m%d-%H%M")
            ))

            try:
                extend_first_start_time_hours = timedelta(hours=load_args["extend_first_start_time_hours"])
                log.info("SQLQueryFileChunkedDataSet: extending first start time to {} hours earlier".format(
                    load_args["extend_first_start_time_hours"]
                ))
                del load_args["extend_first_start_time_hours"]
            except KeyError:
                extend_first_start_time_hours = timedelta(hours=0)
            chunk_start_time = copy(load_args["params"]["start_time"]) -\
                extend_first_start_time_hours
            
            try:
                extend_last_end_time_hours = timedelta(hours=load_args["extend_last_end_time_hours"])
                log.info("SQLQueryFileChunkedDataSet: extending last end time to {} hours later".format(
                    load_args["extend_last_end_time_hours"]
                ))
                del load_args["extend_last_end_time_hours"]
            except KeyError:
                extend_last_end_time_hours = timedelta(hours=0)
            end_time = copy(load_args["params"]["end_time"]) +\
                extend_last_end_time_hours
            
            chunk = timedelta(days=load_args["chunk_size_days"])
            del load_args["chunk_size_days"]

            del load_args["sqlfilepath"]

            data = pd.DataFrame()

            while chunk_start_time < end_time:
                chunk_end_time = min(
                    [
                        chunk_start_time + chunk,
                        end_time,
                    ]
                )
                load_args["params"]["start_time"] = chunk_start_time
                load_args["params"]["end_time"] = chunk_end_time
                log.info('SQLQueryFileChunkedDataSet: running query {} with load_args {}...'.format(
                    self._sqlfilepath,
                    load_args
                ))
                data_chunk = pd.read_sql_query(sqlalchemy.text(sql), **load_args)
                data = data.append(data_chunk)
                chunk_start_time = chunk_end_time
            return data
        except ImportError as import_error:
            raise _get_missing_module_error(import_error)
        except ModuleNotFoundError:
            raise _get_sql_alchemy_missing_error()

    def _save(self, data: pd.DataFrame) -> None:
        raise Exception("`save` is not supported on SQLQueryDataSet")
