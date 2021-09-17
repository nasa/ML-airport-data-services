"""SQLQueryFileDataSet loads and saves data to a local Excel file. The
underlying functionality is supported by pandas, so it supports all
allowed pandas options for loading and saving Excel files.
"""
from typing import Any, Dict

from copy import deepcopy

import pandas as pd
import sqlalchemy
from datetime import timedelta
from datetime import datetime
from datetime import time
from datetime import date

import logging

from kedro.io import AbstractDataSet
from kedro.extras.datasets.pandas.sql_dataset import _get_missing_module_error
from kedro.extras.datasets.pandas.sql_dataset import _get_sql_alchemy_missing_error



class SQLQueryFileChunkedDataSetMultiCon(AbstractDataSet):
    """``SQLQueryFileChunkedDataSetMultiCon`` loads data using a SQL query specified in
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
        >>> data_set = SQLQueryFileChunkedDataSetMultiCon(sqlfilepath=sqlfilepath,
        >>>                                       credentials=credentials,
        >>>                                       chunk_size="1D")
        >>>
        >>> sql_data = data_set.load()
        >>>
    """

    def __init__(
        self,
        sqlfilepath: str,
        credentials: list,
        load_args: Dict[str, Any],
        layer: str = None,
        airport: str = None,
    ) -> None:
        """Creates a new ``SQLQueryFileDataSetMultiCon``.
        Args:
            sqlfilepath: Path to file with sql query statement.
            credentials: A dictionary with several ``SQLAlchemy`` connection strings
                and associated date range and airports
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


        if (not isinstance(credentials, list)):
            raise Exception(
                "credentials need to be a list of connection with its parameters" 
                "(date range and airport)"
            )
        
        if not (credentials[0] and "con" in credentials[0] and credentials[0]["con"]):
            raise Exception(
                "credential list needs to contain at least one connection. Please "
                "provide a SQLAlchemy connection string."
            )

        if not(load_args and "params" in load_args) :
            raise Exception(
                "no parameters for the SQL request"
                )
        
        if not("start_time" in load_args['params'] and load_args['params']["start_time"]) :
            raise Exception(
                "Please provide a start time in load_args['params']"
                )

        if not("end_time" in load_args['params'] and load_args["params"]["end_time"]) :
            raise Exception(
                "Please provide a end time in load_args['params']"
            )

        self._load_args = {'params': dict()}  # type: Dict[str, Any]
        self._load_args.update(load_args)
        self._sqlfilepath = sqlfilepath
        self._layer = layer
        self._credentials = credentials
        self._airport = airport

    def _describe(self) -> Dict[str, Any]:
        load_args = self._load_args.copy()
        return dict(
            sqlfilepath=self._sqlfilepath,
            load_args=load_args,
            layer=self._layer
        )


    def _select_airport_convert_present(self) :
        new_credentials = []
        for credential in self._credentials :
            if (self._airport in credential['airports']) or (credential['airports'] == 'NAS') :
                if (credential['end_date'] == 'present') :
                    credential['end_date'] = datetime.now()
                new_credentials.append(credential)
        self._credentials = new_credentials
        
            
    
    def _find_start_end_time_connection(self,
                                        chunk_start_time,
                                        chunk,
                                        end_time):
        con = None
        chunk_end_time = chunk_start_time  + chunk
        for credential in self._credentials :
            credential_start_time = datetime.combine(credential['start_date'],time(0)) \
                if type(credential['start_date']) == date else deepcopy(credential['start_date'])
            credential_end_time = datetime.combine(credential['end_date'],time(0)) \
                if type(credential['end_date']) == date else deepcopy(credential['end_date'])
            if (credential_start_time <= chunk_start_time) and \
               (credential_end_time > chunk_start_time) :
                con = credential['con']
                chunk_end_time = min([chunk_end_time, credential_end_time, end_time])
                

        if con is None :
            raise Exception(
                "No connection found between {} and {} at airport {}" 
                " in the credentials {}".format(chunk_start_time, chunk_end_time,
                                                self._airport, self._credentials)
            )
        return chunk_start_time, chunk_end_time, con
    
    
    def _load(self) -> pd.DataFrame:
        self._select_airport_convert_present()
        try:
            with open(self._sqlfilepath, 'r') as query_file_h:
                sql = query_file_h.read()
            load_args = deepcopy(self._load_args)

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
            chunk_start_time = deepcopy(load_args["params"]["start_time"]) -\
                extend_first_start_time_hours
            
            try:
                extend_last_end_time_hours = timedelta(hours=load_args["extend_last_end_time_hours"])
                log.info("SQLQueryFileChunkedDataSet: extending last end time to {} hours later".format(
                    load_args["extend_last_end_time_hours"]
                ))
                del load_args["extend_last_end_time_hours"]
            except KeyError:
                extend_last_end_time_hours = timedelta(hours=0)
            end_time = deepcopy(load_args["params"]["end_time"]) +\
                extend_last_end_time_hours
            
            chunk = timedelta(days=load_args["chunk_size_days"])
            del load_args["chunk_size_days"]

            # ensure start and end time are datetime not date, default to midnight if date
            if (type(chunk_start_time) == date) :
                chunk_start_time = datetime.combine(chunk_start_time,time(0))
            if (type(end_time) == date) :
                end_time = datetime.combine(end_time,time(0))

            data = []
            while chunk_start_time < end_time:
                chunk_start_time, chunk_end_time, con = self._find_start_end_time_connection(
                    chunk_start_time, chunk, end_time)
                load_args['con'] = con
                load_args["params"]["start_time"] = chunk_start_time
                load_args["params"]["end_time"] = chunk_end_time
                log.info('SQLQueryFileChunkedDataSet: running query {} with load_args {}...'.format(
                    self._sqlfilepath,
                    load_args
                ))
                data_chunk = pd.read_sql_query(sqlalchemy.text(sql), **load_args)
                data.append(data_chunk)
                chunk_start_time = chunk_end_time
            return pd.concat(data)
        except ImportError as import_error:
            raise _get_missing_module_error(import_error)
        except ModuleNotFoundError:
            raise _get_sql_alchemy_missing_error()

    def _save(self, data: pd.DataFrame) -> None:
        raise Exception("`save` is not supported on SQLQueryDataSet")
