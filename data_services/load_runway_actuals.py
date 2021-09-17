#!/usr/bin/env python

"""Script to load actual runways into a db for easier management
"""

import argparse
import csv
import sqlalchemy
import re

import pandas as pd

from io import StringIO
from pathlib import Path

CREATE_QUERY = """
CREATE TABLE IF NOT EXISTS public.{} (
	gufi text NULL,
	departure_runway_actual_time timestamp NULL,
	departure_runway_actual text NULL,
	distance_from_runway float8 NULL,
	points_on_runway bool NULL,
	arrival_runway_actual_time timestamp NULL,
	arrival_runway_actual text NULL,
	airport_id text NULL
    );
"""

TABLE_NAME = "XXXXX"
HOST = "XXXXX"
PORT = "XXXXX"
DB = "XXXXX"
USERNAME = "XXXXX"
PASSWORD = "XXXXX"

def main(
        file_path: str,
        table_name: str,
        host: str,
        port: str,
        db: str,
        username: str,
        password: str,
        ):
    archive = Path(file_path)

    if archive.exists():
        files_to_load = archive.glob("**/runways_*.csv")
    else:
        raise(FileNotFoundError("Cannot find archive path"))

    engine = sqlalchemy.create_engine(
        "postgresql://{}:{}@{}:{}/{}".format(
            username,
            password,
            host,
            port,
            db,
            )
        )

    with engine.connect() as conn:
        create_query_table = CREATE_QUERY.format(table_name)
        conn.execute(create_query_table)

        for file in files_to_load:
            if ("raw" not in file.parts) and ("0000" in file.name):
                print(file)

                airport = re.findall("[A-Z]{3}", file.name)[0]

                if airport in ["HNL","ANC"]:
                    prefix = "P"
                else:
                    prefix = "K"

                dat = pd.read_csv(
                    file,
                    parse_dates=[
                        "departure_runway_actual_time",
                        "arrival_runway_actual_time",
                        ]
                    )
                dat["airport_id"] = prefix + airport

                dat.to_sql(
                    table_name,
                    conn,
                    if_exists="append",
                    index=False,
                    method=psql_insert_copy,
                    )

# Sample code for implementing COPY FROM taken from:
# https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#io-sql-method
def psql_insert_copy(table, conn, keys, data_iter):
    """
    Execute SQL statement inserting data

    Parameters
    ----------
    table : pandas.io.sql.SQLTable
    conn : sqlalchemy.engine.Engine or sqlalchemy.engine.Connection
    keys : list of str
        Column names
    data_iter : Iterable that iterates the values to be inserted
    """
    # gets a DBAPI connection that can provide a cursor
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        s_buf = StringIO()
        writer = csv.writer(s_buf)
        writer.writerows(data_iter)
        s_buf.seek(0)

        columns = ', '.join('"{}"'.format(k) for k in keys)
        if table.schema:
            table_name = '{}.{}'.format(table.schema, table.name)
        else:
            table_name = table.name

        sql = 'COPY {} ({}) FROM STDIN WITH CSV'.format(
            table_name, columns)
        cur.copy_expert(sql=sql, file=s_buf)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("--table_name", default=TABLE_NAME)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--db", default=DB)
    parser.add_argument("--username", default=USERNAME)
    parser.add_argument("--password", default=PASSWORD)

    args = parser.parse_args()

    main(**vars(args))
