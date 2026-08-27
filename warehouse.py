from __future__ import annotations

from pathlib import Path
from typing import Protocol

import duckdb
import pandas as pd


class Warehouse(Protocol):
    name: str

    def load_source(self, table_name: str, frame: pd.DataFrame) -> None: ...

    def connection(self) -> duckdb.DuckDBPyConnection: ...

    def close(self) -> None: ...


def quote_identifier(identifier: str) -> str:
    if not identifier or "\x00" in identifier:
        raise ValueError("invalid warehouse identifier")
    return '"' + identifier.replace('"', '""') + '"'


class DuckDBWarehouse:
    name = "duckdb"

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._connection = duckdb.connect(self.path)

    def load_source(self, table_name: str, frame: pd.DataFrame) -> None:
        self._connection.register("_source_frame", frame)
        self._connection.execute(
            f"CREATE OR REPLACE TABLE {quote_identifier(table_name)} AS "
            "SELECT * FROM _source_frame"
        )
        self._connection.unregister("_source_frame")

    def connection(self) -> duckdb.DuckDBPyConnection:
        return self._connection

    def close(self) -> None:
        self._connection.close()
