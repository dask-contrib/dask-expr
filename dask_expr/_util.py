from __future__ import annotations

from dask import config


def _convert_to_list(column) -> list | None:
    if column is None or isinstance(column, list):
        pass
    elif isinstance(column, tuple):
        column = list(column)
    elif hasattr(column, "dtype"):
        column = column.tolist()
    else:
        column = [column]
    return column


def _maybe_import_backend():
    if config.get("dataframe.backend", "pandas") == "cudf":
        import dask_cudf  # noqa F401
