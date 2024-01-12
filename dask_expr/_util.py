from __future__ import annotations

import functools
from collections import OrderedDict, UserDict
from collections.abc import Hashable, Sequence
from types import LambdaType
from typing import Any, Literal, NoReturn, TypeVar, cast

import dask
import numpy as np
import pandas as pd
from dask import config
from dask.base import normalize_object, normalize_token, tokenize
from dask.dataframe._compat import is_string_dtype
from dask.utils import get_default_shuffle_method
from packaging.version import Version
from pandas.api.types import is_datetime64_dtype, is_numeric_dtype

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

DASK_VERSION = Version(dask.__version__)
DASK_GT_20231201 = DASK_VERSION > Version("2023.12.1")


def _calc_maybe_new_divisions(df, periods, freq):
    """Maybe calculate new divisions by periods of size freq

    Used to shift the divisions for the `shift` method. If freq isn't a fixed
    size (not anchored or relative), then the divisions are shifted
    appropriately.

    Returning None, indicates divisions ought to be cleared.

    Parameters
    ----------
    df : dd.DataFrame, dd.Series, or dd.Index
    periods : int
        The number of periods to shift.
    freq : DateOffset, timedelta, or time rule string
        The frequency to shift by.
    """
    if isinstance(freq, str):
        freq = pd.tseries.frequencies.to_offset(freq)

    is_offset = isinstance(freq, pd.DateOffset)
    if is_offset:
        if freq.is_anchored() or not hasattr(freq, "delta"):
            # Can't infer divisions on relative or anchored offsets, as
            # divisions may now split identical index value.
            # (e.g. index_partitions = [[1, 2, 3], [3, 4, 5]])
            return None  # Would need to clear divisions
    if df.known_divisions:
        divs = pd.Series(range(len(df.divisions)), index=df.divisions)
        divisions = divs.shift(periods, freq=freq).index
        return tuple(divisions)
    return df.divisions


def _validate_axis(axis=0, none_is_zero: bool = True) -> None | Literal[0, 1]:
    if axis not in (0, 1, "index", "columns", None):
        raise ValueError(f"No axis named {axis}")
    # convert to numeric axis
    numeric_axis: dict[str | None, Literal[0, 1]] = {"index": 0, "columns": 1}
    if none_is_zero:
        numeric_axis[None] = 0

    return numeric_axis.get(axis, axis)


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


def is_scalar(x):
    # np.isscalar does not work for some pandas scalars, for example pd.NA
    if isinstance(x, Sequence) and not isinstance(x, str):
        return False
    elif hasattr(x, "dtype"):
        return isinstance(x, np.ScalarType)
    if isinstance(x, dict):
        return False
    if isinstance(x, (str, int)) or x is None:
        return True

    from dask_expr._expr import Expr

    return not isinstance(x, Expr)


def is_valid_nth_dtype(dtype):
    return is_numeric_dtype(dtype) or is_datetime64_dtype(dtype)


@normalize_token.register(LambdaType)
def _normalize_lambda(func):
    # Free functions also are instances of LambdaType.
    # To be more sure, check the name
    # and if cloudpickle can deterministically pickle it:
    # ref: https://github.com/cloudpipe/cloudpickle/issues/385
    func_str = str(func)
    if func.__name__ == "<lambda>" or "<locals>" in func_str:
        return func_str
    return normalize_object(func)


def _tokenize_deterministic(*args, **kwargs) -> str:
    # Utility to be strict about deterministic tokens
    with config.set({"tokenize.ensure-deterministic": True}):
        return tokenize(*args, **kwargs)


def _tokenize_partial(expr, ignore: list | None = None) -> str:
    # Helper function to "tokenize" the operands
    # that are not in the `ignore` list
    ignore = ignore or []
    return _tokenize_deterministic(
        *[
            op
            for i, op in enumerate(expr.operands)
            if i >= len(expr._parameters) or expr._parameters[i] not in ignore
        ]
    )


class LRU(UserDict[K, V]):
    """Limited size mapping, evicting the least recently looked-up key when full"""

    def __init__(self, maxsize: float) -> None:
        super().__init__()
        self.data = OrderedDict()
        self.maxsize = maxsize

    def __getitem__(self, key: K) -> V:
        value = super().__getitem__(key)
        cast(OrderedDict, self.data).move_to_end(key)
        return value

    def __setitem__(self, key: K, value: V) -> None:
        if len(self) >= self.maxsize:
            cast(OrderedDict, self.data).popitem(last=False)
        super().__setitem__(key, value)


class _BackendData:
    """Helper class to wrap backend data

    The primary purpose of this class is to provide
    caching outside the ``FromPandas`` class.
    """

    def __init__(self, data):
        self._data = data
        self._division_info = LRU(10)

    @functools.cached_property
    def _token(self):
        from dask_expr._util import _tokenize_deterministic

        return _tokenize_deterministic(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, key: str) -> Any:
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            # Return the underlying backend attribute
            return getattr(self._data, key)

    def __reduce__(self):
        return type(self), (self._data,)


@normalize_token.register(_BackendData)
def normalize_data_wrapper(data):
    return data._token


def _maybe_from_pandas(dfs):
    from dask_expr import from_pandas

    dfs = [
        from_pandas(df, 1) if isinstance(df, (pd.Series, pd.DataFrame)) else df
        for df in dfs
    ]
    return dfs


def _get_shuffle_preferring_order(shuffle):
    if shuffle is not None:
        return shuffle

    # Choose tasks over disk since it keeps the order
    shuffle = get_default_shuffle_method()
    if shuffle == "disk":
        return "tasks"

    return shuffle


def _raise_if_object_series(x, funcname):
    """
    Utility function to raise an error if an object column does not support
    a certain operation like `mean`.
    """
    if x.ndim == 1 and hasattr(x, "dtype"):
        if x.dtype == object:
            raise ValueError("`%s` not supported with object series" % funcname)
        elif is_string_dtype(x):
            raise ValueError("`%s` not supported with string series" % funcname)


class RaiseAttributeError:
    """Method or property defined on superclass, but not on subclass.

    Usage::

        class A:
            def x(self): ...

        class B(A):
            x = RaiseAttributeError()
    """

    name: str

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, instance: object | None, owner: type) -> NoReturn:
        raise AttributeError(
            f"{owner.__name__!r} object has no attribute {self.name!r}"
        )


def _is_any_real_numeric_dtype(arr_or_dtype):
    try:
        from pandas.api.types import is_any_real_numeric_dtype

        return is_any_real_numeric_dtype(arr_or_dtype)
    except ImportError:
        # Temporary/soft pandas<2 support to enable cudf dev
        # TODO: Remove `try` block after 4/2024
        from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

        return (
            is_numeric_dtype(arr_or_dtype)
            and not is_complex_dtype(arr_or_dtype)
            and not is_bool_dtype(arr_or_dtype)
        )
