import functools

from tlz import first
from dask import config
from dask.base import DaskMethodsMixin, named_schedulers
from dask.dataframe.core import (
    _concat,
    is_dataframe_like,
    is_series_like,
    is_index_like,
)
from fsspec.utils import stringify_path

from dask_match.expr.core import Expr


#
# Utilities to wrap Expr API
# (Helps limit boiler-plate code in collection APIs)
#


def _wrap_expr_api(*args, wrap_api=None, **kwargs):
    # Use Expr API, but convert to/from Expr objects
    assert wrap_api is not None
    result = wrap_api(
        *[arg.expr if isinstance(arg, FrameBase) else arg for arg in args],
        **kwargs,
    )
    if isinstance(result, Expr):
        return new_collection(result)
    return result


def _wrap_expr_op(self, other, op=None):
    # Wrap expr operator
    assert op is not None
    if isinstance(other, FrameBase):
        other = other.expr
    return new_collection(getattr(self.expr, op)(other))


#
# Collection classes
#


class FrameBase(DaskMethodsMixin):
    """Base class for Expr-backed Collections"""

    __dask_scheduler__ = staticmethod(
        named_schedulers.get("threads", named_schedulers["sync"])
    )
    __dask_optimize__ = staticmethod(lambda dsk, keys, **kwargs: dsk)

    def __init__(self, expr):
        self.__expr = expr

    @property
    def expr(self):
        return self.__expr

    @property
    def _meta(self):
        return self.expr._meta

    def __dask_graph__(self):
        return self.expr.__dask_graph__()

    def __dask_keys__(self):
        return self.expr.__dask_keys__()

    @property
    def dask(self):
        return self.__dask_graph__()

    def __dask_postcompute__(self):
        return _concat, ()

    def __dask_postpersist__(self):
        return from_graph, (self._meta, self.divisions, self._name)

    def __getattr__(self, key):
        try:
            # Prioritize `FrameBase` attributes
            return object.__getattribute__(self, key)
        except AttributeError as err:
            try:
                # Fall back to `expr` API
                # (Making sure to convert to/from Expr)
                val = getattr(self.expr, key)
                if callable(val):
                    return functools.partial(_wrap_expr_api, wrap_api=val)
                return val
            except AttributeError:
                # Raise original error
                raise err


# Add operator attributes
for op in [
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__truediv__",
    "__rtruediv__",
    "__lt__",
    "__rlt__",
    "__gt__",
    "__rgt__",
    "__le__",
    "__rle__",
    "__ge__",
    "__rge__",
    "__eq__",
    "__ne__",
]:
    setattr(FrameBase, op, functools.partialmethod(_wrap_expr_op, op=op))


class DataFrame(FrameBase):
    """DataFrame-like Expr Collection"""

    @property
    def index(self):
        return new_collection(self.expr.index)

    @property
    def size(self):
        return new_collection(self.expr.size)

    def __getattr__(self, key):
        try:
            # Prioritize `DataFrame` attributes
            return object.__getattribute__(self, key)
        except AttributeError as err:
            try:
                # Check if key is in columns if key
                # is not a normal attribute
                if key in self.expr._meta.columns:
                    return Series(self.expr[key])
                raise err
            except AttributeError:
                # Fall back to `BaseFrame.__getattr__`
                return super().__getattr__(key)

    def __getitem__(self, other):
        if isinstance(other, FrameBase):
            return new_collection(self.expr.__getitem__(other.expr))
        return new_collection(self.expr.__getitem__(other))

    def __repr__(self):
        return f"<dask_match.core.DataFrame: expr={self.expr}>"


class Series(FrameBase):
    """Series-like Expr Collection"""

    @property
    def index(self):
        return new_collection(self.expr.index)

    @property
    def size(self):
        return new_collection(self.expr.size)

    def __getitem__(self, other):
        if isinstance(other, FrameBase):
            return new_collection(self.expr.__getitem__(other.expr))
        return new_collection(self.expr.__getitem__(other))

    def __repr__(self):
        return f"<dask_match.core.Series: expr={self.expr}>"


class Index(Series):
    """Index-like Expr Collection"""

    def __repr__(self):
        return f"<dask_match.core.Index: expr={self.expr}>"


class Scalar(FrameBase):
    """Scalar Expr Collection"""

    def __repr__(self):
        return f"<dask_match.core.Scalar: expr={self.expr}>"

    def __dask_postcompute__(self):
        return first, ()


def new_collection(expr):
    """Create new collection from an expr"""

    meta = expr._meta
    if is_dataframe_like(meta):
        return DataFrame(expr)
    elif is_series_like(meta):
        return Series(expr)
    elif is_index_like(meta):
        return Index(expr)
    else:
        return Scalar(expr)


def optimize(collection):
    from dask_match.expr.core import optimize_expr

    return new_collection(optimize_expr(collection.expr))


def from_pandas(*args, **kwargs):
    from dask_match.expr.io.io import FromPandas

    return new_collection(FromPandas(*args, **kwargs))


def from_graph(*args, **kwargs):
    from dask_match.expr.io.io import FromGraph

    return new_collection(FromGraph(*args, **kwargs))


def read_csv(*args, **kwargs):
    from dask_match.expr.io.csv import ReadCSV

    return new_collection(ReadCSV(*args, **kwargs))


def read_parquet(
    path=None,
    columns=None,
    filters=None,
    categories=None,
    index=None,
    storage_options=None,
    use_nullable_dtypes=False,
    calculate_divisions=False,
    ignore_metadata_file=False,
    metadata_task_size=None,
    split_row_groups="infer",
    blocksize="default",
    aggregate_files=None,
    parquet_file_extension=(".parq", ".parquet", ".pq"),
    filesystem="fsspec",
    **kwargs,
):
    from dask_match.expr.io.parquet import _list_columns, ReadParquet

    if use_nullable_dtypes:
        use_nullable_dtypes = config.get("dataframe.dtype_backend")

    if hasattr(path, "name"):
        path = stringify_path(path)

    return new_collection(
        ReadParquet(
            path,
            columns=_list_columns(columns),
            filters=filters,
            categories=categories,
            index=index,
            storage_options=storage_options,
            use_nullable_dtypes=use_nullable_dtypes,
            calculate_divisions=calculate_divisions,
            ignore_metadata_file=ignore_metadata_file,
            metadata_task_size=metadata_task_size,
            split_row_groups=split_row_groups,
            blocksize=blocksize,
            aggregate_files=aggregate_files,
            parquet_file_extension=parquet_file_extension,
            filesystem=filesystem,
            **kwargs,
        )
    )
