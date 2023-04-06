import os

import pandas as pd
import pytest
from dask.dataframe.utils import assert_eq

from dask_match import optimize, read_parquet


def _make_file(dir, format="parquet", df=None):
    fn = os.path.join(str(dir), f"myfile.{format}")
    if df is None:
        df = pd.DataFrame({c: range(10) for c in "abcde"})
    if format == "csv":
        df.to_csv(fn)
    elif format == "parquet":
        df.to_parquet(fn)
    else:
        ValueError(f"{format} not a supported format")
    return fn


def df(fn):
    return read_parquet(fn, columns=["a", "b", "c"])


def df_bc(fn):
    return read_parquet(fn, columns=["b", "c"])


@pytest.mark.parametrize(
    "input,expected",
    [
        (
            # Add -> Mul
            lambda fn: df(fn) + df(fn),
            lambda fn: 2 * df(fn),
        ),
        (
            # Column projection
            lambda fn: df(fn)[["b", "c"]],
            lambda fn: read_parquet(fn, columns=["b", "c"]),
        ),
        (
            # Compound
            lambda fn: 3 * (df(fn) + df(fn))[["b", "c"]],
            lambda fn: 6 * df_bc(fn),
        ),
        (
            # Traverse Sum
            lambda fn: df(fn).sum()[["b", "c"]],
            lambda fn: df_bc(fn).sum(),
        ),
        (
            # Respect Sum keywords
            lambda fn: df(fn).sum(numeric_only=True)[["b", "c"]],
            lambda fn: df_bc(fn).sum(numeric_only=True),
        ),
    ],
)
def test_optimize(tmpdir, input, expected):
    fn = _make_file(tmpdir, format="parquet")
    result = optimize(input(fn))
    assert str(result.expr) == str(expected(fn).expr)


def test_head_graph_culling(tmpdir):
    fn = _make_file(tmpdir, format="parquet")
    ddf = read_parquet(fn)
    result = ddf.head(compute=False)

    assert len(optimize(result).dask) == 2
    assert_eq(optimize(result), result)


def test_predicate_pushdown(tmpdir):
    from dask_match.io import ReadParquet

    original = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5] * 10,
            "b": [0, 1, 2, 3, 4] * 10,
            "c": range(50),
            "d": [6, 7] * 25,
            "e": [8, 9] * 25,
        }
    )
    fn = _make_file(tmpdir, format="parquet", df=original)
    df = read_parquet(fn)
    assert_eq(df, original)
    x = df[df.a == 5][df.c > 20]["b"]
    y = optimize(x)
    assert isinstance(y.expr, ReadParquet)
    assert ("a", "==", 5) in y.expr.operand("filters") or (
        "a",
        "==",
        5,
    ) in y.expr.operand("filters")
    assert ("c", ">", 20) in y.expr.operand("filters")
    assert list(y.columns) == ["b"]

    # Check computed result
    y_result = y.compute()
    assert list(y_result.columns) == ["b"]
    assert len(y_result["b"]) == 6
    assert all(y_result["b"] == 4)
