import pandas as pd
import pytest
from dask.dataframe.utils import assert_eq

from dask_expr import from_pandas


@pytest.fixture
def pdf():
    pdf = pd.DataFrame({"x": list(range(10)) * 10, "y": range(100)})
    yield pdf


@pytest.fixture
def df(pdf):
    yield from_pandas(pdf, npartitions=4)


@pytest.mark.parametrize("api", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("numeric_only", [True, False])
def test_groupby_numeric(pdf, df, api, numeric_only):
    g = df.groupby("x")
    agg = getattr(g, api)(numeric_only=numeric_only)

    expect = getattr(pdf.groupby("x"), api)(numeric_only=numeric_only)
    assert_eq(agg, expect)


def test_groupby_count(pdf, df):
    g = df.groupby("x")
    agg = g.count()

    expect = pdf.groupby("x").count()
    assert_eq(agg, expect)
