import pytest

from dask_expr import from_pandas
from dask_expr._categorical import GetCategories
from dask_expr.tests._util import _backend_library, assert_eq

# Set DataFrame backend for this module
lib = _backend_library()


@pytest.fixture
def pdf():
    pdf = lib.DataFrame({"x": [1, 2, 3, 4, 1, 2], "y": "bcbbbc"})
    return pdf


@pytest.fixture
def df(pdf):
    yield from_pandas(pdf, npartitions=2)


def test_set_categories(pdf):
    pdf = pdf.astype("category")
    df = from_pandas(pdf, npartitions=2)
    assert df.x.cat.known
    assert_eq(df.x.cat.codes, pdf.x.cat.codes)
    ser = df.x.cat.as_unknown()
    assert not ser.cat.known
    ser = ser.cat.as_known()
    assert_eq(ser.cat.categories, lib.Index([1, 2, 3, 4]))
    ser = ser.cat.set_categories([1, 2, 3, 5, 4])
    assert_eq(ser.cat.categories, lib.Index([1, 2, 3, 5, 4]))
    assert not ser.cat.ordered


def test_categorize(df, pdf):
    df = df.categorize()

    assert df.y.cat.known
    assert_eq(df, pdf.astype({"y": "category"}), check_categorical=False)


def test_get_categories_simplify_adds_projection(df):
    optimized = GetCategories(
        df, columns=["y"], index=False, split_every=None
    ).simplify()
    expected = GetCategories(
        df[["y"]].simplify(), columns=["y"], index=False, split_every=None
    )
    assert optimized._name == expected._name


def test_categorical_set_index():
    df = lib.DataFrame({"x": [1, 2, 3, 4], "y": ["a", "b", "b", "c"]})
    df["y"] = lib.Categorical(df["y"], categories=["a", "b", "c"], ordered=True)
    a = from_pandas(df, npartitions=2)

    b = a.set_index("y", divisions=["a", "b", "c"], npartitions=a.npartitions)
    d1, d2 = b.get_partition(0), b.get_partition(1)
    print(d1.index.optimize(fuse=False).pprint())
    assert list(d1.index.compute(fuse=False)) == ["a"]
    assert list(sorted(d2.index.compute())) == ["b", "b", "c"]
