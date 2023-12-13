import numpy as np
import pytest

from dask_expr import from_pandas
from dask_expr.tests._util import _backend_library, assert_eq

lib = _backend_library()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"npartitions": 2},
        {"npartitions": 4},
        {"divisions": (0, 1, 79)},
        {"partition_size": "1kb"},
    ],
)
def test_repartition_combine_similar(kwargs):
    pdf = lib.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8] * 10, "y": 1, "z": 2})
    df = from_pandas(pdf, npartitions=3)
    query = df.repartition(**kwargs)
    query["new"] = query.x + query.y
    result = query.optimize(fuse=False)

    expected = df.repartition(**kwargs).optimize(fuse=False)
    arg1 = expected.x
    arg2 = expected.y
    expected["new"] = arg1 + arg2
    assert result._name == expected._name

    expected_pdf = pdf.copy()
    expected_pdf["new"] = expected_pdf.x + expected_pdf.y
    assert_eq(result, expected_pdf)


def test_repartition_freq():
    ts = lib.date_range("2015-01-01 00:00", "2015-05-01 23:50", freq="10min")
    pdf = lib.DataFrame(
        np.random.randint(0, 100, size=(len(ts), 4)), columns=list("ABCD"), index=ts
    )
    df = from_pandas(pdf, npartitions=1).repartition(freq="MS")

    assert_eq(df, pdf)

    assert df.divisions == (
        lib.Timestamp("2015-1-1 00:00:00"),
        lib.Timestamp("2015-2-1 00:00:00"),
        lib.Timestamp("2015-3-1 00:00:00"),
        lib.Timestamp("2015-4-1 00:00:00"),
        lib.Timestamp("2015-5-1 00:00:00"),
        lib.Timestamp("2015-5-1 23:50:00"),
    )

    assert df.npartitions == 5
