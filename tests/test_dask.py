"""Regression tests for lazy suitability computation."""

import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback

from lsapy import LandSuitabilityAnalysis, SuitabilityCriteria


@pytest.fixture
def indicator():
    """Return an indicator with uneven chunks and a missing value."""
    values = np.arange(15, dtype=float).reshape(3, 5)
    values[0, 0] = np.nan
    return xr.DataArray(values, dims=("y", "x"), coords={"y": range(3), "x": range(5)}, name="temperature")


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("func,fparams", [("logistic", {"a": 1.0, "b": 5.0}), (np.isfinite, {})])
def test_chunked_criterion(indicator, inplace, func, fparams):
    exp = SuitabilityCriteria(name="temperature", indicator=indicator, func=func, fparams=fparams).compute()
    sc = SuitabilityCriteria(
        name="temperature", indicator=indicator.chunk({"y": 2, "x": 2}), func=func, fparams=fparams
    )
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        res = sc.compute(inplace=inplace)
    assert not tasks
    if inplace:
        assert res is None
        assert sc.is_computed
        res = sc.indicator
    assert res.chunks == ((2, 1), (2, 2, 1))
    assert res.dtype == exp.dtype
    xr.testing.assert_identical(res.compute(), exp)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dask": "allowed"},
        {"dask": "parallelized", "output_dtypes": [float]},
        {"dask_gufunc_kwargs": {"meta": np.array([], dtype=float)}},
    ],
)
def test_chunked_overrides(indicator, kwargs):
    sc = SuitabilityCriteria(
        name="temperature", indicator=indicator.chunk({"x": 2}), func="logistic", fparams={"a": 1.0, "b": 5.0}
    )
    res = sc.compute(**kwargs)
    assert res.chunks is not None
    exp = SuitabilityCriteria(
        name="temperature", indicator=indicator, func="logistic", fparams={"a": 1.0, "b": 5.0}
    ).compute()
    xr.testing.assert_identical(res.compute(), exp)


def test_chunked_forbidden(indicator):
    sc = SuitabilityCriteria(
        name="temperature", indicator=indicator.chunk({"x": 2}), func="logistic", fparams={"a": 1.0, "b": 5.0}
    )
    with pytest.raises(ValueError, match="chunked array"):
        sc.compute(dask="forbidden")


@pytest.mark.parametrize("by_category", [False, True])
def test_chunked_analysis(indicator, by_category):
    analyses = []
    for data in [indicator, indicator.chunk({"y": 2, "x": 2})]:
        criteria = {
            "temperature": SuitabilityCriteria(
                name="temperature", category="climate", indicator=data, func="logistic", fparams={"a": 1.0, "b": 5.0}
            ),
            "other": SuitabilityCriteria(
                name="other", category="soil", indicator=data + 1, func="logistic", fparams={"a": -1.0, "b": 10.0}
            ),
        }
        analyses.append(LandSuitabilityAnalysis(land_use="crop", criteria=criteria))
    exp = analyses[0].run(by_category=by_category)
    tasks = []
    with Callback(pretask=lambda key, *_: tasks.append(key)):
        lazy = analyses[1].run(by_category=by_category)
    assert not tasks
    assert all(value.chunks is not None for value in lazy.data_vars.values())
    xr.testing.assert_allclose(lazy.compute(), exp)
