"""Tests for suitability criteria."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def test_attrs(criteria_anpr, sf_anpr):
    sc = criteria_anpr.compute()

    assert sc.name == "annual_precipitation"
    assert sc.attrs["weight"] == 1
    assert sc.attrs["category"] == "climate"
    assert f"func_method: {repr(sf_anpr)}" in sc.attrs["history"]
    assert "from_indicator:" in sc.attrs["history"]
    assert "name: prcptot" in sc.attrs["history"]
    assert "units: mm" in sc.attrs["history"]
    assert "standard_name: lwe_thickness_of_precipitation_amount" in sc.attrs["history"]
    assert "long_name: Total accumulated precipitation" in sc.attrs["history"]


def test_format(criteria_anpr):
    sc = criteria_anpr.compute()

    assert isinstance(sc, xr.DataArray)
    assert sc.name == "annual_precipitation"
    assert sc.dims == ("lat", "lon", "time")
    assert sc.shape == (5, 5, 5)
    np.testing.assert_equal(sc.lat.values, np.arange(5))
    np.testing.assert_equal(sc.lon.values, np.arange(5))
    np.testing.assert_equal(sc.time.values, pd.date_range("2000-01-01", periods=5, freq="YS"))


def test_compute(criteria_anpr, criteria_drain):
    # test computation
    sc = criteria_anpr.compute()
    np.testing.assert_allclose(sc.values, 0.255, atol=0.005)
    sc = criteria_drain.compute()
    np.testing.assert_equal(sc.values, 0.5)

    # test when already computed, should input indicator values
    sc = criteria_anpr
    sc.is_computed = True
    sc = sc.compute()
    np.testing.assert_equal(sc.values, 1000)
    sc = criteria_drain
    sc.is_computed = True
    sc = sc.compute()
    np.testing.assert_equal(sc.values, 3)
