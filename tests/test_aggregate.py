"""Tests for aggregation functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lsapy.aggregate import aggregate


@pytest.fixture
def data(indicators):
    inds = indicators[["prcptot", "growing_degree_days"]]
    inds["prcptot"] = xr.full_like(inds["prcptot"], 0.9, dtype=float)
    inds["growing_degree_days"] = xr.full_like(inds["growing_degree_days"], 0.7, dtype=float)
    return inds


@pytest.fixture
def median_data(data):
    return aggregate(data, method="median")


@pytest.fixture
def mean_data(data):
    return aggregate(data, method="mean")


@pytest.fixture
def wmean_data(data):
    return aggregate(data, method="wmean", weights=[3, 1])


@pytest.fixture
def gmean_data(data):
    return aggregate(data, method="gmean")


@pytest.fixture
def wgmean_data(data):
    return aggregate(data, method="wgmean", weights=[3, 1])


@pytest.fixture
def limfactor_data(data):
    return aggregate(data, method="limfactor")


class TestMedian:
    def test_format(self, median_data):
        assert isinstance(median_data, xr.DataArray)
        assert median_data.dims == ("lat", "lon", "time")
        assert median_data.shape == (5, 5, 5)
        np.testing.assert_equal(median_data.lat.values, np.arange(5))
        np.testing.assert_equal(median_data.lon.values, np.arange(5))
        np.testing.assert_equal(median_data.time.values, pd.date_range("2000-01-01", periods=5, freq="YS"))

    def test_attrs(self, median_data):
        assert median_data.name == "median"
        assert median_data.attrs["method"] == "Median"
        assert median_data.attrs["description"] == "Median of the variables: prcptot, growing_degree_days."

    def test_values(self, median_data, data):
        # multivars median
        np.testing.assert_equal(median_data.values, 0.8)
        # singlevar median
        res = aggregate(data, method="median", variables=["prcptot"])
        np.testing.assert_equal(res.values, 0.9)
        res = aggregate(data, method="median", variables=["growing_degree_days"])
        np.testing.assert_equal(res.values, 0.7)


class TestMean:
    def test_format(self, mean_data):
        assert isinstance(mean_data, xr.DataArray)
        assert mean_data.dims == ("lat", "lon", "time")
        assert mean_data.shape == (5, 5, 5)
        np.testing.assert_equal(mean_data.lat.values, np.arange(5))
        np.testing.assert_equal(mean_data.lon.values, np.arange(5))
        np.testing.assert_equal(mean_data.time.values, pd.date_range("2000-01-01", periods=5, freq="YS"))

    def test_attrs(self, mean_data):
        assert mean_data.name == "mean"
        assert mean_data.attrs["method"] == "Mean"
        assert mean_data.attrs["description"] == "Arithmetic mean of the variables: prcptot, growing_degree_days."

    def test_values(self, mean_data, data):
        # multivars mean
        np.testing.assert_equal(mean_data.values, 0.8)
        # singlevar mean
        res = aggregate(data, method="mean", variables=["prcptot"])
        np.testing.assert_equal(res.values, 0.9)
        res = aggregate(data, method="mean", variables=["growing_degree_days"])
        np.testing.assert_equal(res.values, 0.7)


class TestWeightedMean:
    def test_format(self, wmean_data):
        assert isinstance(wmean_data, xr.DataArray)
        assert wmean_data.dims == ("lat", "lon", "time")
        assert wmean_data.shape == (5, 5, 5)
        np.testing.assert_equal(wmean_data.lat.values, np.arange(5))
        np.testing.assert_equal(wmean_data.lon.values, np.arange(5))
        np.testing.assert_equal(wmean_data.time.values, pd.date_range("2000-01-01", periods=5, freq="YS"))

    def test_attrs(self, wmean_data):
        assert wmean_data.name == "weighted_mean"
        assert wmean_data.attrs["method"] == "Weighted Mean"
        assert wmean_data.attrs["description"] == (
            "Weighted mean of the variables: prcptot (3), growing_degree_days (1)."
        )

    def test_values(self, wmean_data, data, mean_data):
        # multivars weighted mean
        np.testing.assert_array_almost_equal(wmean_data.values, 0.85, decimal=2)
        # singlevar weighted mean
        res = aggregate(data, method="wmean", variables=["prcptot"], weights=[3])
        np.testing.assert_equal(res.values, 0.9)
        res = aggregate(data, method="wmean", variables=["growing_degree_days"], weights=[1])
        np.testing.assert_equal(res.values, 0.7)
        # should equal mean_data for equal weights
        res = aggregate(data, method="wmean")  # default weights are 1
        np.testing.assert_equal(res.values, mean_data.values)


class TestGeometricMean:
    def test_format(self, gmean_data):
        assert isinstance(gmean_data, xr.DataArray)
        assert gmean_data.dims == ("lat", "lon", "time")
        assert gmean_data.shape == (5, 5, 5)
        np.testing.assert_equal(gmean_data.lat.values, np.arange(5))
        np.testing.assert_equal(gmean_data.lon.values, np.arange(5))
        np.testing.assert_equal(gmean_data.time.values, pd.date_range("2000-01-01", periods=5, freq="YS"))

    def test_attrs(self, gmean_data):
        assert gmean_data.name == "geometric_mean"
        assert gmean_data.attrs["method"] == "Geometric Mean"
        assert gmean_data.attrs["description"] == "Geometric mean of the variables: prcptot, growing_degree_days."

    def test_values(self, gmean_data, data, mean_data):
        # multivars geometric mean
        np.testing.assert_array_almost_equal(gmean_data.values, 0.79, decimal=2)
        # singlevar geometric mean
        res = aggregate(data, method="gmean", variables=["prcptot"])
        np.testing.assert_equal(res.values, 0.9)
        res = aggregate(data, method="gmean", variables=["growing_degree_days"])
        np.testing.assert_equal(res.values, 0.7)


class TestWeightedGeometricMean:
    def test_format(self, wgmean_data):
        assert isinstance(wgmean_data, xr.DataArray)
        assert wgmean_data.dims == ("lat", "lon", "time")
        assert wgmean_data.shape == (5, 5, 5)
        np.testing.assert_equal(wgmean_data.lat.values, np.arange(5))
        np.testing.assert_equal(wgmean_data.lon.values, np.arange(5))
        np.testing.assert_equal(wgmean_data.time.values, pd.date_range("2000-01-01", periods=5, freq="YS"))

    def test_attrs(self, wgmean_data):
        assert wgmean_data.name == "weighted_geometric_mean"
        assert wgmean_data.attrs["method"] == "Weighted Geometric Mean"
        assert wgmean_data.attrs["description"] == (
            "Weighted geometric mean of the variables: prcptot (3), growing_degree_days (1)."
        )

    def test_values(self, wgmean_data, data, gmean_data):
        # multivars weighted geometric mean
        np.testing.assert_array_almost_equal(wgmean_data.values, 0.84, decimal=2)
        # singlevar weighted geometric mean
        res = aggregate(data, method="wgmean", variables=["prcptot"], weights=[3])
        np.testing.assert_equal(res.values, 0.9)
        res = aggregate(data, method="wgmean", variables=["growing_degree_days"], weights=[1])
        np.testing.assert_equal(res.values, 0.7)
        # should equal gmean_data for equal weights
        res = aggregate(data, method="wgmean")
        np.testing.assert_equal(res.values, gmean_data.values)


class TestLimitingFactor:
    def test_format(self, limfactor_data):
        assert isinstance(limfactor_data, xr.Dataset)
        assert dict(limfactor_data.sizes) == {"lat": 5, "lon": 5, "time": 5, "variable": 2}
        np.testing.assert_equal(limfactor_data.lat.values, np.arange(5))
        np.testing.assert_equal(limfactor_data.lon.values, np.arange(5))
        np.testing.assert_equal(limfactor_data.time.values, pd.date_range("2000-01-01", periods=5, freq="YS"))
        np.testing.assert_equal(limfactor_data.variable.values, ["prcptot", "growing_degree_days"])

    def test_attrs(self, limfactor_data):
        assert list(limfactor_data.data_vars) == ["limiting_factor", "limiting_variable"]
        assert limfactor_data.attrs["method"] == "Limiting Factor"
        assert limfactor_data.limiting_factor.attrs["description"] == (
            "Value of the limiting factor among the variables: prcptot, growing_degree_days."
        )
        assert limfactor_data.limiting_variable.attrs["description"] == (
            "Limiting variable among: prcptot, growing_degree_days."
        )

    def test_values(self, limfactor_data, data):
        # should return the minimum value across the variables
        np.testing.assert_array_equal(limfactor_data.limiting_factor.values, 0.7)
        # should return True for the variable with the minimum value
        np.testing.assert_array_equal(
            limfactor_data.limiting_variable.sel(variable="prcptot").values,
            0,  # False for prcptot
        )
        np.testing.assert_array_equal(
            limfactor_data.limiting_variable.sel(variable="growing_degree_days").values,
            1,  # True for gdd
        )
        # singlevar limiting factor
        res = aggregate(data, method="limfactor", variables=["prcptot"])
        np.testing.assert_array_equal(res.limiting_factor.values, 0.9)
        np.testing.assert_array_equal(res.limiting_variable.values, 1)  # True as only one variable
        res = aggregate(data, method="limfactor", variables=["growing_degree_days"])
        np.testing.assert_array_equal(res.limiting_factor.values, 0.7)
        np.testing.assert_array_equal(res.limiting_variable.values, 1)  # True as only one variable


class TestAggregateErrors:
    def test_invalid_method(self, data):
        with pytest.raises(
            ValueError,
            match="Invalid method 'custom'. "
            "Supported methods are: 'median', 'mean', 'wmean', 'gmean', 'wgmean', 'limfactor'.",
        ):
            _ = aggregate(data, method="custom")

    def test_invalid_variables(self, data):
        with pytest.raises(ValueError, match="'variables' must be a list of variable names or None."):
            _ = aggregate(data, method="mean", variables=123)

    def test_invalid_weights(self, data):
        # test invalid weights type
        with pytest.raises(ValueError, match="'weights' must be a list of numbers or None."):
            _ = aggregate(data, method="wmean", weights="invalid")
        # test invalid weights length
        with pytest.raises(ValueError, match="Length of 'weights' must match length of 'variables'."):
            _ = aggregate(data, method="wmean", weights=[1])
