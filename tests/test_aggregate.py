"""Tests for aggregation functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lsapy.lsa import (
    _aggregate_vars,  # noqa: PLC2701
    limiting_factor,
    vars_geomean,
    vars_mean,
    vars_weighted_geomean,
    vars_weighted_mean,
)


@pytest.fixture
def data(indicators):
    inds = indicators[["prcptot", "growing_degree_days"]]
    inds["prcptot"] = xr.full_like(inds["prcptot"], 0.9, dtype=float)
    inds["growing_degree_days"] = xr.full_like(inds["growing_degree_days"], 0.7, dtype=float)
    return inds


@pytest.fixture
def mean_data(data):
    return vars_mean(data)


@pytest.fixture
def wgtmean_data(data):
    return vars_weighted_mean(data, weights=[3, 1])


@pytest.fixture
def geomean_data(data):
    return vars_geomean(data)


@pytest.fixture
def wgtgeomean_data(data):
    return vars_weighted_geomean(data, weights=[3, 1])


@pytest.fixture
def limfactor_data(data):
    return limiting_factor(data)


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
        assert mean_data.attrs["description"] == "Mean of variables: prcptot, growing_degree_days."

    def test_values(self, mean_data, data):
        # multivars mean
        np.testing.assert_equal(mean_data.values, 0.8)
        # singlevar mean
        res = vars_mean(data, vars=["prcptot"])
        np.testing.assert_equal(res.values, 0.9)
        res = vars_mean(data, vars=["growing_degree_days"])
        np.testing.assert_equal(res.values, 0.7)


class TestWeightedMean:
    def test_format(self, wgtmean_data):
        assert isinstance(wgtmean_data, xr.DataArray)
        assert wgtmean_data.dims == ("lat", "lon", "time")
        assert wgtmean_data.shape == (5, 5, 5)
        np.testing.assert_equal(wgtmean_data.lat.values, np.arange(5))
        np.testing.assert_equal(wgtmean_data.lon.values, np.arange(5))
        np.testing.assert_equal(wgtmean_data.time.values, pd.date_range("2000-01-01", periods=5, freq="YS"))

    def test_attrs(self, wgtmean_data):
        assert wgtmean_data.name == "weighted_mean"
        assert wgtmean_data.attrs["method"] == "Weighted Mean"
        assert wgtmean_data.attrs["description"] == "Weighted Mean of variables: prcptot (3), growing_degree_days (1)."

    def test_values(self, wgtmean_data, data, mean_data):
        # multivars weighted mean
        np.testing.assert_array_almost_equal(wgtmean_data.values, 0.85, decimal=2)
        # singlevar weighted mean
        res = vars_weighted_mean(data, vars=["prcptot"], weights=[3])
        np.testing.assert_equal(res.values, 0.9)
        res = vars_weighted_mean(data, vars=["growing_degree_days"], weights=[1])
        np.testing.assert_equal(res.values, 0.7)
        # should equal mean_data for equal weights
        res = vars_weighted_mean(data)  # default weights are 1
        np.testing.assert_equal(res.values, mean_data.values)


class TestGeometricMean:
    def test_format(self, geomean_data):
        assert isinstance(geomean_data, xr.DataArray)
        assert geomean_data.dims == ("lat", "lon", "time")
        assert geomean_data.shape == (5, 5, 5)
        np.testing.assert_equal(geomean_data.lat.values, np.arange(5))
        np.testing.assert_equal(geomean_data.lon.values, np.arange(5))
        np.testing.assert_equal(geomean_data.time.values, pd.date_range("2000-01-01", periods=5, freq="YS"))

    def test_attrs(self, geomean_data):
        assert geomean_data.name == "geometric_mean"
        assert geomean_data.attrs["method"] == "Geometric Mean"
        assert geomean_data.attrs["description"] == "Geometric Mean of variables: prcptot, growing_degree_days."

    def test_values(self, geomean_data, data, mean_data):
        # multivars geometric mean
        np.testing.assert_array_almost_equal(geomean_data.values, 0.79, decimal=2)
        # singlevar geometric mean
        res = vars_geomean(data, vars=["prcptot"])
        np.testing.assert_equal(res.values, 0.9)
        res = vars_geomean(data, vars=["growing_degree_days"])
        np.testing.assert_equal(res.values, 0.7)


class TestWeightedGeometricMean:
    def test_format(self, wgtgeomean_data):
        assert isinstance(wgtgeomean_data, xr.DataArray)
        assert wgtgeomean_data.dims == ("lat", "lon", "time")
        assert wgtgeomean_data.shape == (5, 5, 5)
        np.testing.assert_equal(wgtgeomean_data.lat.values, np.arange(5))
        np.testing.assert_equal(wgtgeomean_data.lon.values, np.arange(5))
        np.testing.assert_equal(wgtgeomean_data.time.values, pd.date_range("2000-01-01", periods=5, freq="YS"))

    def test_attrs(self, wgtgeomean_data):
        assert wgtgeomean_data.name == "weighted_geometric_mean"
        assert wgtgeomean_data.attrs["method"] == "Weighted Geometric Mean"
        assert (
            wgtgeomean_data.attrs["description"]
            == "Weighted Geometric Mean of variables: prcptot (3), growing_degree_days (1)."
        )

    def test_values(self, wgtgeomean_data, data, geomean_data):
        # multivars weighted geometric mean
        np.testing.assert_array_almost_equal(wgtgeomean_data.values, 0.84, decimal=2)
        # singlevar weighted geometric mean
        res = vars_weighted_geomean(data, vars=["prcptot"], weights=[3])
        np.testing.assert_equal(res.values, 0.9)
        res = vars_weighted_geomean(data, vars=["growing_degree_days"], weights=[1])
        np.testing.assert_equal(res.values, 0.7)
        # should equal geomean_data for equal weights
        res = vars_weighted_geomean(data)  # default weights are 1
        np.testing.assert_equal(res.values, geomean_data.values)


class TestLimitingFactor:
    def test_format(self, limfactor_data):
        assert isinstance(limfactor_data, xr.Dataset)
        assert dict(limfactor_data.sizes) == {"lat": 5, "lon": 5, "time": 5}
        np.testing.assert_equal(limfactor_data.lat.values, np.arange(5))
        np.testing.assert_equal(limfactor_data.lon.values, np.arange(5))
        np.testing.assert_equal(limfactor_data.time.values, pd.date_range("2000-01-01", periods=5, freq="YS"))

    def test_attrs(self, limfactor_data):
        assert list(limfactor_data.data_vars) == ["limiting_factor", "limiting_var"]
        assert limfactor_data.attrs["method"] == "Limiting Factor"
        assert (
            limfactor_data.attrs["description"]
            == "Value of limiting factor among variables: prcptot, growing_degree_days."
        )
        assert limfactor_data.limiting_var.attrs["legend"] == {"0": "prcptot", "1": "growing_degree_days"}

    def test_values(self, limfactor_data, data):
        # should return the minimum value across the variables
        np.testing.assert_array_equal(limfactor_data.limiting_factor.values, 0.7)
        # should return the index of the variable with the minimum value
        np.testing.assert_array_equal(limfactor_data.limiting_var.values, 1)  # 1 for gdd
        # singlevar limiting factor
        res = limiting_factor(data, vars=["prcptot"])
        np.testing.assert_array_equal(res.limiting_factor.values, 0.9)
        np.testing.assert_array_equal(res.limiting_var.values, 0)
        res = limiting_factor(data, vars=["growing_degree_days"])
        np.testing.assert_array_equal(res.limiting_factor.values, 0.7)
        np.testing.assert_array_equal(res.limiting_var.values, 0)

    def test_not_limvar(self, data):
        # should return the limiting factor with no limiting variable
        res = limiting_factor(data, vars=["prcptot", "growing_degree_days"], limiting_var=False)
        assert isinstance(res, xr.Dataset)
        np.testing.assert_array_equal(res.limiting_factor.values, 0.7)
        # should not have limiting_var if limiting_var=False
        with pytest.raises(AttributeError):
            _ = res.limiting_var


class TestAggregateVars:
    def test_methods(self, data, mean_data, wgtmean_data, geomean_data, wgtgeomean_data, limfactor_data):
        # mean
        res = _aggregate_vars(data, method="mean")
        np.testing.assert_array_equal(res.values, mean_data.values)
        # weighted mean
        res = _aggregate_vars(data, method="weighted_mean", weights=[3, 1])
        np.testing.assert_array_almost_equal(res.values, wgtmean_data.values, decimal=2)
        # geometric mean
        res = _aggregate_vars(data, method="geomean")
        np.testing.assert_array_almost_equal(res.values, geomean_data.values, decimal=2)
        # weighted geometric mean
        res = _aggregate_vars(data, method="weighted_geomean", weights=[3, 1])
        np.testing.assert_array_almost_equal(res.values, wgtgeomean_data.values, decimal=2)
        # limiting factor
        res = _aggregate_vars(data, method="limiting_factor")
        np.testing.assert_array_equal(res.limiting_factor.values, limfactor_data.limiting_factor.values)
        np.testing.assert_array_equal(res.limiting_var.values, limfactor_data.limiting_var.values)
        # not implemented methods
        with pytest.raises(ValueError, match="Aggregation method 'custom' not recognized."):
            _ = _aggregate_vars(data, method="custom")

    def test_weights(self, data, mean_data, geomean_data):
        # should omit weights when not required
        res = _aggregate_vars(data, method="mean", weights=[3, 1])
        np.testing.assert_array_equal(res.values, mean_data.values)
        res = _aggregate_vars(data, method="geomean", weights=[3, 1])
        np.testing.assert_array_almost_equal(res.values, geomean_data.values, decimal=2)

    def test_kwargs(self, data):
        # kwargs should work for limiting factor
        res = _aggregate_vars(data, method="limiting_factor", limiting_var=False)
        assert isinstance(res, xr.Dataset)
        np.testing.assert_array_equal(res.limiting_factor.values, 0.7)
        # should not have limiting_var if limiting_var=False
        with pytest.raises(AttributeError):
            _ = res.limiting_var
