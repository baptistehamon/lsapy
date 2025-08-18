"""Tests for utility functions."""

from __future__ import annotations

import pooch
import pytest
import xarray as xr

from lsapy.utils import (
    DATA_REALMS,
    _load_data,  # noqa: PLC2701
    kuri,
    load_climate_data,
    load_soil_data,
    open_data,
)
from lsapy.utils._utils import _check_realm_vars, _format_vars_names  # noqa: PLC2701


class TestLoadData:
    def test_invalid_name(self):
        with pytest.raises(
            ValueError,
            match="Invalid data name: invalid. Must be one of 'soil' or 'climate'.",
        ):
            _load_data("invalid")

    def test_load_soil(self):
        data = _load_data("soil")
        assert isinstance(data, xr.Dataset)
        assert dict(data.sizes) == {"lat": 100, "lon": 100}
        expected_vars = ["ASP", "SAL", "DRC", "PS", "PH", "PAW", "PRD", "SLP", "STN", "LUC"]
        assert all(var in data.data_vars for var in expected_vars)

    def test_load_climate(self):
        data = _load_data("climate")
        assert isinstance(data, xr.Dataset)
        assert dict(data.sizes) == {"time": 3653, "lat": 20, "lon": 20}
        expected_vars = ["hurs", "huss", "pr", "rlds", "rsds", "sfcWind", "tas", "tasmax", "tasmin"]
        assert all(var in data.data_vars for var in expected_vars)


class TestLoadSoilData:
    def test_load_soil_data(self):
        data = load_soil_data()
        assert data.equals(_load_data("soil"))


class TestLoadClimateData:
    def test_load_climate_data(self):
        data = load_climate_data()
        assert data.equals(_load_data("climate"))


class TestKuriPooch:
    def test_kuri(self):
        _kuri = kuri()
        assert isinstance(_kuri, pooch.Pooch)
        assert "NEX-GDDP-CMIP6_day_ACCESS-CM2_historical_r1i1p1f1_20000101-20041231.nc" in _kuri.registry
        assert "New-Zealand-Gridded-Land-Information-Dataset_NZ5km.nc" in _kuri.registry
        assert "nzglid_5km.zip" in _kuri.registry


class TestRealmVars:
    def test_errors(self):
        with pytest.raises(ValueError, match="Realm must be 'climate' or 'land', got 'invalid'."):
            _check_realm_vars("invalid")
        with pytest.raises(TypeError, match="Variable must be a string or a list of strings."):
            _check_realm_vars("climate", 123)
        with pytest.raises(ValueError, match="Variable 'invalid_var' is not supported in realm 'climate'."):
            _check_realm_vars("climate", "invalid_var")

    def test_return_none(self):
        assert _check_realm_vars("climate") is None
        assert _check_realm_vars("land") is None


class TestOpenData:
    def test_open_climate(self):
        climate_vars = _format_vars_names(DATA_REALMS["climate"])
        # all variables
        data = open_data("climate")
        assert isinstance(data, xr.Dataset)
        assert all(v in data.data_vars for v in climate_vars)
        # single variable
        data = open_data("climate", "tas")
        assert isinstance(data, xr.DataArray)
        assert "tas" in data.name

    def test_open_land(self):
        land_vars = _format_vars_names(DATA_REALMS["land"])
        # all variables
        data = open_data("land")
        assert isinstance(data, xr.Dataset)
        assert all(v in data.data_vars for v in land_vars)
        # single variable
        data = open_data("land", "slope")
        assert isinstance(data, xr.DataArray)
        assert data.name == "slope"
