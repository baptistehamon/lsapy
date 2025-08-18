"""Tests for utility functions."""

from __future__ import annotations

import pooch
import pytest
import xarray as xr

from lsapy.utils import (
    DATA_REALMS,
    kuri,
    open_data,
)
from lsapy.utils._utils import _check_realm_vars  # noqa: PLC2701


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
        # all variables
        data = open_data("climate")
        assert isinstance(data, xr.Dataset)
        assert all(v in data.data_vars for v in DATA_REALMS["climate"])
        # single variable
        data = open_data("climate", "tas")
        assert isinstance(data, xr.DataArray)
        assert "tas" in data.name

    def test_open_land(self):
        # all variables
        data = open_data("land")
        assert isinstance(data, xr.Dataset)
        assert all(v in data.data_vars for v in DATA_REALMS["land"])
        # single variable
        data = open_data("land", "slope")
        assert isinstance(data, xr.DataArray)
        assert data.name == "slope"
