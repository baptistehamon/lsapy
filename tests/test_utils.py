"""Tests for utility functions."""

from __future__ import annotations

import pytest
import xarray as xr

from lsapy.utils import (
    _load_data,  # noqa: PLC2701
    load_climate_data,
    load_soil_data,
)


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
