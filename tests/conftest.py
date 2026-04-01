# ruff: noqa: D100, D103
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import lsapy.standardize as std
from lsapy import SuitabilityCriteria


@pytest.fixture
def annual_precip():
    """Returns annual precipitation testing data."""
    return xr.DataArray(
        np.ones((5, 5, 5)).astype(np.int32) * 1000,
        coords={"lat": range(5), "lon": range(5), "time": pd.date_range("2000-01-01", periods=5, freq="YS")},
        dims=["lat", "lon", "time"],
        name="prcptot",
        attrs={
            "units": "mm",
            "standard_name": "lwe_thickness_of_precipitation_amount",
            "long_name": "Total accumulated precipitation",
        },
    )


@pytest.fixture
def growing_degree_days():
    """Returns growing degree days testing data."""
    return xr.DataArray(
        np.ones((5, 5, 5)).astype(np.int32) * 1500,
        coords={"lat": range(5), "lon": range(5), "time": pd.date_range("2000-01-01", periods=5, freq="YS")},
        dims=["lat", "lon", "time"],
        name="growing_degree_days",
        attrs={
            "units": "C days",
            "standard_name": "integral_of_air_temperature_excess_wrt_time",
            "long_name": "Cumulative sum of temperature degrees for mean daily temperature above 4.0 degc",
        },
    )


@pytest.fixture
def potential_rooting_depth():
    """Returns potential rooting depth testing data."""
    return xr.DataArray(
        np.ones((5, 5)) * 0.9,
        coords={"lat": range(5), "lon": range(5)},
        dims=["lat", "lon"],
        name="potential_rooting_depth",
        attrs={
            "units": "m",
            "long_name": "Potential rooting depth",
        },
    )


@pytest.fixture
def drainage():
    """Returns drainage class testing data."""
    return xr.DataArray(
        np.ones((5, 5)) * 3,
        coords={"lat": range(5), "lon": range(5)},
        dims=["lat", "lon"],
        name="drainage",
        attrs={
            "units": "",
            "long_name": "Drainage Class",
            "flag_values": "1, 2, 3, 4, 5",
            "flag_meanings": "very-poor poor imperfect moderately-well well",
        },
    )


@pytest.fixture
def indicators(annual_precip, growing_degree_days, potential_rooting_depth, drainage):
    """Returns a dataset of all testing data."""
    ds = xr.merge([annual_precip, growing_degree_days, potential_rooting_depth, drainage])
    ds.attrs = {}
    return ds


@pytest.fixture
def criteria_anpr(annual_precip) -> SuitabilityCriteria:
    return SuitabilityCriteria(
        name="annual_precipitation",
        category="climate",
        indicator=annual_precip,
        weight=1,
        func=std.vetharaniam2022_eq5,
        fparams={"a": -0.71, "b": 1100},
    )


@pytest.fixture
def anpr_attrs() -> dict:
    return {
        "weight": 1.0,
        "category": "climate",
        "history": "func_method: functools.partial(<function vetharaniam2022_eq5 at 0x...>, a=-0.71, b=1100); "
        "from_indicator: [name: prcptot; units: mm; standard_name: lwe_thickness_of_precipitation_amount; "
        "long_name: Total accumulated precipitation]",
    }


@pytest.fixture
def criteria_gdd(growing_degree_days) -> SuitabilityCriteria:
    return SuitabilityCriteria(
        name="growing_degree_days",
        category="climate",
        indicator=growing_degree_days,
        weight=3,
        func=std.vetharaniam2022_eq5,
        fparams={"a": -0.55, "b": 1350},
    )


@pytest.fixture
def gdd_attrs() -> dict:
    return {
        "weight": 3.0,
        "category": "climate",
        "history": "func_method: functools.partial(<function vetharaniam2022_eq5 at 0x...>, a=-0.55, b=1350); "
        "from_indicator: [name: growing_degree_days; units: C days; "
        "standard_name: integral_of_air_temperature_excess_wrt_time; "
        "long_name: Cumulative sum of temperature degrees for mean daily temperature above 4.0 degc]",
    }


@pytest.fixture
def criteria_prd(potential_rooting_depth) -> SuitabilityCriteria:
    return SuitabilityCriteria(
        name="potential_rooting_depth",
        category="soilTerrain",
        indicator=potential_rooting_depth,
        weight=2,
        func=std.vetharaniam2022_eq5,
        fparams={"a": -9.8, "b": 0.45},
    )


@pytest.fixture
def prd_attrs() -> dict:
    return {
        "weight": 2.0,
        "category": "soilTerrain",
        "history": "func_method: functools.partial(<function vetharaniam2022_eq5 at 0x...>, a=-9.8, b=0.45); "
        "from_indicator: [name: potential_rooting_depth; units: m; long_name: Potential rooting depth]",
    }


@pytest.fixture
def criteria_drain(drainage) -> SuitabilityCriteria:
    return SuitabilityCriteria(
        name="drainage_class",
        category="soilTerrain",
        indicator=drainage,
        weight=2,
        func=std.discrete,
        fparams={"rules": {1: 0, 2: 0.1, 3: 0.5, 4: 0.9, 5: 1}},
    )


@pytest.fixture
def drain_attrs() -> dict:
    return {
        "weight": 2.0,
        "category": "soilTerrain",
        "history": "func_method: functools.partial(<function discrete at 0x...>, "
        "rules={1: 0, 2: 0.1, 3: 0.5, 4: 0.9, 5: 1}); from_indicator: [name: drainage; units: ; "
        "long_name: Drainage Class; flag_values: 1, 2, 3, 4, 5; "
        "flag_meanings: very-poor poor imperfect moderately-well well]",
    }


@pytest.fixture
def criteria(criteria_anpr, criteria_gdd, criteria_prd, criteria_drain) -> dict[str, SuitabilityCriteria]:
    """Returns a dictionary of all suitability criteria."""
    return {
        "annual_precipitation": criteria_anpr,
        "growing_degree_days": criteria_gdd,
        "potential_rooting_depth": criteria_prd,
        "drainage_class": criteria_drain,
    }


@pytest.fixture
def assert_criteria_attrs():
    """Fixture to assert that the attributes of a suitability criteria match the expected attributes."""

    def _assert_criteria_attrs(res_attrs, expected_attrs):
        history = res_attrs.pop("history", None)  # Remove history for comparison
        assert res_attrs == {k: v for k, v in expected_attrs.items() if k != "history"}
        history = re.sub(r"at 0x[0-9a-fA-F]+", "at 0x...", history)  # Normalize memory addresses in history
        assert history == expected_attrs.get("history", None)

    return _assert_criteria_attrs
