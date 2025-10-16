# ruff: noqa: D100, D103
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import lsapy.standardize as std
from lsapy import SuitabilityCriteria, SuitabilityFunction


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
def sf_anpr() -> SuitabilityFunction:
    return SuitabilityFunction(name="vetharaniam2022_eq5", params={"a": -0.71, "b": 1100})


@pytest.fixture
def sf_gdd() -> SuitabilityFunction:
    return SuitabilityFunction(name="vetharaniam2022_eq5", params={"a": -0.55, "b": 1350})


@pytest.fixture
def sf_prd() -> SuitabilityFunction:
    return SuitabilityFunction(name="vetharaniam2022_eq5", params={"a": -9.8, "b": 0.45})


@pytest.fixture
def sf_drain() -> SuitabilityFunction:
    return SuitabilityFunction(name="discrete", params={"rules": {1: 0, 2: 0.1, 3: 0.5, 4: 0.9, 5: 1}})


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
def criteria(criteria_anpr, criteria_gdd, criteria_prd, criteria_drain) -> dict[str, SuitabilityCriteria]:
    """Returns a dictionary of all suitability criteria."""
    return {
        "annual_precipitation": criteria_anpr,
        "growing_degree_days": criteria_gdd,
        "potential_rooting_depth": criteria_prd,
        "drainage_class": criteria_drain,
    }
