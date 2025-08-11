"""Tests for suitability criteria."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lsapy import SuitabilityCriteria
from lsapy.criteria import _get_indicator_description  # noqa: PLC2701


class TestSuitabilityCriteria:
    def test_repr(self, criteria_anpr, criteria_drain):
        # test for annual precipitation criteria
        criteria_anpr.attrs.update(
            {
                "long_name": "Annual Precipitation",
                "description": "This is the annual precipitation criteria.",
                "comment": "Some comment about annual precipitation.",
            }
        )
        expected_repr = (
            "<SuitabilityCriteria> 'annual_precipitation'(weight: 1, category: climate)\n"
            "Function:\n"
            "    SuitabilityFunction(func=vetharaniam2022_eq5, params={'a': -0.71, 'b': 1100})\n"
            "Indicator:\n"
            "    Name          prcptot \n"
            "    Data          int32 500B 1000 1000 1000 1000 1000 ... 1000 1000 1000 1000\n"
            "    Dimensions    lat: 5, lon: 5, time: 5 \n"
            "Attributes:\n    long_name:     Annual Precipitation\n"
            "    description:   This is the annual precipitation criteria.\n"
            "    comment:       Some comment about annual precipitation."
        )
        assert repr(criteria_anpr) == expected_repr

        # test for drainage criteria
        expected_repr = (
            "<SuitabilityCriteria> 'drainage_class'(weight: 2, category: soilTerrain)\n"
            "Function:\n"
            "    SuitabilityFunction(func=discrete, params={'rules': {1: 0, 2: 0.1, 3: 0.5, 4:...\n"
            "Indicator:\n"
            "    Name        drainage \n"
            "    Data        float64 200B 3.0 3.0 3.0 3.0 3.0 3.0 ... 3.0 3.0 3.0 3.0 3.0 3.0\n"
            "    Dimensions  lat: 5, lon: 5 \n"
            "Attributes:\n"
            "    *empty*"
        )
        assert repr(criteria_drain) == expected_repr

    def test_attrs(self, criteria_anpr, annual_precip, sf_anpr):
        assert criteria_anpr.attrs == {}
        criteria_anpr.attrs = {
            "long_name": "Annual Precipitation",
            "description": "This is the annual precipitation criteria.",
            "comment": "Some comment about annual precipitation.",
        }
        assert criteria_anpr.attrs["long_name"] == "Annual Precipitation"
        assert criteria_anpr.attrs["description"] == "This is the annual precipitation criteria."
        assert criteria_anpr.attrs["comment"] == "Some comment about annual precipitation."

        # test when provided when creating the criteria
        sc = SuitabilityCriteria(
            name="annual_precipitation",
            category="climate",
            indicator=annual_precip,
            func=sf_anpr,
            long_name="Annual Precipitation",
            description="This is the annual precipitation criteria.",
            comment="Some comment about annual precipitation.",
            attrs={"another_attr": "value"},
        )
        assert sc.attrs["long_name"] == "Annual Precipitation"
        assert sc.attrs["description"] == "This is the annual precipitation criteria."
        assert sc.attrs["comment"] == "Some comment about annual precipitation."
        assert sc.attrs["another_attr"] == "value"

    def test_format(self, criteria_anpr):
        sc = criteria_anpr.compute()

        assert isinstance(sc, xr.DataArray)
        assert sc.name == "annual_precipitation"
        assert sc.dims == ("lat", "lon", "time")
        assert sc.shape == (5, 5, 5)
        np.testing.assert_equal(sc.lat.values, np.arange(5))
        np.testing.assert_equal(sc.lon.values, np.arange(5))
        np.testing.assert_equal(sc.time.values, pd.date_range("2000-01-01", periods=5, freq="YS"))

        # test output attrs
        for k in criteria_anpr.attrs:
            assert k in sc.attrs
            assert sc.attrs[k] == criteria_anpr.attrs[k]

    def test_compute_func(self, criteria_anpr, criteria_drain):
        # test suitability function computation
        sc = criteria_anpr.compute()
        np.testing.assert_array_almost_equal(sc.values, 0.25, decimal=2)

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

        # test when suitability function is not defined
        sc = SuitabilityCriteria(
            name="test",
            indicator=criteria_anpr.indicator,
        )
        with pytest.raises(
            ValueError, match="The suitability function is not defined. Please provide a valid function."
        ):
            sc.compute()


class TestGetIndicatorDescription:
    def test_with_attrs(self, annual_precip):
        desc = _get_indicator_description(annual_precip)
        assert "name: prcptot" in desc
        assert "units: mm" in desc
        assert "standard_name: lwe_thickness_of_precipitation_amount" in desc
        assert "long_name: Total accumulated precipitation" in desc

    def test_without_attrs(self, potential_rooting_depth):
        potential_rooting_depth.attrs = {}  # Clear attributes to simulate missing attrs
        desc = _get_indicator_description(potential_rooting_depth)
        assert desc == "name: potential_rooting_depth"
