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
        criteria_anpr.long_name = "Annual Precipitation"
        criteria_anpr.description = "This is the annual precipitation criteria."
        anpr_repr = repr(criteria_anpr)

        assert "SuitabilityCriteria(" in anpr_repr
        assert "name='annual_precipitation'" in anpr_repr
        assert "indicator=prcptot" in anpr_repr
        assert f"func={repr(criteria_anpr.func)}" in anpr_repr
        assert "weight=1" in anpr_repr
        assert "category='climate'" in anpr_repr
        assert "long_name='Annual Precipitation'" in anpr_repr
        assert "description='This is the annual precipitation criteria.'" in anpr_repr

        # test for drainage criteria
        criteria_drain.comment = "Some comment about drainage."
        criteria_drain.is_computed = True
        drain_repr = repr(criteria_drain)

        assert "SuitabilityCriteria(" in drain_repr
        assert "name='drainage_class'" in drain_repr
        assert "indicator=drainage" in drain_repr
        assert f"func={repr(criteria_drain.func)}" in drain_repr
        assert "weight=2" in drain_repr
        assert "category='soilTerrain'" in drain_repr
        assert "comment='Some comment about drainage.'" in drain_repr
        assert "is_computed=True" in drain_repr

    def test_attrs(self, criteria_anpr, sf_anpr):
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

    def test_format(self, criteria_anpr):
        sc = criteria_anpr.compute()

        assert isinstance(sc, xr.DataArray)
        assert sc.name == "annual_precipitation"
        assert sc.dims == ("lat", "lon", "time")
        assert sc.shape == (5, 5, 5)
        np.testing.assert_equal(sc.lat.values, np.arange(5))
        np.testing.assert_equal(sc.lon.values, np.arange(5))
        np.testing.assert_equal(sc.time.values, pd.date_range("2000-01-01", periods=5, freq="YS"))

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
