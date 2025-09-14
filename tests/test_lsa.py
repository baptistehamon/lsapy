"""Tests for LandSuitabilityAnalysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from lsapy import LandSuitabilityAnalysis


@pytest.fixture
def lsa(criteria) -> LandSuitabilityAnalysis:
    """Returns a LandSuitabilityAnalysis instance for testing."""
    return LandSuitabilityAnalysis(
        land_use="land_use",
        short_name="test_land_use",
        long_name="Test Land Use",
        description="This is a test land use.",
        comment="An optional comment.",
        criteria=criteria,
        attrs={"another_attr": "value"},
    )


class TestLandSuitabilityAnalysis:
    def test_init(self, lsa, criteria):
        # test with no name
        with pytest.raises(
            TypeError, match=r"LandSuitabilityAnalysis.__init__\(\) missing 1 required positional argument: 'land_use'"
        ):
            LandSuitabilityAnalysis(criteria=criteria)
        # test with no criteria
        with pytest.raises(
            TypeError, match=r"LandSuitabilityAnalysis.__init__\(\) missing 1 required positional argument: 'criteria'"
        ):
            LandSuitabilityAnalysis(land_use="test_land_use")
        # test methods in init
        # test weights based criteria sorting
        assert list(lsa.criteria.keys()) == [
            "growing_degree_days",
            "potential_rooting_depth",
            "drainage_class",
            "annual_precipitation",
        ]
        # test params by category
        assert lsa.criteria_by_category == {
            "climate": ["growing_degree_days", "annual_precipitation"],
            "soilTerrain": ["potential_rooting_depth", "drainage_class"],
        }
        assert lsa.weights_by_category == {"climate": 4, "soilTerrain": 4}

    def test_attrs(self, lsa):
        # test when provided when creating the lsa
        assert lsa.attrs["long_name"] == "Test Land Use"
        assert lsa.attrs["description"] == "This is a test land use."
        assert lsa.attrs["comment"] == "An optional comment."
        assert lsa.attrs["another_attr"] == "value"
        lsa = LandSuitabilityAnalysis(land_use="land_use", criteria=lsa.criteria)
        assert lsa.attrs == {}
        lsa.attrs = {
            "long_name": "Test Land Use",
            "description": "This is a test land use.",
            "comment": "An optional comment.",
        }
        assert lsa.attrs["long_name"] == "Test Land Use"
        assert lsa.attrs["description"] == "This is a test land use."
        assert lsa.attrs["comment"] == "An optional comment."

    def test_repr(self, lsa):
        expected_repr = (
            "<LandSuitabilityAnalysis> 'land_use'\n"
            "Criteria:\n"
            "    growing_degree_days      (w=3.0) climate vetharaniam2022_eq5(a=-0.55, b=1...\n"
            "    potential_rooting_depth  (w=2.0) soilTerrain vetharaniam2022_eq5(a=-9.8, ...\n"
            "    drainage_class           (w=2.0) soilTerrain discrete(rules={1: 0, 2: 0.1...\n"
            "    annual_precipitation     (w=1.0) climate vetharaniam2022_eq5(a=-0.71, b=1...\n"
            "Attributes:\n"
            "    short_name:               test_land_use\n"
            "    long_name:                Test Land Use\n"
            "    description:              This is a test land use.\n"
            "    comment:                  An optional comment.\n"
            "    another_attr:             value"
        )
        assert repr(lsa) == expected_repr

    def test_run_invalid_params(self, lsa):
        # test invalid suitability_type
        with pytest.raises(
            ValueError, match="'suitability_type' must be one of 'criteria', 'category', or 'overall'. Got 'invalid'."
        ):
            lsa.run("invalid")
        # test no aggregation method for category
        with pytest.raises(ValueError, match="No aggregation method provided for 'category'."):
            lsa.run("category", agg_methods={"overall": "mean"})
        # test no aggregation method for overall
        with pytest.raises(ValueError, match="No aggregation method provided for 'overall'."):
            lsa.run("overall", agg_methods={"category": "mean"})
        # test wrong agg_methods type
        with pytest.raises(TypeError, match="'agg_methods' must be a string or a dictionary. Got <class 'list'>."):
            lsa.run("overall", agg_methods=["mean", "gmean"])

    def test_run_criteria(self, lsa):
        res = lsa.run("criteria")
        # test format, shape and attrs
        assert isinstance(res, xr.Dataset)
        assert dict(res.sizes) == {"lat": 5, "lon": 5, "time": 5}
        np.testing.assert_equal(res.lat.values, np.arange(5))
        np.testing.assert_equal(res.lon.values, np.arange(5))
        np.testing.assert_equal(res.time.values, pd.date_range("2000-01-01", periods=5, freq="YS"))
        assert all([c in res.data_vars for c in lsa._criteria_list])
        assert res.attrs["criteria"] == lsa._criteria_list
        assert res.attrs["land_use"] == lsa.land_use
        assert res.attrs["short_name"] == "test_land_use"
        assert res.attrs["long_name"] == "Test Land Use"
        assert res.attrs["description"] == "This is a test land use."
        # test values
        np.testing.assert_array_almost_equal(res.growing_degree_days.values, 0.75, decimal=2)
        np.testing.assert_array_almost_equal(res.potential_rooting_depth.values, 0.95, decimal=2)
        np.testing.assert_array_almost_equal(res.drainage_class.values, 0.5, decimal=2)
        np.testing.assert_array_almost_equal(res.annual_precipitation.values, 0.25, decimal=2)

    def test_agg_kwargs_formatting(self, lsa):
        res = lsa._format_agg_kwargs(
            agg_methods={
                "climate": "wgmean",
                "soilTerrain": "wgmean",
                "suitability": "wmean",
            },
            agg_on={
                "climate": ["growing_degree_days", "annual_precipitation"],
                "soilTerrain": ["potential_rooting_depth", "drainage_class"],
                "suitability": ["potential_rooting_depth", "drainage_class", "climate"],
            },
        )
        assert res == {
            "climate": {"weights": [3, 1]},
            "soilTerrain": {"weights": [2, 2]},
            "suitability": {"weights": [2, 2, 4]},
        }

    def test_run_category(self, lsa):
        res = lsa.run("category", agg_methods="wgmean")
        # test format, shape and attrs
        assert isinstance(res, xr.Dataset)
        assert dict(res.sizes) == {"lat": 5, "lon": 5, "time": 5}
        np.testing.assert_equal(res.lat.values, np.arange(5))
        np.testing.assert_equal(res.lon.values, np.arange(5))
        np.testing.assert_equal(res.time.values, pd.date_range("2000-01-01", periods=5, freq="YS"))
        assert all([c in res.data_vars for c in lsa.category])
        assert res.attrs["land_use"] == lsa.land_use
        assert res.attrs["criteria"] == lsa._criteria_list
        assert res.attrs["short_name"] == "test_land_use"
        assert res.attrs["long_name"] == "Test Land Use"
        assert res.attrs["description"] == "This is a test land use."
        # test values
        np.testing.assert_array_almost_equal(res.climate.values, 0.57, decimal=2)
        np.testing.assert_array_almost_equal(res.soilTerrain.values, 0.69, decimal=2)
        # test if no category in criteria, should return criteria
        lsa.category = [None]  # force no category
        res = lsa.run("category", agg_methods="wgmean")
        assert res.equals(lsa.run("criteria"))

    def test_run_overall(self, lsa):
        res = lsa.run("overall", agg_methods="wgmean")
        # test format, shape and attrs
        assert isinstance(res, xr.Dataset)
        assert dict(res.sizes) == {"lat": 5, "lon": 5, "time": 5}
        np.testing.assert_equal(res.lat.values, np.arange(5))
        np.testing.assert_equal(res.lon.values, np.arange(5))
        np.testing.assert_equal(res.time.values, pd.date_range("2000-01-01", periods=5, freq="YS"))
        assert "suitability" in res.data_vars
        assert res.attrs["land_use"] == lsa.land_use
        assert res.attrs["criteria"] == lsa._criteria_list
        assert res.attrs["short_name"] == "test_land_use"
        assert res.attrs["long_name"] == "Test Land Use"
        assert res.attrs["description"] == "This is a test land use."
        # test values
        np.testing.assert_array_almost_equal(res.suitability.values, 0.63, decimal=2)
        # test with by_category=False, should return aggregation of criteria
        res = lsa.run("overall", agg_methods="wmean", by_category=False)
        np.testing.assert_array_almost_equal(res.suitability.values, 0.68, decimal=2)
        # test with by_category=[True, None] but without category in criteria
        lsa.category = [None]  # force no category
        res = lsa.run("overall", agg_methods="wmean", by_category=True)
        np.testing.assert_array_almost_equal(res.suitability.values, 0.68, decimal=2)
        res = lsa.run("overall", agg_methods="wmean")
        np.testing.assert_array_almost_equal(res.suitability.values, 0.68, decimal=2)

    def test_run_keep_vars(self, lsa):
        # test with keep_vars=False, default=True
        res = lsa.run("category", keep_vars=False)
        assert list(res.data_vars) == ["climate", "soilTerrain"]
        # test for overall
        res = lsa.run("overall", keep_vars=False)
        assert list(res.data_vars) == ["suitability"]

    def test_run_inplace(self, lsa):
        # test default inplace=False
        res = lsa.run("criteria")
        assert isinstance(res, xr.Dataset)
        # test inplace=True
        res = lsa.run("criteria", inplace=True)
        assert res is None
        assert lsa.data.equals(lsa.run("criteria", inplace=False))
        # test inplace if existing data
        lsa.data = xr.Dataset()
        res = lsa.run("criteria", inplace=True)
        assert res is None

    def test_aggregate(self, lsa):
        lsa.run("criteria", inplace=True)
        # test limiting factor
        res = lsa._aggregate(
            lsa.data, agg_on={"climate": ["growing_degree_days", "annual_precipitation"]}, methods="limfactor"
        )
        assert isinstance(res, xr.Dataset)
        # test wrong methods type
        with pytest.raises(TypeError, match="'methods' must be a string or a dictionary of strings."):
            lsa._aggregate(lsa.data, agg_on={"climate": ["growing_degree_days", "annual_precipitation"]}, methods=1)
