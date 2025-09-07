"""Tests for statistics functions."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr

# from lsapy import LandSuitabilityAnalysis
from lsapy.stats import spatial_stats_summary, stats_summary


@pytest.fixture
def data():
    """Returns sample DataArray for testing."""
    return xr.Dataset(
        {
            "var1": (("time", "lat", "lon"), np.ones((5, 5, 5)) * 0.8),
            "var2": (("time", "lat", "lon"), np.ones((5, 5, 5)) * 0.6),
            "var3": (("time", "lat", "lon"), np.ones((5, 5, 5)) * 0.4),
        },
        coords={
            "lon": range(5),
            "lat": range(5),
            "time": pd.date_range("2000-01-01", periods=5, freq="YS"),
        },
    )


@pytest.fixture
def regions():
    """Returns sample regions GeoDataFrame for testing."""
    left_poly_coords = [(-0.5, -0.5), (1.5, -0.5), (1.5, 4.5), (-0.5, 4.5), (-0.5, -0.5)]
    right_poly_coords = [(1.5, -0.5), (4.5, -0.5), (4.5, 4.5), (1.5, 4.5), (1.5, -0.5)]
    _regions = gpd.GeoDataFrame(
        {"region": ["left", "right"]},
        geometry=gpd.GeoSeries.from_wkt(
            [
                f"POLYGON(({', '.join([f'{x} {y}' for x, y in left_poly_coords])}))",
                f"POLYGON(({', '.join([f'{x} {y}' for x, y in right_poly_coords])}))",
            ]
        ),
        crs="EPSG:4326",
    )
    return _regions


class TestStatsSummary:
    def test_outputs(self, data):
        stats = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        df = stats_summary(data)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (15, 10)  # 3 variables * 5 years
        np.testing.assert_array_equal(df.columns, ["variable", "time"] + stats)
        np.testing.assert_array_equal(df["variable"].unique(), ["var1", "var2", "var3"])
        np.testing.assert_array_equal(df["time"].unique(), data.time.values)
        np.testing.assert_array_equal(df["count"].values, 25)
        np.testing.assert_allclose(df.loc[df["variable"] == "var1"]["mean"].values, 0.8)
        np.testing.assert_allclose(df["std"].values, 0, atol=1e-6)
        np.testing.assert_allclose(df.loc[df["variable"] == "var1"]["min"].values, 0.8)
        np.testing.assert_allclose(df.loc[df["variable"] == "var2"]["25%"].values, 0.6)
        np.testing.assert_allclose(df.loc[df["variable"] == "var2"]["50%"].values, 0.6)
        np.testing.assert_allclose(df.loc[df["variable"] == "var3"]["75%"].values, 0.4)
        np.testing.assert_allclose(df.loc[df["variable"] == "var3"]["max"].values, 0.4)

    def test_vars_dims_selection(self, data):
        df = stats_summary(
            data,
            on_vars=["var1", "var3"],
            on_dims=["time"],
            on_dim_values={"time": slice("2000", "2002")},
        )
        assert df.shape == (6, 10)  # 2 variables * 3 years
        np.testing.assert_array_equal(df["variable"].unique(), ["var1", "var3"])
        np.testing.assert_array_equal(df["time"].unique(), data.time.values[:3])

    def test_bins(self, data):
        df = stats_summary(
            data,
            on_dim_values={"time": "2000"},
            bins=[0, 0.5, 0.75, 1],
            bins_labels=["low", "medium", "high"],
            include_lowest=True,
            all_bins=True,
        )
        assert all(c in df.columns for c in ["bin", "bin_label"])
        np.testing.assert_array_equal(df.loc[df["variable"] == "var1"]["bin_label"].values, ["high", "bins_range"])
        np.testing.assert_array_equal(df.loc[df["variable"] == "var1"]["bin"].values, ["(0.75, 1.0]", "[0.0, 1.0]"])
        np.testing.assert_array_equal(df.loc[df["variable"] == "var2"]["bin_label"].values, ["medium", "bins_range"])
        np.testing.assert_array_equal(df.loc[df["variable"] == "var2"]["bin"].values, ["(0.5, 0.75]", "[0.0, 1.0]"])
        np.testing.assert_array_equal(df.loc[df["variable"] == "var3"]["bin_label"].values, ["low", "bins_range"])
        np.testing.assert_array_equal(df.loc[df["variable"] == "var3"]["bin"].values, ["[0.0, 0.5]", "[0.0, 1.0]"])
        # test different bins and labels length
        with pytest.raises(ValueError):
            stats_summary(data, bins=[0, 0.5, 0.75, 1], bins_labels=["low", "medium"])

    def test_cell_area(self, data):
        df = stats_summary(
            data,
            on_dim_values={"time": "2000"},
            cell_area=(100, "km2"),
        )
        assert "area_km2" in df.columns
        np.testing.assert_array_equal(df["area_km2"].values, 2500)  # 25 cells * 100 km2

    def test_dropna(self, data):
        data_nan = data.copy()
        data_nan["var1"][2, :, :] = np.nan
        df = stats_summary(data_nan, on_vars=["var1", "var2"], dropna=True)
        assert df.shape == (9, 10)
        assert len(df.loc[df["variable"] == "var1"]) == 4
        assert len(df.loc[df["variable"] == "var2"]) == 5


class TestSpatialStatsSummary:
    def test_outputs(self, data, regions):
        # change some values for left region
        data["var1"][:, :, :2] = 0.7
        data["var2"][:, :, :2] = 0.5
        data["var3"][:, :, :2] = 0.3

        df = spatial_stats_summary(data, regions, name="region")
        assert "region" in df.columns
        np.testing.assert_array_equal(df["region"].unique(), ["Region0", "Region1"])
        # left region
        df_reg0 = df.loc[df["region"] == "Region0"]
        np.testing.assert_array_equal(df_reg0["count"].values, 10)
        np.testing.assert_allclose(df_reg0.loc[df_reg0["variable"] == "var1"]["mean"].values, 0.7)
        np.testing.assert_allclose(df_reg0.loc[df_reg0["variable"] == "var2"]["mean"].values, 0.5)
        np.testing.assert_allclose(df_reg0.loc[df_reg0["variable"] == "var3"]["mean"].values, 0.3)
        # right region
        df_reg1 = df.loc[df["region"] == "right"]
        np.testing.assert_array_equal(df_reg1["count"].values, 15)
        np.testing.assert_allclose(df_reg1.loc[df_reg1["variable"] == "var1"]["mean"].values, 0.8)
        np.testing.assert_allclose(df_reg1.loc[df_reg1["variable"] == "var2"]["mean"].values, 0.6)
        np.testing.assert_allclose(df_reg1.loc[df_reg1["variable"] == "var3"]["mean"].values, 0.4)

    def test_mask_kwargs(self, data, regions):
        df = spatial_stats_summary(data, regions, name="region", mask_kwargs={"names": "region"})
        np.testing.assert_array_equal(df["region"].unique(), ["left", "right"])
