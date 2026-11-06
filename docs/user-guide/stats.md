---
file_format: mystnb
---

```{eval-rst}
.. currentmodule:: lsapy
```

(stats)=

# Statistics

`lsapy.stats` provides two functions to statistically describe data
in an {py:class}`xarray.Dataset`:

- {py:func}`lsapy.stats.stats_summary` describes values over the spatial
  dimensions of a dataset.
- {py:func}`lsapy.stats.spatial_stats_summary` applies the same summary
  separately to polygons in a {py:class}`geopandas.GeoDataFrame`.

Both functions return a {py:class}`pandas.DataFrame` with one row per variable and
dimension combination, and the following descriptive statistics as columns: `count`,
`mean`, `std`, `min`, the 25th, 50th, and 75th percentiles, and `max`.

## Example dataset

The statistics functions operate on an {py:class}`xarray.Dataset`. In a typical LSA
workflow, this is the dataset returned by {py:meth}`lsapy.LandSuitabilityAnalysis.run`:

```python
result = lsa.run(suitability_type="overall")
stats = stats_summary(result)
```

The following small dataset is used in the examples
below.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import xarray as xr

data = xr.Dataset(
    {
        "suitability": (
            ("time", "lat", "lon"),
            np.array(
                [
                    [[0.0, 0.2, 0.4, 0.6], [0.4, 0.6, 0.8, 1.0]],
                    [[0.1, 0.2, 0.3, 0.4], [0.7, 0.8, 0.9, 1.0]],
                ]
            ),
        ),
        "temperature": (
            ("time", "lat", "lon"),
            np.array(
                [
                    [[10.0, 10.5, 11.0, 11.5], [12.0, 12.5, 13.0, 13.5]],
                    [[10.1, 10.2, 10.3, 10.4], [12.7, 12.8, 12.9, 13.0]]
                ]
            ),
        ),
    },
    coords={
        "time": pd.date_range("2000-01-01", periods=2, freq="YS"),
        "lat": [45.0, 45.5],
        "lon": [170.0, 170.5, 171.0, 171.5],
    },
)
data
```

## Statistical summary

Call {py:func}`lsapy.stats.stats_summary` with the dataset:

```{code-cell} ipython3
from lsapy.stats import stats_summary

df_stats = stats_summary(data)
df_stats
```

By default, all variables of the `Dataset` are summarized over spatial dimensions
(i.e., `lat` and `lon`, or `x` and `y`). In this example, each row represents one
variable and one time value, while the height grid cells contribute to the statistics.

### Selecting variables and dimensions

You can use the `on_vars` argument to select the variables to summarize.

```{code-cell} ipython3
df_stats = stats_summary( data, on_vars=["suitability"])
df_stats
```

Use `on_dims` to control which dimensions are retained in the output. In the
following example, the statistics are calculated only along the `lon` dimension.

```{code-cell} ipython3
df_stats = stats_summary(data, on_dims=["time", "lat"])
df_stats
```

You can use `on_dim_values` to compute the statistics only on a subset of data.
Values are passed to {py:meth}`xarray.Dataset.sel`, so labels, slices, and other
valid indexers can be used:

```{code-cell} ipython3
df_stats = stats_summary(data, on_dim_values={"time": "2000"})
df_stats
```

## Grouping values into bins

In some cases (i.e., suitability), it can be useful to group values
into intervals before calculating statistics. The function uses
{py:func}`pandas.cut` to create bins and you can specify the interval
and labels using the `bins` and `bins_labels` arguments. Additional
keyword arguments are passed to {py:func}`pandas.cut`.

For suitability scores, bins can represent meaningful suitability classes:

```{code-cell} ipython3
df_stats = stats_summary(
    data,
    on_vars=["suitability"],
    bins=[0, 0.25, 0.5, 0.75, 1],
    bins_labels=[
        "unsuitable",
        "poorly suitable",
        "moderately suitable",
        "highly suitable",
    ],
    include_lowest=True,
)
df_stats
```

Set `all_bins=True` to add one extra row representing the complete range from
the first to the last bin edge:

```{code-cell} ipython3
df_stats = stats_summary(
    data,
    on_vars=["suitability"],
    bins=[0, 0.25, 0.5, 0.75, 1],
    bins_labels=[
        "unsuitable",
        "poorly suitable",
        "moderately suitable",
        "highly suitable"
    ],
    all_bins=True,
    include_lowest=True,
)
df_stats
```

## Estimating area from cell counts

When using `bins`, it might be useful to estimate the area falling into each bin.
You can pass the cell area and unit through the `cell_area` argument and the function
will estimate the area based on the `count` of non-null cells in each bin and adds the
`area_<unit>` column to the output:

```{code-cell} ipython3
df_stats = stats_summary(
    data,
    on_vars=["suitability"],
    bins=[0, 0.5, 1],
    bins_labels=["lower", "higher"],
    include_lowest=True,
    cell_area=(5, "ha"),
)
df_stats[["variable", "time", "bin_label", "count", "area_ha"]]
```

```{caution}
The calculated area is an approximation assuming that all cells have the same area, and we don't
recommend using this if an accurate area calculation is required.
```

## Missing values

By default, rows containing missing values are retained in the returned
table, and the descriptive statistics follow pandas' missing-value behavior.
Set `dropna=True` to remove rows with missing values from the final DataFrame:

```python
stats_summary(data, dropna=True)
```

Remember that dropping rows can remove an entire variable and
dimension combination if all its values are missing.

## Statistics by geographic areas

Use {py:func}`lsapy.stats.spatial_stats_summary` when statistics are needed
for polygons such as administrative boundaries, catchments, or management
units. The input polygons must be provided as a
{py:class}`geopandas.GeoDataFrame` and should use a coordinate reference
system compatible with the xarray coordinates. The functions uses
[regionmask] determine which cells fall
within each polygon.

```{code-cell} ipython3
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

regions = gpd.GeoDataFrame(
    {"name": ["north", "south"]},
    geometry=[box(169.75, 45.25, 171.75, 45.75), box(169.75, 44.75, 171.75, 45.25)],
    crs="EPSG:4326",
)

fig, ax = plt.subplots()
data["suitability"].isel(time=0).plot(ax=ax)
regions.boundary.plot(ax=ax, color="red")
ax.set_xlim(169.725, 171.775)
ax.set_ylim(44.725, 45.775)
plt.show()
```

Calculate statistics for each polygon:

```{code-cell} ipython3
from lsapy.stats import spatial_stats_summary

df_stats = spatial_stats_summary(
    data,
    regions,
    name="region",
    on_vars=["suitability"],
)
df_stats
```

The `name` argument specifies the output column name for the polygon labels.
By default, `regionmask` assigns generated names such as `Region0` and `Region1`.
To use a column from the GeoDataFrame, pass its name through `mask_kwargs`:

```{code-cell} ipython3
df_stats = spatial_stats_summary(
    data,
    regions,
    name="area",
    mask_kwargs={"names": "name"},
)
df_stats
```

Here, `area` is the output column name and `name` is the source column
containing polygon labels. All additional keyword arguments are forwarded to
{py:func}`lsapy.stats.stats_summary`, so filtering, bins, cell area, and
missing-value handling work the same way.

```{code-cell} ipython3
df_stats = spatial_stats_summary(
    data,
    regions,
    name="area",
    mask_kwargs={"names": "name"},
    on_vars=["suitability"],
    bins=[0, 0.5, 1],
    bins_labels=["lower", "higher"],
    include_lowest=True,
    cell_area=(5, "ha"),
)
df_stats
```

[regionmask]: https://regionmask.readthedocs.io/
