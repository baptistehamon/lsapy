---
file_format: mystnb
---

# Indicator: How to open your data?

In LSAPy, an [indicator](./terminology.rst#term-Indicator) is the input spatial gridded data used in downstream analysis. It is expected to be a {py:class}`xarray.DataArray` object, but the source data are often stored in various file format.

This guide explains how to open common spatial data formats in a ready-to-use form for LSAPy. It also includes a short example showing how to convert vector data into raster data before analysis.

````{tip}
In most cases, [xarray] can open common gridded spatial formats directly (e.g., GeoTIFF, netCDF, GRIB, etc.). A full list of supported formats is available in the [xarray I/O documentation](https://docs.xarray.dev/en/stable/user-guide/io.html).

```python
import xarray as xr

ind = xr.open_dataarray("file.nc")  # or "file.tif", "file.grib", etc.
```
````

## GeoTIFF

GeoTIFF is one of the most common formats for raster data and is frequently encountered in geospatial workflows. To open a GeoTIFF file as an {py:class}`xarray.DataArray`, you can use [xarray] with the [rasterio] backend.

```python
import xarray as xr

ind = xr.open_dataarray("file.tif", engine="rasterio")
```

```{note}
[rasterio] is an optional dependency of [xarray], so you may need to install it separately if this functionality is not already available in your environment.
```

Alternatively, you can use the [rioxarray] package, which relies on [rasterio] to read both the data and geospatial metadata. It also adds useful functionality such as reprojection and clipping.

```python
import rioxarray

ind = rioxarray.open_rasterio("file.tif")

# Reproject to a different coordinate reference system (CRS)
ind = ind.rio.reproject("EPSG:4326")
```

## netCDF

Another common format for spatial gridded data, especially for climate data, is [netCDF](https://fr.wikipedia.org/wiki/NetCDF). Files in this format can easily be opened as {py:class}`xarray.DataArray` using [xarray] directly. You may need to install a backend such as [netCDF4](https://github.com/Unidata/netcdf4-python) or [h5netcdf](https://h5netcdf.org/).

```python
import xarray as xr

ind = xr.open_dataarray("file.nc")
ind
```

If a netCDF file contains multiple variables, you can use {py:func}`xarray.open_dataset` to open the file as an {py:class}`xarray.Dataset` and then select the variable you want to use.

```python
import xarray as xr

ds = xr.open_dataset("dataset.nc")
ind = ds["variable_name"]
```

## From vector data

Some datasets are stored as vector data (for example, Shapefile, GeoJSON, or GeoPackage) and must be rasterized before they can be used with LSAPy. In these cases, the [geocube] package and its `make_geocube` function are useful.

```python
from geocube.api.core import make_geocube
import geopandas as gpd

layer = gpd.read_file("file.shp")  # or "file.gpkg", "file.geojson", etc.

ds = make_geocube(
    layer,
    measurements=["column_name"],  # attribute to rasterize (you can also supply a list of attributes)
    output_crs="EPSG:4326",  # output coordinate reference system
    resolution=(-0.1, 0.1),  # output resolution (x, y)
)
ind = ds["column_name"]  # select the rasterized variable
```

```{note}
You can also use the `like` parameter to specify a reference a {py:class}`xarray.Dataset` to define the output grid from that object.
```

## Other formats

If your data are in another format not covered in this guide, we suggest you to consult the [xarray IO guide](https://docs.xarray.dev/en/stable/user-guide/io.html), which describes how to open data from a wide range of formats.

[geocube]: https://corteva.github.io/geocube/stable/index.html
[rasterio]: https://rasterio.readthedocs.io/en/stable/
[rioxarray]: https://corteva.github.io/rioxarray/stable/
[xarray]: https://docs.xarray.dev/en/stable/
