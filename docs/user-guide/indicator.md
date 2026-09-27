---
file_format: mystnb
---

# Indicator: How to open your data?

In LSAPy, an [indicator](./terminology.rst#term-Indicator) refers to the input spatial gridded data used for downstream analysis and is expected to be a [`xarray.DataArray`] object. However, such data can be stored in various file formats and this guide describes how to open them in a ready-to-use format for LSAPy. A short example of how to convert vector data to raster data is also provided.

````{tip}
Overall, [xarray] should be able to open your data if it is stored in a common format for spatial gridded data (e.g., GeoTIFF, netCDF, GRIB, etc.). You can find a list of supported formats [here](https://docs.xarray.dev/en/stable/user-guide/io.html).

```python
import xarray as xr

ind = xr.open_dataarray("file.nc") # or "file.tif", "file.grib", etc.
```
````

## GeoTIFF

GeoTIFF is one of the most common formats for raster data, and you will likely encounter it when working with spatial data. To open a GeoTIFF file as [`xarray.DataArray`], you can use the [xarray] package using the [rasterio] backend engine.

```python
import xarray as xr

ind = xr.open_dataarray("file.tif")
```

```{note}
[rasterio] is a optional dependency of [xarray], so you may need to install it separately if you want to use this functionality.
```

Alternatively, you can also use the [rioxarray] package, which relies on [rasterio] to read the data and metadata, and which provides additional functionality such as reprojection and clipping.

```python
import rioxarray

ind = rioxarray.open_rasterio("file.tif")

# reproject to a different coordinate reference system (CRS)
ind = ind.rio.reproject("EPSG:4326")
```

## netCDF

Another common format for spatial gridded data, especially for climate data, is [netCDF](https://fr.wikipedia.org/wiki/NetCDF). Files in this format can easily be opened as [`xarray.DataArray`] using [xarray] directly. To read a netCFD file, you will first need to install one the the backend libraries (e.g., [netCDF4](https://github.com/Unidata/netcdf4-python), [h5netcdf](https://h5netcdf.org/)).

```python
import xarray as xr

ind = xr.open_dataarray("file.nc")
ind
```

If the variable is store in a netCDF file with multiple variables, you can use [`xarray.open_dataset`](https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html#xarray.open_dataset) to open the file as an [`xarray.Dataset`] and then select the variable you want to use.

```python
ds = xr.open_dataset("dataset.nc")
ind = ds["variable_name"]
```

## From vector data

Some data you want to use may be stored as vector data (e.g., Shapefile, GeoJSON, GeoPackage) and will need to be rasterized before it can be used with LSAPy. You can use the [geocube] package and the `make_geocube` function for this purpose.

```python
from geocube.api.core import make_geocube
import geopandas as gpd

layer = gpd.read_file("file.shp")  # or "file.gpkg", "file.geojson", etc.

ds = make_geocube(
    layer,
    measurements=["column_name"],  # attribute to rasterize (you can also use a list of attributes)
    output_crs="EPSG:4326",  # output coordinate reference system
    resolution=(-0.1, 0.1),  # output resolution (x, y)
)
ind = ds["column_name"]  # select the rasterized variable
```

```{note}
You can also use the `like` parameter to specify a reference a [`xarray.Dataset`] to define the output grid.
```

## Other formats

If your data are in another format not covered in this guide, we suggest you to have a look to this [guide](https://docs.xarray.dev/en/stable/user-guide/io.html) describing in more details how to open data from various formats using [xarray].

[geocube]: https://corteva.github.io/geocube/stable/index.html
[rasterio]: https://rasterio.readthedocs.io/en/stable/
[rioxarray]: https://corteva.github.io/rioxarray/stable/
[xarray]: https://docs.xarray.dev/en/stable/
[`xarray.dataarray`]: https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html
[`xarray.dataset`]: https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html
