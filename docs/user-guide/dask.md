---
file_format: mystnb
---

```{eval-rst}
.. currentmodule:: lsapy
```

# Parallel Computing with Dask

By relying on [xarray], LSAPy computations can be parallelized with [Dask] without changing your workflow.
When an indicator is backed by Dask (i.e., it is chunked), {py:meth}`SuitabilityCriteria.compute` and
{py:meth}`LandSuitabilityAnalysis.run` build a lazy task graph. The calculation is not started until
the computation is triggered (e.g., by calling `compute()` or writing the results to disk).

This is useful when indicators are larger than memory, when several indicators
must be processed, or when work should be distributed across multiple cores or
machines.

This guide only briefly covers Dask usage. For more information, see the
[Dask documentation](https://docs.dask.org/en/stable/) and the xarray
[parallel computing guide](https://docs.xarray.dev/en/stable/user-guide/dask.html).

## Installation

To use [Dask] with LSAPy, you will first need to install Dask. You can do it using the
`parallel` extra when installing LSAPy:

```console
python -m pip install "lsapy[parallel]"
```

Or install Dask separately:

```console
python -m pip install dask
# or
conda install dask
```

## Chunking

An xarray object becomes Dask-backed when it is chunked. You can create
chunked data using {py:meth}`xarray.DataArray.chunk` or {py:meth}`xarray.Dataset.chunk`. For example:

```{code-cell} ipython3
import numpy as np
import xarray as xr

temp = xr.DataArray(
    np.random.normal(15, 5, size=24).reshape(4, 6),
    dims=("y", "x"),
    coords={"y": range(4), "x": range(6)},
    name="temperature",
).chunk({"y": 2, "x": 3})

temp.chunks
```

For data read from disk, prefer opening it with chunks rather than loading the
whole file first. The exact chunks supported depend on the backend:

```python
temperature = xr.open_dataarray(
    "temperature.nc",
    chunks={"time": 365, "y": 24, "x": 56},
)
```

```{note}
Chunking sizes is important and bad chunking choice can lead to poor performance or memory errors. Moreover, chunking alone does not guarantee a speed improvement. See the Dask [best practices guide](https://docs.dask.org/en/stable/best-practices.html) for more information.
```

## Dask in LSAPy

LSAPy automatically detects chunked data and defaults to using [Dask] for parallelization. Therefore, the workflow remains the same when using Dask. However, there are some key concepts and characteristics you need to understand or be aware of when using Dask with LSAPy.

### SuitabilityCriteria

When a chunked indicator is passed to a {py:class}`SuitabilityCriteria`, it is important to differentiate between the `compute()` method of the `SuitabilityCriteria` and the `compute()` method of the underlying Dask array. Calling {py:meth}`SuitabilityCriteria.compute` will build a lazy task graph and return a lazy {py:class}`xarray.DataArray`. In contrast, calling `.compute()` on the underlying Dask array will trigger the computation immediately.

```{code-cell} ipython3
from lsapy import SuitabilityCriteria

sc = SuitabilityCriteria(
    name="temperature",
    indicator=temp,
    func="logistic",
    fparams={"a": 1.0, "b": 12.0},
)

res = sc.compute()
res
```

The computation is not executed until you call `.compute()` on the result:

```{code-cell} ipython3
computed = res.compute()
computed
```

### LandSuitabilityAnalysis

Likewise, the workflow for {py:class}`LandSuitabilityAnalysis` is unchanged. When chunked indicators are used, the result of `LandSuitabilityAnalysis.run()` is a lazy {py:class}`xarray.Dataset`. The computation is not executed until you call `.compute()` on the result.

```{code-cell} ipython3
from lsapy import LandSuitabilityAnalysis

# create a precipitation indicator
precip = xr.DataArray(
    np.random.normal(1000, 200, size=24).reshape(4, 6),
    dims=("y", "x"),
    coords={"y": range(4), "x": range(6)},
    name="precipitation",
).chunk({"y": 2, "x": 3})

sc = {
    "temperature": SuitabilityCriteria(
        name="temperature",
        indicator=temp,
        func="logistic",
        fparams={"a": 1.0, "b": 12.0},
    ),
    "precipitation": SuitabilityCriteria(
        name="precipitation",
        indicator=precip,
        func="logistic",
        fparams={"a": 0.03, "b": 1000.0},
    ),
}

lsa = LandSuitabilityAnalysis(
    land_use="example_crop",
    criteria=sc,
)

res = lsa.run() # lazy xarray.Dataset
res = res.compute() # trigger computation
```

If you write the results of the LSA to disk, it not necessary to explicitly trigger computation, as Dask will automatically compute the results when writing to disk. For example:

```python
res = lsa.run()
res.to_netcdf("filename.nc")  # triggers computation and writes to disk
```

### Dask options in LSAPy

LSAPy automatically detects chunked data and defaults to using [Dask] for parallelization. This default behavior can be changed or customized by passing `kwargs` accepted by {py:func}`xarray.apply_ufunc` to {py:meth}`SuitabilityCriteria.compute` or {py:meth}`LandSuitabilityAnalysis.run`. For example, the default behavior can be overridden by passing the `dask` argument:

- `dask="parallelized"` (default) applies the standardization function to each chunk in parallel.
- `dask="forbidden"` raises an error if the indicator is chunked.
- `dask="allowed"` passes the Dask array directly to the standardization function. Use this only when the function supports Dask arrays natively.

Others accepted dask options can be found in {py:func}`xarray.apply_ufunc`.

[dask]: https://www.dask.org/
[xarray]: https://docs.xarray.dev/en/stable/
