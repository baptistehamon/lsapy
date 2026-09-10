Chunked indicators
==================

Install ``lsapy[dask]`` to use Dask-backed indicators. Standardization is applied
independently to each chunk by default, and the analysis result remains lazy until
you explicitly compute it. This example uses synthetic data and illustrative
parameters, not calibrated crop parameters.

.. code-block:: python

    import numpy as np
    import xarray as xr

    from lsapy import LandSuitabilityAnalysis, SuitabilityCriteria

    temperature = xr.DataArray(
        np.arange(24, dtype=float).reshape(4, 6),
        dims=("y", "x"),
        coords={"y": range(4), "x": range(6)},
        name="temperature",
    ).chunk({"y": 2, "x": 3})
    sc = SuitabilityCriteria(
        name="temperature",
        indicator=temperature,
        func="logistic",
        fparams={"a": 1.0, "b": 12.0},
    )
    lsa = LandSuitabilityAnalysis(land_use="example_crop", criteria={"temperature": sc})
    result = lsa.run()
    assert result.suitability.chunks is not None
    computed = result.compute()
    print(computed.suitability)

Execution options
-----------------

``SuitabilityCriteria.compute`` defaults to ``dask="parallelized"`` only when the
indicator is chunked. ``LandSuitabilityAnalysis.run`` forwards its keyword arguments
to each criterion. Explicit options are respected:

* ``lsa.run(dask="forbidden")`` rejects chunked inputs.
* ``lsa.run(dask="allowed")`` passes Dask arrays directly to the function; use this
  only with functions that support Dask natively.
* ``lsa.run(output_dtypes=[float])`` supplies the output dtype and avoids Dask's
  sample call for dtype inference. Without this option, Dask infers the dtype from
  a small synthetic sample, without computing the full indicator. Custom functions
  that cannot accept that sample should supply dtype information explicitly.
* ``sc.compute(dask_gufunc_kwargs={"meta": np.array([], dtype=float)})``
  is an alternative to ``output_dtypes``; do not supply both.

For custom functions that reduce core dimensions or change output shape, supply
the appropriate ``input_core_dims`` and ``output_core_dims`` and choose compatible
chunks as described in the `xarray apply_ufunc documentation`_. Default chunkwise
execution assumes an elementwise standardization function. Choose chunks that fit
in memory; chunking alone does not guarantee a speed improvement.

.. _xarray apply_ufunc documentation: https://docs.xarray.dev/en/stable/generated/xarray.apply_ufunc.html
