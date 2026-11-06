---
file_format: mystnb
kernelspec:
  name: python3
---

```{eval-rst}
.. currentmodule:: lsapy
```

# LSAPy Overview

LSAPy helps you turn gridded indicators into land suitability maps. It is built on [xarray], so indicators keep their dimensions, coordinates, and metadata throughout the analysis. This page gives a quick tour of the workflow. For complete explanations and
additional examples, follow the links at each step.

For the purpose of the following examples, we first create a small synthetic dataset of two indicators: slope and annual mean temperature.

```{code-cell} ipython3
import numpy as np
import xarray as xr

ds = xr.Dataset(
    {
        "slope": (("y", "x"), [[2, 8, 10], [12, 15, 20]]),
        "temperature": (("y", "x"), [[10, 12, 14], [16, 18, 20]]),
    },
    coords={"y": [45, 46], "x": [2, 3, 4]},
)
ds
```

## Standardize your data

Standardization converts raw indicator values into comparable scores, typically between 0 and 1. `lsapy.standardize` provides a variety of standardization functions, including threshold, categorical, sigmoid, and Gaussian-like functions. For example, input values can be standardized to boolean scores comparing them against a threshold and a specified operator using {py:meth}`lsapy.standardize.boolean`:

```{code-cell} ipython3
import numpy as np
import lsapy.standardize as lstd

lstd.boolean(ds["slope"], op="<", thresh=10)
```

Or standardized to continuous scores using, for example, {py:meth}`lsapy.standardize.logistic`:

```{code-cell} ipython3
lstd.logistic(ds["temperature"], a=1, b=15)
```

Discover more about standardization functions in the [Standardization](../user-guide/standardization.md) guide.

## Defining a SuitabilityCriteria

A {py:class}`SuitabilityCriteria` associates an indicator with a standardization function and its parameters for subsequent use in a land suitability analysis. This keeps the information together, making the analysis reproducible and preserving useful metadata:

```{code-cell} ipython3
from lsapy import SuitabilityCriteria

sc = {
    "slope": SuitabilityCriteria(
        name="slope",
        indicator=ds["slope"],
        func=lstd.boolean,
        fparams={"op": "<", "thresh": 10},
    ),
    "temperature": SuitabilityCriteria(
        name="temperature",
        indicator=ds["temperature"],
        func=lstd.logistic,
        fparams={"a": 1, "b": 15},
    ),
}
```

And the specified function can be applied to the indicator to compute the scores using {py:meth}`SuitabilityCriteria.compute`:

```{code-cell} ipython3
sc["slope"].compute()
```

Learn more about {py:class}`SuitabilityCriteria` use in the [Defining a SuitabilityCriteria](../user-guide/suitability-criteria.md) guide.

## Performing a Land Suitability Analysis

{py:class}`LandSuitabilityAnalysis` is used to combine multiple criteria into an overall suitability score using {py:meth}`LandSuitabilityAnalysis.run`:

```{code-cell} ipython3
from lsapy import LandSuitabilityAnalysis

lsa = LandSuitabilityAnalysis(
    land_use="example_crop",
    criteria=sc,
)

res = lsa.run()
res
```

See [Performing a Land Suitability Analysis](../user-guide/lsa.md) for more details on the analysis workflow, including aggregation methods, category-level results, and output options, to customize the analysis to your needs.

## Statistics summary

The `lsapy.stats` module provides functions to compute descriptive statistics of a {py:class}`xarray.Dataset`, which is useful for summarizing the results of a land suitability analysis:

```{code-cell} ipython3
from lsapy.stats import stats_summary

df = stats_summary(res)
df
```

Find out more about statistics, including spatial statistics, in the [Statistics](../user-guide/stats.md) guide.

[xarray]: https://docs.xarray.dev/en/stable/
