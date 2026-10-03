---
file_format: mystnb
---

```{eval-rst}
.. currentmodule:: lsapy
```

(standardization)=

# Standardizing your data

In LSAPy, a standardization function converts raw indicator values into scores that can be compared and combined across criteria. The goal is usually to map environmental or spatial variables to a common scale, often between 0 and 1, before they are used in a land suitability analysis.

This step is essential because indicators often have different units and ranges. For example, slope, rainfall, temperature, and distance to roads are not directly comparable, but they can all be standardized to a suitability scale before aggregation.

A standardized criterion is then typically used in a [`SuitabilityCriteria`](./suitability-criteria.md) object and later combined in a [`LandSuitabilityAnalysis`](./lsa.md).

(std.why)=

## Why standardization matters

Raw values can be noisy, skewed, or measured in different units. Standardization makes those values interpretable as suitability scores:

- 0 means not suitable at all
- 1 means fully suitable
- values in between represent partial suitability

The exact meaning depends on the function used. Some functions enforce hard thresholds, while others create gradual transitions around a midpoint or optimum.

(std.families)=

## Families of standardization functions

LSAPy provide a range of standardization functions from simple one to more complex functions found in the literature. This section does not
aim to provide an example for each function included in LSAPy, but rather to give an overview of the different families of functions and their intended use cases. You can find the full list of available functions in the [API reference](../api/standardize.rst).

(std.boolean)=

### Boolean function

{py:func}`standardize.boolean` is useful when a criterion has a strict threshold. Examples include conditions such as “slope must be lower than 10 degrees” or “soil depth must be greater than 50 cm”.

```{code-cell} ipython3
import numpy as np
import lsapy.standardize as lstd

x = np.array([2, 8, 10, 12])
lstd.boolean(x, op="<=", thresh=10)
```

This returns a boolean mask, where values passing the rule are `True` and others are `False`.

By default (`skipna=True`), `NaN` values are preserved in the output. If you want to treat `NaN` as failing the threshold, set `skipna=False`.

```{code-cell} ipython3
x = np.array([1, 2, np.nan, 4, 5])
lstd.boolean(x, op=">", thresh=3, skipna=False)
```

(std.discrete)=

### Discrete function

{py:func}`standardize.discrete` is appropriate for ordinal or categorical indicators, where a discrete set of known classes or levels should be mapped to a score, such as soil type or land-cover class.

```{code-cell} ipython3
rules = {
    0: 0.0,
    1: 0.25,
    2: 0.5,
    3: 0.75,
    4: 1.0,
}

lstd.discrete([0, 2, 4], rules)
```

(std.sigmoid)=

### Sigmoid-like functions

Sigmoid-like functions create a transition from 0 to 1 around a midpoint. Depending of the function and parameters, the transition can be sharp or gradual, reversed (from 1 to 0), or asymmetric.

A simple example is the {py:func}`standardize.logistic` function, with the parameters `a` (steepness) and `b` (midpoint):

```{code-cell} ipython3
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)
y = lstd.logistic(x, a=1, b=5.0)
ybis = lstd.logistic(x, a=2, b=5.0)

plt.plot(x, y, label="a=1")
plt.plot(x, ybis, label="a=2")
plt.legend()
plt.show()
```

A steeper curve (higher `a`) means that more abruptly changes occur around the midpoint, while a lower `a` creates a more gradual transition.

A special case of the logistic function is the basic sigmoid ({py:func}`standardize.sigmoid`), which is centered on zero and has a fixed steepness:

```{code-cell} ipython3
x = np.linspace(-5, 5, 100)
y = lstd.sigmoid(x)

plt.plot(x, y)
plt.show()
```

More sigmoid-like functions can be found in the [here](../api/standardize.rst#sigmoid-like-standardization).

(std.gaussian)=

### Gaussian-like functions

Gaussian-like functions create a bell-shaped response around an central value and decrease away from it. Such functions are useful when an optimal value is sought, such as optimal temperature or rainfall ranges. Like sigmoid-like functions, Gaussian-like functions can be symmetric or asymmetric, can have a steep or gradual decline, and the width of the plateau can be adjusted.

An example is the {py:func}`standardize.vetharaniam2024_eq8` function, which creates a smooth bell-shaped or plateau-like response around a midpoint.

```{code-cell} ipython3
x = np.linspace(0, 20, 100)
y = lstd.vetharaniam2024_eq8(x, a=0.00005, b=10.0, c=6.0)
ybis = lstd.vetharaniam2024_eq8(x, a=0.001, b=10.0, c=6.0)

plt.plot(x, y, label="a=0.00005")
plt.plot(x, ybis, ls="--", label="a=0.001")
plt.legend()
plt.show()
```

More Gaussian-like functions can be found in the [here](../api/standardize.rst#gaussian-like-standardization).

(std.fit)=

## Fitting a function to your data

If you are unsure which standardization function is appropriate, {py:func}`standardize.fit` can help you fit candidate equations to known data points.

```{code-cell} ipython3
x = np.array([0, 10, 20, 30, 40])
y = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

func, params = lstd.fit(x, y, kind="sigmoid_like", verbose=True)
print(func)
print(params)
```

The `fit` helper tries the available equations and returns the best fitting function and its parameters. This is useful when you want to calibrate a suitability curve from expert judgment or empirical data.

You can also visualize the fitted curves by setting `plot=True`.

(std.xarray)=

## Using standardization with xarray indicators

LSAPy is designed around the concept of {py:class}`xarray.DataArray` indicators, and standardization functions can be applied directly to these objects. This allows you to preserve the spatial coordinates and metadata.

```{code-cell} ipython3
import xarray as xr

ind = xr.DataArray(
    np.array([[0, 2, 4], [6, 8, 10]]),
    dims=("y", "x"),
    coords={"y": [10, 20], "x": [30, 40, 50]},
)

lstd.logistic(ind, a=0.7, b=5.0)
```

(std.choice)=

## Choosing the right function

A good standardization function depends on the criteria. Here are some general guidelines:

- Use `boolean` when a hard threshold is appropriate.
- Use `discrete` for ordinal or categorical indicators.
- Use *sigmoid-like* functions for transitions around a midpoint.
- Use *Gaussian-like* functions when an optimum is sought.

(std.workflow)=

## Standardization in the LSAPy workflow

Standardization functions can be integrated into the LSAPy workflow by passing them as the `func` argument of a
{py:class}`SuitabilityCriteria`. The parameters for the function can be specified in the `fparams` dictionary.

```{code-cell} ipython3
from lsapy import SuitabilityCriteria

sc = SuitabilityCriteria(
    name="slope",
    indicator=ind,
    func=lstd.logistic,
    fparams={"a": 0.7, "b": 5.0},
)
```

The suitability criteria can be combined with others in a {py:class}`LandSuitabilityAnalysis` to compute the land use suitability.

This guide only covers the core concepts and families of functions available in `standardize`. For function-specific details, see the API reference in the standardization section of the [documentation](../api/standardize.rst).
