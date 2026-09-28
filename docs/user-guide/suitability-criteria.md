---
file_format: mystnb
---

# Defining a SuitabilityCriteria

{py:class}`lsapy.SuitabilityCriteria` connects an indicator to the rule that
converts it into a suitability score. A criteria contains:

- a `name` and a {py:class}`xarray.DataArray` `indicator`;
- a suitability function and its parameters;
- an optional `weight` and `category` for aggregation; and
- optional metadata describing the criteria.

## Creating a SuitabilityCriteria

Here, we define a slope criteria. First, we create a simple example
{py:class}`xarray.DataArray` indicator, with two spatial dimensions and
a slope in degrees.

```{code-cell} ipython3
import numpy as np
import xarray as xr

import lsapy.standardize as lstd
from lsapy import SuitabilityCriteria

slope_data = xr.DataArray(
	np.array([[2, 8, 10], [12, 15, 20]]),
	dims=("y", "x"),
	coords={"y": [45, 46], "x": [2, 3, 4]},
	name="slope",
	attrs={"units": "degrees"},
)
```

We can then create a {py:class}`lsapy.SuitabilityCriteria` using the
indicator and a standardization function. In this example, we use the
{py:func}`lsapy.standardize.logistic` function with parameters `a=-1`
and `b=15`, which means that suitability decreases from 1 to 0
with a midpoint at 15 degrees.

```{code-cell} ipython3
slope = SuitabilityCriteria(
	name="slope",
	indicator=slope_data,
	func=lstd.logistic,
	fparams={"a": -1, "b": 15},
	long_name="Slope suitability"
)
```

*See the [Standardizing your data](./standardization.md) guide to learn more
about the available standardization functions and how to choose their
parameters.*

## Computing suitability

Call `compute()` to apply the function and obtain a new `DataArray`:

```{code-cell} ipython3
slope.compute()
```

The result is named after the criteria and contains the computed values and
the criteria metadata. The `history` attribute is written recording the
function and source indicator. Coordinates and dimensions remain unchanged.

By default, `compute()` returns the computed result without modifying the
criteria. Use `inplace=True` to store the result in the criteria and mark it as computed:

```{code-cell} ipython3
slope.compute(inplace=True)
print(slope.is_computed)
```

After this, `slope.indicator` contains suitability values and subsequent calls
to `compute()` use those values directly.

## Weights and categories

Depending on the method you want to use to compute the suitability using
{py:class}`lsapy.LandSuitabilityAnalysis`, criteria can be assigned a `weight`
and a `category`:

- Weights control the relative contribution of criteria when a weighted
  aggregation method is used. They must be positive numbers; omitted or `None`
  weights become `1.0`.
- Categories group related criteria (e.g.,`"climate"` and `"soil"`).
  Categories are optional, but assigning them allows the analysis to aggregate
  criteria by category before calculating an overall suitability.

Metadata can be provided through the `attrs` mapping for additional fields:

```python
drainage = SuitabilityCriteria(
    name="drainage_class",
    indicator=drainage_ind,
    func=lstd.discrete,
    fparams={"rules": {1: 0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1}},
    category="soil",
    weight=3,
    attrs={"description": "A description of the criteria", "comment": "Even more information"},
)
```

The metadata is copied to the computed `DataArray`, making the output
self-describing when it is saved or passed to another workflow.

## Using precomputed suitability values

Sometimes, the indicator may already contain suitability values, for example
if it was computed in a previous step or provided by an external source. In this case,
you can create a {py:class}`lsapy.SuitabilityCriteria` with `is_computed=True`.
This tells LSAPy that the indicator already contains suitability values and that
no further computation is needed.

```python
precomputed = SuitabilityCriteria(name="criteria_name", indicator=ind, is_computed=True)

result = precomputed.compute()
```

## Combining criteria in an analysis

Once each criteria has been defined, put them in a dictionary and pass it to
{py:class}`lsapy.LandSuitabilityAnalysis`. The dictionary keys conventionally
match the criteria names.

```{code-cell} ipython3
from lsapy import LandSuitabilityAnalysis

rooting_depth_ind = xr.DataArray(
	np.array([[0.2, 0.5, 0.8], [0.9, 0.4, 0.1]]),
	dims=("y", "x"),
	coords=slope_data.coords,
	name="rooting_depth",
	attrs={"units": "m"},
)

rooting_depth = SuitabilityCriteria(
	name="rooting_depth",
	indicator=rooting_depth_ind,
	func=lstd.logistic,
	fparams={"a": 8, "b": 0.5},
)

lsa = LandSuitabilityAnalysis(
	land_use="land_use_name",
	criteria={"slope": slope, "rooting_depth": rooting_depth}
)

result = lsa.run()
```

`LandSuitabilityAnalysis.run()` computes the criteria before applying the
selected aggregation method. See [Land suitability analysis](./lsa.md) for
aggregation methods, category-level suitability, and analysis metadata.
