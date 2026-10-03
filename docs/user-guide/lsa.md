---
file_format: mystnb
---

```{eval-rst}
.. currentmodule:: lsapy
```

(lsa)=

# Performing a Land Suitability Analysis

{py:class}`LandSuitabilityAnalysis` combines several
{py:class}`SuitabilityCriteria` objects to perform a
Land Suitability Analysis (LSA). It computes the suitability
of each criteria and can then aggregate those scores into
category and overall suitability values. A LSA is defined by:

- `land_use`, the land use being evaluated;
- `criteria`, a dictionary of {py:class}`SuitabilityCriteria`;
- optional metadata set using `attrs`

The criteria workflow is described in [Defining a SuitabilityCriteria](./suitability-criteria.md).

(lsa.create)=

## Creating a LandSuitabilityAnalysis

Here, we are going to do a simple example of a land suitability analysis, using three criteria: slope, rooting depth, and water requirement.
Note that a real LSA would likely include a larger number and more complex criteria.

First, we create a {py:class}`xarray.Dataset` containing three variables: `slope`, `rooting_depth`, and `prctot` (total precipitation).

```{code-cell} ipython3
import numpy as np
import xarray as xr

coords = {"y": [0, 1], "x": [0, 1, 2]}
inds = xr.Dataset(
    {
        "slope": xr.DataArray(
            np.array([[2, 8, 10], [12, 15, 20]]),
            dims=("y", "x"),
            coords=coords,
            attrs={"units": "degrees"},
        ),
        "rooting_depth": xr.DataArray(
            np.array([[0.2, 0.5, 0.8], [0.9, 0.4, 0.1]]),
            dims=("y", "x"),
            coords=coords,
            attrs={"units": "m"},
        ),
        "prctot": xr.DataArray(
            np.array([[800, 1200, 1500], [1600, 1000, 500]]),
            dims=("y", "x"),
            coords=coords,
            attrs={"units": "mm"},
        ),
    }
)

inds
```

We can now create the relevant {py:class}`SuitabilityCriteria` objects with the above indicators, and define the {py:class}`LandSuitabilityAnalysis`.

```{code-cell} ipython3
import lsapy.standardize as lstd
from lsapy import LandSuitabilityAnalysis, SuitabilityCriteria

sc = {
    "slope": SuitabilityCriteria(
        name="slope",
        indicator=inds["slope"],
        func=lstd.logistic,
        fparams={"a": -1.0, "b": 15.0},
        description="Slope suitability",
        comment="A comment about slope suitability",
        category="soil",
        weight=2,
    ),
    "rooting_depth": SuitabilityCriteria(
        name="rooting_depth",
        indicator=inds["rooting_depth"],
        func=lstd.logistic,
        fparams={"a": 10, "b": 0.5},
        description="Rooting depth suitability",
        comment="A comment about rooting depth suitability",
        category="soil",
        weight=1,
    ),
    "water_requirement": SuitabilityCriteria(
        name="water_requirement",
        indicator=inds["prctot"],
        func=lstd.logistic,
        fparams={"a": 0.01, "b": 1000.0},
        description="Water requirement suitability",
        comment="A comment about water requirement suitability",
        category="climate",
        weight=3,
    ),
}

lsa = LandSuitabilityAnalysis(
	land_use="land_use_name",
	long_name="Land Suitability",
	criteria=sc
)
```

Criteria are sorted by descending weight when the analysis is created. The
analysis exposes the grouping information used during aggregation:

```python
lsa.category
# ['climate', 'soil']
lsa.criteria_by_category
# {'climate': ['water_requirement'], 'soil': ['slope', 'rooting_depth']}
lsa.weights_by_category
# {'climate': 3, 'soil': 3}
```

(lsa.run)=

## Computing suitability

{py:meth}`LandSuitabilityAnalysis.run` is the only method needed to compute suitability. Several types of suitability can be computed, depending on the `suitability_type` argument, and are described below.

(lsa.run.criteria)=

### Criteria suitability

Set `suitability_type="criteria"` to compute every criteria without
aggregating it:

```{code-cell} ipython3
lsa.run(suitability_type="criteria")
```

The result is an `xarray.Dataset` with one data variable per criteria. Each
variable retains the criteria metadata and contains suitability values in the
range expected from its standardization function. The dataset also includes
analysis metadata such as `land_use` and the list of criteria.

(lsa.run.category)=

### Aggregating by category

To compute category suitability, set `suitability_type="category"`. This computes the suitability of each criteria and then aggregates them by category using the specified [aggregation method](#lsa.agg).

```{code-cell} ipython3
res = lsa.run(suitability_type="category")
res
```

This returns the same dataset as the criteria calculation, but with an additional variable for each category (i.e., `climate` and `soil`).
The information about the aggregation method used is stored in the `attrs` of each category variable.

```{code-cell} ipython3
res["climate"].attrs
```

(lsa.run.overall)=

### Computing overall suitability

By setting `suitability_type="overall"`, the overall suitability is computed:

```{code-cell} ipython3
lsa.run(suitability_type="overall")
```

By default, the overall suitability is computed by aggregating the suitability of categories if categories are defined, or by aggregating the criteria directly if no categories are defined. This behavior can be controlled with the `by_category` argument:

```{code-cell} ipython3
lsa.run(suitability_type="overall", by_category=False)
```

(lsa.agg)=

## Aggregation methods

The available aggregation methods are the following:

| Name        | Description                                   |
| ----------- | --------------------------------------------- |
| `mean`      | Arithmetic mean                               |
| `median`    | Median                                        |
| `wmean`     | Weighted arithmetic mean                      |
| `gmean`     | Geometric mean                                |
| `wgmean`    | Weighted geometric mean                       |
| `limfactor` | Minimum suitability and the limiting variable |

The weighted methods (i.e., `wmean` and `wgmean`) use the `weight` attribute of each criteria. The `limfactor` method returns a dataset containing `limiting_factor` and `limiting_variable`.

The `mean` aggregation method is used by default, but any of the other methods can be specified with the `agg_methods` argument. For example, to compute the overall suitability using a weighted geometric mean:

```{code-cell} ipython3
lsa.run(suitability_type="overall", agg_methods="wgmean")
```

For different methods at the category and overall levels, pass a dictionary. The `category` key controls category aggregation and `overall` controls the final aggregation:

```{code-cell} ipython3
lsa.run(
    suitability_type="overall",
    agg_methods={"category": "wgmean", "overall": "mean"},
)
```

Here the category suitability is computed using a weighted geometric mean, and the overall suitability is computed using an arithmetic mean of the category scores.

More details about the aggregation methods is available in the [Aggregation Methods](./aggregation.md) guide.

(lsa.vars)=

## Controlling the output

`keep_vars=True` is the default. It keeps intermediate criterion and category
variables alongside the requested result, which is useful for inspecting the
calculation or keeping the intermediate results. Set it to `False` to return only the requested output:

```python
res = lsa.run(
    suitability_type="category",
    keep_vars=False,
)
list(res.data_vars)
# ['climate', 'soil']

res = lsa.run(
    suitability_type="overall",
    keep_vars=False,
)
list(res.data_vars)
# ['suitability']
```

(lsa.align)=

## Input alignment

All criteria indicators must have exactly matching dimensions and coordinates.
If indicators come from different grids, align or resample them before
creating the analysis. For example using {py:meth}`xarray.DataArray.interp_like` to align a
climate indicator to a soil indicator:

```python
climate_indicator = climate_indicator.interp_like(
    soil_indicator,
    method="nearest",
)
```

Otherwise, `run()` raises an `xarray` alignment error.
