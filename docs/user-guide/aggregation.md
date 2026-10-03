---
file_format: mystnb
---

```{eval-rst}
.. currentmodule:: lsapy
```

(aggregation)=

# Aggregating data

Aggregation is the process of combining multiple variables into a single value. In `LSAPy`, it is used in {py:meth}`LandSuitabilityAnalysis.run` to combine suitability scores from multiple criteria into a single overall suitability score.
The aggregation function can also be used directly on an `xarray.Dataset` of suitability scores with {py:func}`aggregate.aggregate`.

(agg.methods)=

## Available methods

The following aggregation methods are available:

| Method      | Description                                     |
| ----------- | ----------------------------------------------- |
| `mean`      | Arithmetic mean of all values                   |
| `median`    | Median value                                    |
| `wmean`     | Weighted arithmetic mean                        |
| `gmean`     | Geometric mean                                  |
| `wgmean`    | Weighted geometric mean                         |
| `limfactor` | Minimum suitability score and limiting variable |

The weighted methods, `wmean` and `wgmean`, use the weight assigned to each criterion. In a `LandSuitabilityAnalysis`, these weights come from each criterion's `weight` attribute. If you call the lower-level {py:func}`aggregate.aggregate` helper directly, you can pass a `weights` list explicitly.

(agg.comparison)=

## Method comparison

Below is a comparison of the different aggregation methods applied to a simple example dataset with three variables.

```{code-cell} ipython3
import numpy as np
import xarray as xr
from lsapy.aggregate import aggregate

ds = xr.Dataset(
    {
        "var1": xr.DataArray(np.array([[0.2, 0.5], [0.4, 0.0]])),
        "var2": xr.DataArray(np.array([[0.1, 0.6], [0.8, 0.7]])),
        "var3": xr.DataArray(np.array([[0.3, 0.9], [0.4, 1.0]])),
    }
)

# we can aggregate the dataset with different methods
methods = {
    # "method_name": "variable_name_in_dataset",
    "mean": "mean",
    "median": "median",
    "wmean": "weighted_mean",
    "gmean": "geometric_mean",
    "wgmean": "weighted_geometric_mean",
    "limfactor": "limiting_factor",
}
weights = [1, 2, 2]

ds_agg = xr.merge([aggregate(ds, method=method, weights=weights) for method in methods.keys()])
ds_agg
```

We can plot the results to visualize the differences between the aggregation methods:

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(12, 6))
for ax, method in zip(axes.flatten(), methods.values()):
    ds_agg[method].plot(ax=ax, cmap="viridis", vmin=0, vmax=1)
    ax.set_title(method)
plt.tight_layout()
plt.show()
```

The methods produce notably different results, highlighting the importance of selecting an aggregation method suited to your analysis.

(agg.subset)=

## Aggregating a subset of data

If you don't want to aggregate all the variables in the dataset, you can select a subset of the variables to aggregate. For example, if you only want to aggregate `var1` and `var2`, you can do:

```python
aggregate(ds, method="mean", variables=["var1", "var2"])
```

If you use weighted methods, the weights need to match the number of variables selected:

```python
aggregate(ds, method="wmean", variables=["var1", "var2"], weights=[1, 2])
```

(agg.limfactor)=

## Additional information for the `limfactor` method

(agg.limvar)=

### Limiting variable format

When using the `limfactor` method, {py:meth}`aggregate.aggregate` returns the limiting variable (i.e., the variable that had the lowest value) alongside the limiting factor value. It is stored in the `limiting_variable` variable in the aggregated dataset, and corresponds to an {py:class}`xarray.DataArray` with the same shape as the input dataset, but with an additional `variable` dimension that contains boolean values indicating if the variable was the limiting factor.

```{code-cell} ipython3
limvar = ds_agg["limiting_variable"]
limvar
```

We can easily visualize for each variable where it is the limiting factor (1=True, 0=False):

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, var in zip(axes.flatten(), ds.data_vars):
    limvar.sel(variable=var).plot(ax=ax)
    ax.set_title(f"{var}")
plt.tight_layout()
plt.show()
```

This is useful because it shows the areas where a given variable is the limiting factor. For example, we can see that `var1` is the limiting factor in the top-left, top-right and bottom-right corners, while `var2` is the limiting factor in the bottom-left corner. By providing the limiting variable as a boolean value for each variable, we can also identify where two or more variables are limiting factors (i.e., where multiple variables have the same minimum value). Here, we can see that `var1` and `var3` are both limiting factors in the top-left corner.

(agg.limvar.analysis)=

### Advanced limiting variable analysis

While returning the limiting variable in that way is useful, it can be more convenient to have a single variable that indicates which variable is the limiting factor. A quick way to do it is to use the {py:meth}`xarray.DataArray.argmax` method, which returns the index of the maximum (i.e., 1=TRUE) value along a given dimension. In this case, we can use it to find the index of the limiting variable along the `variable` dimension:

```{code-cell} ipython3
limvar.argmax(dim="variable")
```

Here, a value of 0 corresponds to `var1`, 1 to `var2`, and 2 to `var3`. {py:meth}`xarray.DataArray.idxmax` can also be used to return the name of the limiting variable instead of its index:

```{code-cell} ipython3
limvar.idxmax(dim="variable")
```

This two methods will work in most cases, but if there are multiple limiting variables (e.g., top-left corner in this example), they will return the first one found (i.e., `var1` in this case). If you want to find all limiting variables, it is more complicated, but you can use the following code as a workaround:

```{code-cell} ipython3
from itertools import combinations

# create a mapping variable storing an index for each combination of limiting
# variables: i.e., {1: "var1", 2: "var2", 3: "var3", 12: "var1, var2", ...}
mapping = {
    int("".join(str(index + 1) for index in indexes)): ", ".join(
        names[index] for index in indexes
    )
    for size in range(1, len(ds.data_vars) + 1)
    for indexes in combinations(range(len(ds.data_vars)), size)
    for names in [list(ds.data_vars)]
}
mapping
```

```{code-cell} ipython3
# use `xr.apply_ufunc` to apply a function that returns the index
# of the limiting variable(s) for each
limvar = xr.apply_ufunc(
    lambda variable: "".join(
        str(i+1) for i, limiting in enumerate(variable) if limiting
    ),
    limvar,
    input_core_dims=[["variable"]],
    vectorize=True,
    output_dtypes=[int],
)
limvar
```

```{code-cell} ipython3
# create a sorting dictionary to sort the mapping by index
sorting = {k:i for i, k in enumerate(mapping.keys())}
print(sorting)
# update the mapping dictionary to sort it by index
mapping = {k:v for k, v in zip(sorting.values(), mapping.values())}
print(mapping)
```

```{code-cell} ipython3
# use `xr.apply_ufunc` to apply a function that returns the sorted
# index of the limiting variable(s) for each value in `limvar`
limvar = xr.apply_ufunc(
    lambda value: sorting.get(int(value), ""),
    limvar,
    vectorize=True,
    output_dtypes=[int]
)

# assign the mapping as an attribute to the `limvar` DataArray
limvar = limvar.assign_attrs(
    {"mapping": "; ".join(f"{k}: {v}" for k, v in mapping.items())}
)
limvar
```

This workflow results in values indicating which variable(s) is/are the limiting. Each possible combination of limiting variables is assigned a unique index (starting from 1), and the mapping between the index and the variable(s) is stored as an attribute of the `limvar` DataArray. For example, in this case, a value of 1 corresponds to `var1`, 2 to `var2`, 3 to `var3`, 4 to `var1, var2`, etc.

You can plot `limvar` to visualize the limiting variable(s) for each location:

```{code-cell} ipython3
limvar.plot(cmap="tab20", vmin=0, vmax=len(mapping))
plt.show()
```

Interpreting the plot can be difficult, so you can use the following code to create a legend for the mapping:

```{code-cell} ipython3
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

# create a colormap and normalization for the limiting variable(s)
colors = {
    key: plt.get_cmap("tab20")(i)
    for i, key in enumerate(mapping)
}
level_cmap = ListedColormap([colors[key] for key in mapping])
level_norm = BoundaryNorm(
    [key - 0.5 for key in mapping] + [max(mapping) + 0.5],
    level_cmap.N,
)

# define mosaic layout for the plot
mosaic = "AA;BB"

fig, axes = plt.subplot_mosaic(mosaic, figsize=(6, 8), height_ratios=[3, 1])
# add the plot
limvar.plot(
    ax=axes["A"],
    cmap=level_cmap,
    norm=level_norm,
    add_colorbar=False,
)

# create patches for the legend
legend_patches = [
    Rectangle((0, 0), 1, 1, facecolor=colors[key])
    for key in mapping
]

# add the legend
axes["B"].legend(
    legend_patches,
    list(mapping.values()),
    title="Limiting variables",
    loc="center",
    ncols=3,
    frameon=False,
)
axes["B"].axis("off")
plt.show()
```

Now, we can clearly see that `var1` is the limiting variable on the top-right and bottom-right corners, `var2` is the limiting factor in the bottom-left corner, and that `var1` and `var3` are both limiting factors in the top-left corner.
