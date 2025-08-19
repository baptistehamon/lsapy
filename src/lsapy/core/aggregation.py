"""Module for LSA criteria aggregation."""

import numpy as np
import xarray as xr
from scipy import stats

__all__ = ["aggregate"]


def _agg_weights(ds: xr.Dataset, variables: str | list[str] = None, weights: list[float | int] = None) -> xr.DataArray:
    """Returns weights as an xarray.DataArray with given variables as dimensions."""
    if variables is None:
        variables = list(ds.data_vars)
    elif isinstance(variables, str):
        variables = [variables]

    if weights is None:
        weights = np.ones(len(variables))

    if len(variables) != len(weights):
        raise ValueError("Length of variables must match length of weights.")

    shape = tuple(ds.sizes[d] for d in ds.sizes)

    return xr.DataArray(
        np.array([np.full(shape, w) for w in weights]),
        dims=["variable"] + list(ds.dims),
        coords={"variable": variables, **ds.coords},
    )


def _add_agg_attrs(
    da: xr.DataArray | xr.Dataset, method: str, variables: list[str], weights=None
) -> xr.DataArray | xr.Dataset:
    """Add aggregation method attributes to the DataArray or Dataset."""
    if method in ["wmean", "wgmean"] and weights is not None:
        desc_vars = ", ".join([f"{v} ({w})" for v, w in zip(variables, weights, strict=False)])
    else:
        desc_vars = ", ".join(variables)

    attrs = {
        "median": {
            "method": "Median",
            "description": f"Median of the variables: {desc_vars}.",
        },
        "mean": {
            "method": "Mean",
            "description": f"Arithmetic mean of the variables: {desc_vars}.",
        },
        "wmean": {"method": "Weighted Mean", "description": f"Weighted mean of the variables: {desc_vars}."},
        "gmean": {"method": "Geometric Mean", "description": f"Geometric mean of the variables: {desc_vars}."},
        "wgmean": {
            "method": "Weighted Geometric Mean",
            "description": f"Weighted geometric mean of the variables: {desc_vars}.",
        },
        "limiting_factor": {
            "method": "Limiting Factor",
            "description": f"Value of the limiting factor among the variables: {desc_vars}.",
        },
        "limiting_variable": {"method": "Limiting Factor", "description": f"Limiting variable among: {desc_vars}."},
    }

    names = {
        "median": "median",
        "mean": "mean",
        "wmean": "weighted_mean",
        "gmean": "geometric_mean",
        "wgmean": "weighted_geometric_mean",
    }

    if method == "limfactor":
        da["limiting_factor"].attrs.update(attrs.get("limiting_factor", {}))
        da["limiting_variable"].attrs.update(attrs.get("limiting_variable", {}))
    else:
        da.attrs.update(attrs.get(method, {}))
        da.name = names.get(method, method)
    return da


def aggregate(ds: xr.Dataset, method: str = "mean", variables=None, weights=None) -> xr.DataArray | xr.Dataset:
    """
    Aggregate variables of an xarray.Dataset using specified methods.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing the variables to aggregate.
    method : str, optional
        Aggregation method to use. Options include 'mean', 'median', 'wmean' (weighted mean),
        'gmean' (geometric mean), 'wgmean' (weighted geometric mean), and 'limfactor' (limiting factor).
        Default is 'mean'.
    variables : list[str] | None, optional
        List of variable names to aggregate. If None, all variables in the dataset are used.
    weights : list[float | int], optional
        Weights for the variables when using weighted methods ('wmean', 'wgmean').
        If None, equal weights are assumed.

    Returns
    -------
    xr.DataArray | xr.Dataset
        Aggregated data as an xarray.DataArray or xarray.Dataset.
    """
    if isinstance(variables, list):
        ds = ds[variables]
    elif variables is None:
        variables = list(ds.data_vars)
    else:
        raise ValueError("variables must be a list of variable names or None.")

    _weights = _agg_weights(ds, variables, weights)

    da = ds.to_dataarray()

    if method == "median":
        out = da.median(dim="variable").rename("median")
    if method in ["mean", "wmean"]:
        out = da.weighted(_weights).mean(dim="variable")
    if method in ["gmean", "wgmean"]:
        out = da.reduce(stats.gmean, dim="variable", weights=_weights)
    if method == "limfactor":
        limval = da.min(dim="variable").rename("limiting_factor")
        limvar = (
            xr.concat([ds[v] == limval for v in ds.data_vars], dim="variable")
            .assign_coords(variable=list(ds.data_vars))
            .rename("limiting_variable")
        )
        out = xr.merge([limval, limvar]).assign_attrs({"method": "Limiting Factor"})
    return _add_agg_attrs(out, method, variables, weights)
