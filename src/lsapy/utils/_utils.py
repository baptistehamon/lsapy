"""Module for utility functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pooch
import xarray as xr
from pooch import Unzip

__all__ = ["DATA_REALMS", "kuri", "open_data"]

DATA_REALMS = {
    "climate": [
        "pr",
        "tas",
        "tasmax",
        "tasmin",
    ],
    "land": [
        "aspect",
        "cation_exchange_capacity",
        "depth_slowly_permeable_horizon",
        "drainage",
        "erosion_severity",
        "flood_return_interval",
        "land_cover",
        "land_use_capability",
        "lucas_land_use",
        "particle_size",
        "permeability_profile",
        "ph",
        "phosphate_retention",
        "potential_rooting_depth",
        "profile_readily_available_water",
        "profile_total_available_water",
        "rock",
        "salinity",
        "slope",
        "soil_temperature_regime",
        "topsoil_gravel_content",
        "total_carbon",
    ],
}


def kuri() -> pooch.Pooch:
    """
    Pooch instance for LSAPy data.

    Returns
    -------
    pooch.Pooch
        The LSAPy data pooch instance.
    """
    _kuri = pooch.create(
        path=pooch.os_cache("lsapy"),
        base_url="https://raw.githubusercontent.com/baptistehamon/lsapy/main/src/lsapy/data/",
        allow_updates=True,
    )
    _kuri.load_registry(Path(__file__).parent / "../data/registry.txt")

    return _kuri


def _check_realm_vars(realm: str, variables: str | list | None = None) -> list | None:
    """Check validity of realm and variables."""
    if realm not in ["climate", "land"]:
        raise ValueError(f"Realm must be 'climate' or 'land', got '{realm}'.")

    if variables is None:
        return None
    elif isinstance(variables, str):
        variables = [variables]
    elif not isinstance(variables, list):
        raise TypeError("Variable must be a string or a list of strings.")

    for v in variables:
        if v not in DATA_REALMS[realm]:
            raise ValueError(
                f"Variable '{v}' is not supported in realm '{realm}'. "
                f"Supported variables are: '{'', ''.join(DATA_REALMS[realm])}'."
            )

    return variables


def _format_vars_names(variables: list) -> str | list[str]:
    """Format variable names by replacing underscores with hyphens."""
    variables = [v.replace("_", "-") for v in variables]
    return variables


def open_data(realm: str, variables: str | list | None = None, **kwargs: Any) -> xr.Dataset | xr.DataArray:
    """
    Open sample data.

    Parameters
    ----------
    realm : str
        The realm of the dataset, either 'climate' or 'land'.
    variables : str or list, optional
        The variable(s) to load from the dataset. If None (default), all variables for the realm
        will be loaded.
    **kwargs : Any
        Additional keyword arguments to pass to `xarray.open_mfdataset`.

    Returns
    -------
    xr.Dataset or xr.DataArray
        The sample data.
    """
    variables = _check_realm_vars(realm, variables)

    if realm == "climate":
        fname = "NEX-GDDP-CMIP6_day_ACCESS-CM2_historical_r1i1p1f1_20000101-20041231.nc"
    elif realm == "land" and not variables:
        fname = "New-Zealand-Gridded-Land-Information-Dataset_NZ5km.nc"
    elif realm == "land" and variables:
        fname = "nzglid_5km.zip"
        unpack = Unzip(members=[f"NZGLID_{v}_NZ5km.nc" for v in _format_vars_names(variables)])
    if "unpack" not in locals():
        unpack = None

    fnames = kuri().fetch(fname, processor=unpack)

    if variables is None:
        variables = DATA_REALMS[realm]
    elif len(variables) == 1:
        variables = variables[0]
    return xr.open_mfdataset(fnames, **kwargs)[variables].compute()
