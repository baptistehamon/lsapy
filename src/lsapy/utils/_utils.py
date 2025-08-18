"""Module for utility functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pooch
import xarray as xr
from pooch import Unzip

__all__ = ["open_data", "kuri"]

DATA_REALMS = {
    "climate": [
        "pr",
        "tas",
        "tasmax",
        "tasmin",
    ],
    "land": [
        "aspect",
        "cation-exchange-capacity",
        "depth-slowly-permeable-horizon",
        "drainage",
        "erosion-severity",
        "flood-return-interval",
        "land-cover",
        "land-use-capability",
        "lucas-land-use",
        "particle-size",
        "permeability-profile",
        "ph",
        "phosphate-retention",
        "potential-rooting-depth",
        "profile-readily-available-water",
        "profile-total-available-water",
        "rock",
        "salinity",
        "slope",
        "soil-temperature-regime",
        "topsoil-gravel-content",
        "total-carbon",
    ],
}

registry_file = Path(__file__).parent / "../data/registry.txt"


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
    )
    _kuri.load_registry(registry_file)

    return _kuri


def _check_realm_vars(realm: str, variables: str | list | None = None) -> list:
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
    """Format variable names by replacing hyphens with underscores."""
    variables = [v.replace("-", "_") for v in variables]
    if len(variables) == 1:
        return variables[0]
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
    else:
        fname = "nzglid_5km.zip"
        unpack = Unzip(members=[f"NZGLID_{v}_NZ5km.nc" for v in variables])
    if "unpack" not in locals():
        unpack = None

    fnames = kuri().fetch(fname, progressbar=True, processor=unpack)

    if variables is None:
        variables = DATA_REALMS[realm]
    variables = _format_vars_names(variables)
    return xr.open_mfdataset(fnames, **kwargs)[variables]
