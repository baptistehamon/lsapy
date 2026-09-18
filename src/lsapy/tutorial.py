"""Module containing functions for tutorials and examples."""

from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path

import xarray as xr

__all__ = ["open_dataset", "load_dataset"]

_default_cache_dir_name = "lsapy_tutorial_data"
base_url = "https://github.com/baptistehamon/lsapy-data"
default_branch = "main"

citrus_dataset = [
    "drainage",
    "ph",
    "potential-rooting-depth",
    "prcptot",
    "profile-total-available-water",
    "salinity",
    "slope",
    "tgmean_0915-1115",
    "tnmean_0815-1015",
    "topsoil-gravel-content",
    "txmean_0101-0215",
    "year-with-hotweek_1201-0228",
]


def _check_netcdf_dependencies():
    """Check if either h5netcdf or netCDF4 is installed."""
    if not find_spec("h5netcdf") and not find_spec("netCDF4"):
        raise ImportError(
            "'lsapy' tutorials data are in netCDF format and opening them requires either 'h5netcdf' or 'netCDF4'."
        ) from None


# idea borrowed from xarray.tutorial
def open_dataset(
    names,
    cache_dir: str | Path | None = None,
    cache: bool = True,
    **kwargs,
) -> xr.Dataset:
    """
    Open a dataset from the LSAPy tutorial data repository (requires internet).

    If a local copy of the file(s) exists, it will be used to avoid network traffic.

    Available datasets:

    * ``"drainage"``: Drainage class*
    * ``"ph"``: Soil pH*
    * ``"potential-rooting-depth"``: Soil potential rooting depth*
    * ``"prcptot"``: Annual total precipitation*
    * ``"profile-total-available-water"``: Soil profile total available water*
    * ``"salinity"``: Soil salinity*
    * ``"slope"``: Slope*
    * ``"tgmean_0915-1115"``: Mean annual temperature (Sep 15 to Nov 15)*
    * ``"tnmean_0815-1015"``: Mean annual minimum temperature (Aug 15 to Oct 15)*
    * ``"topsoil-gravel-content"``: Topsoil gravel content*
    * ``"txmean_0101-0215"``: Mean annual maximum temperature (Jan 1 to Feb 15)*
    * ``"year-with-hotweek_1201-0228"``: Number of years with at least one hot week (3 days over 35C in a 7-day period)
      between Dec 1 and Feb 28 (over 10 years)*
    * ``"citrus-dataset"``: A dataset containing various soil and climate variables for a citrus suitability analysis*

    *These datasets covers the North Island of New Zealand at ~5 km resolution.

    Parameters
    ----------
    names : str or list of str
        Name(s) of the dataset(s) to open. See above for available datasets.
    cache_dir : str or Path, optional
        Directory where to read or write the cached data.
    cache : bool, optional
        If True (default), cache the data locally for later use.
    **kwargs : Any
        Additional keyword arguments passed to ``xarray.open_mfdataset``.

    Returns
    -------
    xarray.Dataset
        The requested dataset(s).

    See Also
    --------
    load_dataset
    """
    if not find_spec("pooch"):
        raise ImportError(
            "'lsapy' depends on 'pooch' to download tutorial data. To proceed, please install 'pooch'."
        ) from None
    _check_netcdf_dependencies()
    import pooch

    logger = pooch.get_logger()
    logger.setLevel("WARNING")

    if cache_dir is None:
        cache_dir = pooch.os_cache(_default_cache_dir_name)
    else:
        cache_dir = Path(cache_dir)

    if isinstance(names, str):
        if names == "citrus-dataset":
            names = citrus_dataset
        else:
            names = [names]

    headers = {"User-Agent": f"lsapy {sys.modules['lsapy'].__version__}"}
    downloader = pooch.HTTPDownloader(headers=headers)

    # retrieve file(s)
    filepaths = []
    for name in names:
        url = f"{base_url}/raw/{default_branch}/{name + '_nz-north-island.nc'}"
        filepath = pooch.retrieve(
            url=url,
            known_hash=None,
            path=cache_dir,
            downloader=downloader,
        )
        filepaths.append(filepath)

    ds = xr.open_mfdataset(filepaths, **kwargs)

    if not cache:
        ds = ds.load()
        for f in filepaths:
            Path(f).unlink()  # delete the file if not cached

    return ds


def load_dataset(*args, **kwargs) -> xr.Dataset:  # numpydoc ignore=PR01,PR02
    """
    Open, load and close a dataset from the LSAPy tutorial data repository (requires internet).

    if a local copy of the file(s) exists, it will be used to avoid network traffic.

    Available datasets:

    * ``"drainage"``: Drainage class*
    * ``"ph"``: Soil pH*
    * ``"potential-rooting-depth"``: Soil potential rooting depth*
    * ``"prcptot"``: Annual total precipitation*
    * ``"profile-total-available-water"``: Soil profile total available water*
    * ``"salinity"``: Soil salinity*
    * ``"slope"``: Slope*
    * ``"tgmean_0915-1115"``: Mean annual temperature (Sep 15 to Nov 15)*
    * ``"tnmean_0815-1015"``: Mean annual minimum temperature (Aug 15 to Oct 15)*
    * ``"topsoil-gravel-content"``: Topsoil gravel content*
    * ``"txmean_0101-0215"``: Mean annual maximum temperature (Jan 1 to Feb 15)*
    * ``"year-with-hotweek_1201-0228"``: Number of years with at least one hot week (3 days over 35C in a 7-day period)
      between Dec 1 and Feb 28 (over 10 years)*
    * ``"citrus-dataset"``: A dataset containing various soil and climate variables for a citrus suitability analysis*

    *These datasets covers the North Island of New Zealand at ~5 km resolution.

    Parameters
    ----------
    names : str or list of str
        Name(s) of the dataset(s) to open. See above for available datasets.
    cache_dir : str or Path, optional
        Directory where to read or write the cached data.
    cache : bool, optional
        If True (default), cache the data locally for later use.
    **kwargs : Any
        Additional keyword arguments passed to ``xarray.open_mfdataset``.

    Returns
    -------
    xarray.Dataset
        The requested dataset(s).

    See Also
    --------
    open_dataset
    """
    with open_dataset(*args, **kwargs) as ds:
        return ds.load()
