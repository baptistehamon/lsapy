"""Miscellaneous Utilities."""

from pathlib import Path

import xarray as xr


def _load_data(name: str) -> xr.Dataset:
    """
    A sample dataset.

    Returns a sample dataset for testing and example purposes.

    Parameters
    ----------
    name : {soil, climate}
        The name of the dataset to load.

    Returns
    -------
    xr.Dataset
        The sample dataset.
    """
    if name.lower() not in ["soil", "climate"]:
        raise ValueError(f"Invalid data name: {name}. Must be one of 'soil' or 'climate'.")

    data_path = Path(__file__).parent / "data"
    return xr.open_dataset(data_path / f"{name.lower()}_sample.nc")


def load_soil_data() -> xr.Dataset:
    """
    A soil dataset.

    Returns a soil dataset provided for testing and example purposes.

    Returns
    -------
    xr.Dataset
        The sample soil dataset.
    """
    return _load_data("soil")


def load_climate_data() -> xr.Dataset:
    """
    A climate dataset.

    Returns a climate dataset provided for testing and example purposes.

    Returns
    -------
    xr.Dataset
        The sample climate dataset.
    """
    return _load_data("climate")
