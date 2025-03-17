"""Miscellaneous Utilities"""

from pathlib import Path

import xarray as xr


def _load_data(name: str) -> xr.Dataset:
    """Load a sample dataset for testing or example purposes."""
    if name.lower() not in ["soil", "climate"]:
        raise ValueError(f"Invalid data name: {name}. Must be one of 'soil' or 'climate'.")

    data_path = Path(__file__).parent / "data"
    return xr.open_dataset(data_path / f"{name.lower()}_sample.nc")


def load_soil_data() -> xr.Dataset:
    """Load soil dataset for testing or example purposes."""
    return _load_data("soil")


def load_climate_data() -> xr.Dataset:
    """Load climate dataset for testing or example purposes."""
    return _load_data("climate")
