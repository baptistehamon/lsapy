"""Discrete Function Module."""

import numpy as np

from lsapy.core.functions import declare_equation

__all__ = ["discrete"]


@declare_equation("categorical")
def discrete(x, rules: dict[str | int, int | float]) -> np.ndarray:
    """
    Discrete suitability function.

    This function maps the indicator values to a set of rules that define the suitability values.

    Parameters
    ----------
    x : any
        Indicator values to map.
    rules : dict[str | int, int | float]
        Rules to map the indicator values to suitability values. The keys correspond to the indicator values and the
        values to its associated suitability values.

    Returns
    -------
    np.ndarray
        Suitability values.
    """
    return np.vectorize(rules.get, otypes=[np.float32])(x, np.nan)  # type: ignore[return-value]
