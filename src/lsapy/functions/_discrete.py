"""Discrete Function Module."""

import operator

import numpy as np

from lsapy.core.functions import declare_equation

__all__ = ["discrete", "boolean"]

binary_ops = {">": "gt", "<": "lt", ">=": "ge", "<=": "le", "==": "eq", "!=": "ne"}


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


@declare_equation("boolean")
@np.vectorize
def boolean(x, op: str, thresh: int | float, skipna: bool = True):
    """
    Boolean function.

    This function applies a boolean operation to the values based on a threshold.

    Parameters
    ----------
    x : any
        Input values.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator.
    thresh : any
        Threshold value.
    skipna : bool, optional
        Whether to skip NaN values. If True, NaN values in `x` will remain NaN in the output. Default is True.

    Returns
    -------
    np.ndarray
        Boolean mask of the operation.
    """
    if op in binary_ops:
        op = binary_ops[op]
    elif op in binary_ops.values():
        pass
    else:
        raise ValueError(f"Operator '{op}' not recognized.")

    res = getattr(operator, op)(x, thresh)
    if skipna:
        mask = np.isnan(x)
        return np.where(mask, np.nan, res)
    return res
