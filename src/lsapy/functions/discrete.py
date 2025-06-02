"""Discrete Suitability Function definition."""

import numpy as np

from lsapy.core.functions import SuitabilityFunction

__all__ = [
    "DiscreteFunction",
    "discrete",
]


class DiscreteFunction(SuitabilityFunction):
    """
    Discrete Suitability Function.

    Discrete functions are used to transform discrete indicator values into suitability values. The discrete functions
    map the indicator values to a set of rules that define the suitability values.

    Parameters
    ----------
    rules : dict[str, int | float] | None, optional
        Rules to map the indicator values to suitability values. The keys correspond to the indicator values and the
        values to its associated suitability values.

    See Also
    --------
    MembershipFunction : Membership Suitability Function.

    Examples
    --------
    >>> func = DiscreteFunction(rules={1: 0, 2: 0.1, 3: 0.5, 4: 0.9, 5: 1})

    ``DiscreteFunction`` also support keys as strings.

    >>> func = DiscreteFunction(rules={"1": 0, "2": 0.1, "3": 0.5, "4": 0.9, "5": 1})
    """

    def __init__(self, rules: dict[str | int, int | float] | None = None):
        super().__init__(func=discrete, name="discrete", params={"rules": rules})


def discrete(x, rules: dict[str | int, int | float]) -> float:
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
    float
        Suitability values.
    """
    return np.vectorize(rules.get, otypes=[np.float32])(x, np.nan)
