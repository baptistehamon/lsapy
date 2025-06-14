"""Suitability Function Utilities."""

import warnings
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from attr import asdict, define, field

__all__ = ["SuitabilityFunction"]


@define
class SuitabilityFunction:
    """
    Suitability Function base class.

    Suitability function define how the criteria indicator is transformed into a suitability value. The suitability
    function are available for continuous and discrete indicators. For continuous indicators, a membership function
    is used to transform the indicator into a suitability value. For discrete indicators, a set of rules is mapped
    on the indicator.

    Parameters
    ----------
    func : Callable | None, optional
        Function to compute the suitability value.
    name : str | None, optional
        Name of the function to compute the suitability value. By default, the name of the function is used if provided,
        otherwise it is set to `None`.
    params : dict[str, Any], optional
        Parameters of the function.

    See Also
    --------
    MembershipFunction : Membership Suitability Function.
    DiscreteFunction : Discrete Suitability Function.

    Examples
    --------
    >>> from lsapy.functions import logistic
    >>> sf = SuitabilityFunction(func=logistic, params={"a": 1, "b": 5})

    ``SuitabilityFunction`` can also be used for discrete functions.

    >>> from lsapy.functions import discrete
    >>> sf = SuitabilityFunction(func=discrete, params={1: 0, 2: 0.1, 3: 0.5, 4: 0.9, 5: 1})
    """

    func: Callable | None = field(repr=lambda f: f.__name__ if f else None)
    name: str | None
    params: dict[str, Any] | None

    def __init__(self, func: Callable | None = None, name: str | None = None, params: dict[str, Any] = None):
        if func is not None and not callable(func):
            raise TypeError("`func` must be a callable function.")
        self.func = func

        if name:
            self.name = name
        elif func is not None:
            self.name = func.__name__
        else:
            self.name = None

        self.params = params

    def __call__(self, x):
        """Call the suitability function."""
        if self.func is None:
            raise ValueError("No function has been provided.")
        return np.vectorize(self.func, otypes=[np.float32])(x, **self.params)

    def map(self, x):
        """
        Map the suitability function.

        This method converts the input values into suitability values for the defined function.

        Parameters
        ----------
        x : any
            Input values to map.

        Returns
        -------
        any
            Suitability values.

        Raises
        ------
        ValueError
            If no function has been provided.

        Examples
        --------
        >>> from lsapy.functions import logistic

        >>> sf = SuitabilityFunction(func=logistic, params={"a": 1, "b": 5})
        >>> sf.map(3)
        array(0.11920292, dtype=float32)

        .. deprecated:: 0.1.0-dev2
          `map` will be removed in LSAPy 0.1.0 because it is redundant with the `__call__` method.
          Please use the `__call__` method directly instead.
        """
        warnings.warn(
            "`map` is deprecated and will be removed in LSAPy 0.1.0. Use `__call__` directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self(x)

    def plot(self, x) -> None:
        """
        Basic plot of the suitability function.

        Parameters
        ----------
        x : any
            Input values to plot.

        Examples
        --------
        >>> import numpy as np  # doctest: +SKIP
        >>> from lsapy.functions import logistic

        >>> sf = SuitabilityFunction(func=logistic, params={"a": 1, "b": 5})
        >>> sf.plot(np.linspace(0, 10, 100))  # doctest: +SKIP
        """
        plt.plot(x, self(x))

    @property
    def attrs(self):
        """
        Dictionary of the suitability function attributes.

        Returns
        -------
        dict
            Dictionary containing the function name and parameters. If both are undefined, an empty dictionary
            is returned.
        """
        return {k: v for k, v in asdict(self).items() if v is not None and k not in ["func"]}
