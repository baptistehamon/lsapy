"""Suitability Function Module."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from lsapy.core.functions import get_function_from_name

__all__ = [
    "SuitabilityFunction",
]


class SuitabilityFunction:
    """
    Suitability Function.

    Suitability function define how the criteria indicator is transformed into a suitability value. The suitability
    function are available for continuous and discrete indicators. For continuous indicators, a membership function
    is applied to convert indicator values into a suitability. For discrete indicators, a set of rules is mapped
    on the indicator.

    Parameters
    ----------
    func : Callable | None, optional
        Function to compute the suitability value.
    name : str | None, optional
        Name of the implemented function to use. If `func` is provided, this parameter is ignored.
    params : dict[str, Any], optional
        Parameters of the function.

    Examples
    --------
    >>> func = SuitabilityFunction(name="logistic", params={"a": 1, "b": 5})
    >>> func(3)
    array(0.11920292, dtype=float32)

    ``SuitabilityFunction`` can also be used for discrete functions.

    >>> func = SuitabilityFunction(name="discrete", params={"rules": {1: 0, 2: 0.1, 3: 0.5, 4: 0.9, 5: 1}})
    >>> func(3)
    array(0.5, dtype=float32)
    """

    def __init__(self, func: Callable | None = None, name: str | None = None, params: dict[str, Any] | None = None):
        if func is None:
            if name is None:
                raise ValueError("Either `func` or `name` must be provided to define the suitability function.")
            elif not isinstance(name, str):
                raise TypeError("`name` must be a string when `func` is not provided.")
            else:
                self.func = get_function_from_name(name)
        else:
            if not isinstance(func, Callable):
                raise TypeError("`func` must be a callable function.")
            if name is not None:
                warnings.warn("`name` is ignored when `func` is provided", stacklevel=2)
            self.func = func

        self.params = params

    def __call__(self, x):
        """Call the suitability function."""
        if self.func is None:
            raise ValueError("No function has been provided.")
        if self.params:

            def func(val):
                """Wrap the function to include parameters."""
                return self.func(val, **self.params)
        else:

            def func(val):
                """Wrap the function without parameters."""
                return self.func(val)

        return np.vectorize(func, otypes=[np.float32])(x)

    def __repr__(self):
        """Return the string representation of the object."""
        return f"{self.__class__.__name__}(func={self.func.__name__}, params={self.params})"

    @property
    def attrs(self) -> dict[str, Any]:
        """
        Dictionary of the suitability function attributes.

        Returns
        -------
        dict
            Dictionary containing the function name and parameters. If both are undefined, an empty dictionary
            is returned.
        """
        return {k: v for k, v in {"func": self.func, "params": self.params}.items() if v is not None}

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
            FutureWarning,
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
