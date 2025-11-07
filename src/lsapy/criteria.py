"""Suitability Criteria definition."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import xarray as xr

import lsapy.core.formatting as fmt
from lsapy.core.functions import get_function_from_name

__all__ = ["SuitabilityCriteria"]


class SuitabilityCriteria:
    """
    A data structure for suitability criteria.

    Suitability criteria are used to compute the suitability of a location from an indicator and based on a set of rules
    defined by a suitability function. The suitability criteria can be weighted and categorized defining how it will be
    aggregated with other criteria.

    Parameters
    ----------
    name : str
        Name of the suitability criteria.
    indicator : xr.DataArray
        Indicator on which the criteria is based.
    func : Callable, optional
        Standardization function that takes the indicator as input and returns the suitability values.
    fparams : dict, optional
        A dictionary of parameters to pass to the function `func`. The default is None.
    weight : int | float, optional
        Weight of the criteria used in the aggregation process if a weighted aggregation method is used.
        The default is 1.
    category : str, optional
        Category of the criteria. The default is None.
    long_name : str, optional
        A long name for the criteria. The default is None. If provided, it will be stored as an attribute.
    description : str, optional
        A description for the criteria. The default is None. If provided, it will be stored as an attribute.
    comment : str, optional
        Additional information about the criteria. The default is None.
        If provided, it will be stored as an attribute.
    attrs : Mapping[Any, Any], optional
        Arbitrary metadata to store with the criteria, in addition to the attributes
        `long_name`, `description`, and `comment`. The default is None.
    is_computed : bool, optional
        If the indicator data already contains the computed suitability values. Default is False.

    Examples
    --------
    Here is an example using the sample soil data with the drainage class (DRC) as indicator for the criteria.

    >>> from lsapy.utils import open_data
    >>> from lsapy.functions import SuitabilityFunction
    >>> from xclim.indicators.atmos import growing_degree_days

    >>> drainage = open_data("land", variables="drainage")
    >>> sc = SuitabilityCriteria(
    ...     name="drainage_class",
    ...     long_name="Drainage Class Suitability",
    ...     weight=3,
    ...     category="soilTerrain",
    ...     indicator=drainage,
    ...     func="discrete",
    ...     fparams={"rules": {0: 0, 1: 0.1, 2: 0.5, 3: 0.9, 4: 1}},
    ... )

    Here is another example using the sample climate data with the growing degree days (GDD)
    as indicator for the criteria computing using the `xclim` package.

    >>> tas = open_data("climate", variables="tas")
    >>> gdd = growing_degree_days(tas, thresh="10 degC", freq="YS-JUL")
    >>> sc = SuitabilityCriteria(
    ...     name="growing_degree_days",
    ...     long_name="Growing Degree Days Suitability",
    ...     weight=1,
    ...     category="climate",
    ...     indicator=gdd,
    ...     func="vetharaniam2022_eq5",
    ...     fparams={"a": -1.41, "b": 801},
    ... )
    """

    def __init__(
        self,
        name: str | None = None,
        indicator: xr.DataArray | None = None,
        func: Callable | None = None,
        fparams: dict[str, Any] | None = None,
        weight: int | float | None = 1,
        category: str | None = None,
        long_name: str | None = None,
        description: str | None = None,
        comment: str | None = None,
        attrs: Mapping[Any, Any] | None = None,
        is_computed: bool = False,
    ) -> None:
        self.name = name
        self.indicator = indicator
        self.weight = weight
        self.category = category

        if isinstance(func, str):
            func = get_function_from_name(func)
        self.func = partial(func, **fparams) if fparams else func

        self._attrs = {}
        if long_name:
            self._attrs["long_name"] = long_name
        if description:
            self._attrs["description"] = description
        if comment:
            self._attrs["comment"] = comment
        if attrs and isinstance(attrs, Mapping):
            self._attrs.update(attrs)

        self.is_computed = is_computed

    def __repr__(self) -> str:
        """Return a string representation of the suitability criteria."""
        return fmt.sc_repr(self)

    @property
    def name(self) -> str:
        """
        The name of the criteria.

        Returns
        -------
        str
            The name of the criteria.
        """
        return self._name

    @name.setter
    def name(self, value: str | None) -> None:
        """
        Set the name of the criteria.

        Parameters
        ----------
        value : str | None
            The name of the criteria to set.
        """
        self._name = value

    @property
    def indicator(self) -> xr.DataArray:
        """
        The indicator DataArray.

        Returns
        -------
        xr.DataArray
            The indicator DataArray.
        """
        return self._indicator

    @indicator.setter
    def indicator(self, value: xr.DataArray) -> None:
        """
        Set the indicator DataArray.

        Parameters
        ----------
        value : xr.DataArray
            The indicator DataArray to set.
        """
        if not isinstance(value, xr.DataArray) and value is not None:
            raise TypeError("The indicator must be an xarray DataArray.")
        if value is not None:
            self._from_indicator = _get_indicator_description(value)
        self._indicator = value

    @property
    def func(self) -> Callable | partial | None:
        """
        The standardization function.

        Returns
        -------
        Callable | None
            The standardization function.
        """
        return self._func

    @func.setter
    def func(self, value: Callable | partial | None) -> None:
        """
        Set the standardization function.

        Parameters
        ----------
        value : Callable | None
            The standardization function to set.
        """
        if not isinstance(value, Callable) and value is not None:
            raise TypeError("The function must be a callable.")
        self._func = value

    @property
    def weight(self) -> float:
        """
        The weight of the suitability criteria.

        Returns
        -------
        float
            The weight of the suitability criteria.
        """
        return self._weight

    @weight.setter
    def weight(self, value: int | float | None) -> None:
        """
        Set the weight of the suitability criteria.

        Parameters
        ----------
        value : int | float | None
            The weight of the suitability criteria. If None, the weight is set to 1.
        """
        if value is None:
            self._weight = 1.0
        elif not isinstance(value, (int, float)):
            raise TypeError("The weight must be a number.")
        elif value <= 0:
            raise ValueError("The weight must be a positive number.")
        else:
            self._weight = float(value)

    @property
    def category(self) -> str | None:
        """
        The category of the suitability criteria.

        Returns
        -------
        str | None
            The category of the suitability criteria.
        """
        return self._category

    @category.setter
    def category(self, value: str | None) -> None:
        """
        Set the category of the suitability criteria.

        Parameters
        ----------
        value : str | None
            The category of the suitability criteria. If None, the category is set to None.
        """
        if value is not None and not isinstance(value, str):
            raise TypeError("The category must be a string.")
        self._category = value

    @property
    def is_computed(self) -> bool:
        """
        Whether the indicator data already contains the computed suitability values.

        Returns
        -------
        bool
            True if the indicator data already contains the computed suitability values, False otherwise.
        """
        return self._is_computed

    @is_computed.setter
    def is_computed(self, value: bool) -> None:
        """
        Set whether the indicator data already contains the computed suitability values.

        Parameters
        ----------
        value : bool
            True if the indicator data already contains the computed suitability values, False otherwise.
        """
        if not isinstance(value, bool):
            raise TypeError("is_computed must be a boolean.")
        self._is_computed = value

    @property
    def attrs(self) -> dict[Any, Any]:
        """
        Dictionary of the suitability criteria attributes.

        Returns
        -------
        dict
            Dictionary containing the suitability criteria attributes.
        """
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Any, Any]) -> None:
        """
        Set the attributes of the suitability criteria.

        Parameters
        ----------
        value : Mapping[Any, Any]
            Mapping of attributes to set for the suitability criteria.
        """
        self._attrs = dict(value)

    def compute(self, **kwargs) -> xr.DataArray:
        """
        Compute the suitability of the criteria.

        Returns a xarray DataArray with criteria suitability. The attributes of the DataArray describe how
        the suitability was computed.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the xarray apply_ufunc function.

        Returns
        -------
        xr.DataArray
            Criteria suitability.
        """
        if self.is_computed:
            out = self.indicator
        elif self.func is None:
            raise ValueError("The suitability function is not defined. Please provide a valid function.")
        else:
            out = xr.apply_ufunc(self.func, self.indicator, **kwargs)

        attrs: dict[str, Any] = {"weight": self.weight}
        if self.category:
            attrs["category"] = self.category
        attrs.update(self._attrs)
        attrs["history"] = (
            f"func_method: {self.func if self.func is not None else 'unknown'}; "
            f"from_indicator: [{self._from_indicator}]"
        )
        return out.rename(self.name).assign_attrs(attrs)


def _get_indicator_description(indicator: xr.Dataset | xr.DataArray) -> str:
    if indicator.attrs != {}:
        return f"name: {indicator.name}; " + "; ".join([f"{k}: {v}" for k, v in indicator.attrs.items()])
    else:
        return f"name: {indicator.name}"
