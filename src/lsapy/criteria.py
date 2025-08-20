"""Suitability Criteria definition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import xarray as xr

import lsapy.core.formatting as fmt
from lsapy.functions import SuitabilityFunction

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
    func : SuitabilityFunction, optional
        Suitability function describing how the suitability of the criteria is computed.
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
    ...     func=SuitabilityFunction(name="discrete", params={"rules": {0: 0, 1: 0.1, 2: 0.5, 3: 0.9, 4: 1}}),
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
    ...     func=SuitabilityFunction(name="vetharaniam2022_eq5", params={"a": -1.41, "b": 801}),
    ... )
    """

    def __init__(
        self,
        name: str,
        indicator: xr.DataArray,
        func: SuitabilityFunction | None = None,
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
        self.func = func
        self.weight = weight
        self.category = category

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
        self._from_indicator = _get_indicator_description(indicator)

    def __repr__(self) -> str:
        """Return a string representation of the suitability criteria."""
        return fmt.sc_repr(self)

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
