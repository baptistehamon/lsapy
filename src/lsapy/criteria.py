"""Suitability Criteria definition."""

from __future__ import annotations

from collections.abc import Callable

import xarray as xr

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
    func : SuitabilityFunction | MembershipFunction | DiscreteSuitFunction
        Suitability function describing how the suitability of the criteria is computed.
    weight : int | float | None, optional
        Weight of the criteria used in the aggregation process if a weighted aggregation method is used.
        The default is 1.
    category : str | None, optional
        Category of the criteria. The default is None.
    long_name : str | None, optional
        A long name for the criteria. The default is None.
    description : str | None, optional
        A description for the criteria. The default is None.
    comment : str | None, optional
        Additional information about the criteria.
    is_computed : bool, optional
        If the indicator data already contains the computed suitability values. Default is False.

    Examples
    --------
    Here is an example using the sample soil data with the drainage class (DRC) as indicator for the criteria.

    >>> from lsapy.utils import load_soil_data, load_climate_data
    >>> from lsapy.functions import SuitabilityFunction
    >>> from xclim.indicators.atmos import growing_degree_days

    >>> soil_data = load_soil_data()
    >>> sc = SuitabilityCriteria(
    ...     name="drainage_class",
    ...     long_name="Drainage Class Suitability",
    ...     weight=3,
    ...     category="soilTerrain",
    ...     indicator=soil_data["DRC"],
    ...     func=SuitabilityFunction(name="discrete", params={"rules": {"1": 0, "2": 0.1, "3": 0.5, "4": 0.9, "5": 1}}),
    ... )

    Here is another example using the sample climate data with the growing degree days (GDD)
    as indicator for the criteria computing using the `xclim` package.

    >>> clim_data = load_climate_data()
    >>> gdd = growing_degree_days(clim_data["tas"], thresh="10 degC", freq="YS-JUL")
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
        func: SuitabilityFunction | Callable = None,
        weight: int | float | None = 1,
        category: str | None = None,
        long_name: str | None = None,
        description: str | None = None,
        comment: str | None = None,
        is_computed: bool = False,
    ) -> None:
        self.name = name
        self.indicator = indicator
        self.func = func
        self.weight = weight
        self.category = category
        self.long_name = long_name
        self.description = description
        self.comment = comment
        self.is_computed = is_computed
        self._from_indicator = _get_indicator_description(indicator)

    def __repr__(self) -> str:
        """Return a string representation for a particular criteria."""
        attrs = []
        attrs.append(f"name='{self.name}'")
        attrs.append(f"indicator={self.indicator.name}")
        attrs.append(f"func={self.func if self.func is not None else 'unknown'}")
        attrs.append(f"weight={self.weight}")
        if self.category is not None:
            attrs.append(f"category='{self.category}'")
        if self.long_name is not None:
            attrs.append(f"long_name='{self.long_name}'")
        if self.description is not None:
            attrs.append(f"description='{self.description}'")
        if self.comment is not None:
            attrs.append(f"comment='{self.comment}'")
        if self.is_computed:
            attrs.append("is_computed=True")
        return f"{self.__class__.__name__}({', '.join(attrs) if attrs else ''})"

    def compute(self) -> xr.DataArray:
        """
        Compute the suitability of the criteria.

        Returns a xarray DataArray with criteria suitability. The attributes of the DataArray describe how
        the suitability was computed.

        Returns
        -------
        xr.DataArray
            Criteria suitability.
        """
        if self.is_computed:
            sc = self.indicator
        elif self.func is None:
            raise ValueError("The suitability function is not defined. Please provide a valid function.")
        else:
            sc: xr.DataArray = xr.apply_ufunc(self.func, self.indicator)
        return sc.rename(self.name).assign_attrs(
            dict(
                {
                    k: v
                    for k, v in self.attrs.items()
                    if k not in ["name", "func_method", "from_indicator", "is_computed"]
                },
                **{
                    "history": f"func_method: {self.func if self.func is not None else 'unknown'}; "
                    f"from_indicator: [{self._from_indicator}]"
                },
            )
        )

    @property
    def attrs(self) -> dict:
        """
        Dictionary of the criteria attributes.

        Returns
        -------
        dict
            Dictionary with the criteria attributes.
        """
        return {
            k: v
            for k, v in {
                "name": self.name,
                "weight": self.weight,
                "category": self.category,
                "long_name": self.long_name,
                "description": self.description,
                "func_method": self.func if self.func is not None else "unknown",
                "from_indicator": self._from_indicator,
                "is_computed": self.is_computed,
            }.items()
            if v is not None
        }


def _get_indicator_description(indicator: xr.Dataset | xr.DataArray) -> str:
    if indicator.attrs != {}:
        return f"name: {indicator.name}; " + "; ".join([f"{k}: {v}" for k, v in indicator.attrs.items()])
    else:
        return f"name: {indicator.name}"
