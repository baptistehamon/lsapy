"""Suitability Criteria definition."""

import xarray as xr

from lsapy.functions import DiscreteSuitFunction, MembershipFunction, SuitabilityFunction

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
        A description for the criteria. The default is None. # TODO: check default behavior.

    Examples
    --------
    Here is an example using the sample soil data with the drainage class (DRC) as indicator for the criteria.

    >>> from lsapy.utils import load_soil_data, load_climate_data
    >>> from xclim.indicators.atmos import growing_degree_days  # doctest: +SKIP
    <BLANKLINE>
    >>> soil_data = load_soil_data()
    >>> sc = SuitabilityCriteria(  # doctest: +SKIP
    ...     name="drainage_class",
    ...     long_name="Drainage Class Suitability",
    ...     weight=3,
    ...     category="soilTerrain",
    ...     indicator=soil_data["DRC"],
    ...     func=SuitabilityFunction(
    ...         func_method="discrete", func_params={"rules": {"1": 0, "2": 0.1, "3": 0.5, "4": 0.9, "5": 1}}
    ...     ),
    ... )

    Here is another example using the sample climate data with the growing degree days (GDD)
    as indicator for the criteria computing using the `xclim` package.

    >>> gdd = growing_degree_days(clim_data["tas"], thresh="10 degC", freq="YS-JUL")  # doctest: +SKIP
    >>> sc = SuitabilityCriteria( # doctest: +SKIP
    ...     name = "growing_degree_days"
    ...     long_name= "Growing Degree Days Suitability",
    ...     weight= 1,
    ...     category= "climate",
    ...     indicator=gdd,
    ...     func = SuitabilityFunction(func_method='vetharaniam2022_eq5', func_params={'a': -1.41, 'b': 801}))
    """

    def __init__(
        self,
        name: str,
        indicator: xr.Dataset | xr.DataArray,  # TODO: check if it's work with ds
        func: SuitabilityFunction | MembershipFunction | DiscreteSuitFunction,
        weight: int | float | None = 1,
        category: str | None = None,
        long_name: str | None = None,
        description: str | None = None,
    ) -> None:
        self.name = name
        self.indicator = indicator
        self.func = func
        self.weight = weight
        self.category = category
        self.long_name = long_name
        self.description = description
        self._from_indicator = _get_indicator_description(indicator)

    def __repr__(self) -> str:
        """Return a string representation for a particular criteria."""
        attrs = []
        attrs.append(f"name='{self.name}'")
        attrs.append(f"indicator={self.indicator.name}")
        attrs.append(f"func={self.func}")
        attrs.append(f"weight={self.weight}")
        if self.category is not None:
            attrs.append(f"category='{self.category}'")
        if self.long_name is not None:
            attrs.append(f"long_name='{self.long_name}'")
        if self.description is not None:
            attrs.append(f"description='{self.description}'")
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
        if self.func.func_method == "discrete":  # need to vectorize the discrete function
            sc: xr.DataArray = xr.apply_ufunc(self.func.map, self.indicator).rename(self.name)
        else:
            sc: xr.DataArray = self.func.map(self.indicator).rename(self.name)
        return sc.assign_attrs(
            dict(
                {k: v for k, v in self.attrs.items() if k not in ["name", "func_method", "from_indicator"]},
                **{"history": f"func_method: {self.func}; from_indicator: [{self._from_indicator}]", "compute": "done"},
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
                "func_method": self.func,
                "from_indicator": self._from_indicator,
            }.items()
            if v is not None
        }


def _get_indicator_description(indicator: xr.Dataset | xr.DataArray) -> str:
    if indicator.attrs != {}:
        return f"name: {indicator.name}; " + "; ".join([f"{k}: {v}" for k, v in indicator.attrs.items()])
    else:
        return f"name: {indicator.name}"
