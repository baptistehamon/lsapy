"""Suitability Criteria definition."""

import xarray as xr

from lsapy.functions import DiscreteSuitFunction, MembershipSuitFunction, SuitabilityFunction

__all__ = ["SuitabilityCriteria"]


class SuitabilityCriteria:
    def __init__(
            self,
            name: str,
            indicator: xr.Dataset | xr.DataArray,
            func: SuitabilityFunction | MembershipSuitFunction | DiscreteSuitFunction,
            weight: int | float | None = 1,
            category: str | None = None,
            long_name: str | None = None,
            description: str | None = None

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
        if self.func.func_method == 'discrete':  # need to vectorize the discrete function
            sc: xr.DataArray = xr.apply_ufunc(self.func.map, self.indicator).rename(self.name)
        else:
            sc: xr.DataArray = self.func.map(self.indicator).rename(self.name)
        return sc.assign_attrs(
            dict({k: v for k, v in self.attrs.items() if k not in ['name', 'func_method', 'from_indicator']},
                 **{
                     'history': f"func_method: {self.func}; from_indicator: [{self._from_indicator}]",
                     'compute': 'done'
                 })
        )

    @property
    def attrs(self):
        return {k: v for k, v in {
                    'name': self.name,
                    'weight': self.weight,
                    'category': self.category,
                    'long_name': self.long_name,
                    'description': self.description,
                    'func_method': self.func,
                    'from_indicator': self._from_indicator
                }.items() if v is not None}


def _get_indicator_description(indicator: xr.Dataset | xr.DataArray) -> str:
    if indicator.attrs != {}:
        return f"name: {indicator.name}; " + "; ".join([f"{k}: {v}" for k, v in indicator.attrs.items()])
    else:
        return f"name: {indicator.name}"
