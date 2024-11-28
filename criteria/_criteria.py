
from typing import Optional, Union, Any
import xarray as xr

from lsapy.functions import SuitabilityFunction, MembershipSuitFunction, DiscreteSuitFunction
from lsapy.criteria.indicators import CriteriaIndicator

__all__ = ["SuitabilityCriteria"]

class SuitabilityCriteria:
    def __init__(
            self,
            name: str,
            indicator: CriteriaIndicator,
            func: Union[SuitabilityFunction, MembershipSuitFunction, DiscreteSuitFunction],
            weight: Optional[float] = 1.0,
            category: Optional[str] = None,
            long_name: Optional[str] = 'Suitability',
            description: Optional[str] = None

    ) -> None:
        self.name = name
        self.indicator = indicator
        self.func = func
        self.weight = weight
        self.category = category
        self.long_name = long_name
        self.description = description
        self._func_method = func.attrs
    
    def __str__(self) -> str:
        return f"{self.name}"
    
    def compute(self, x: xr.DataArray | xr.Dataset) -> xr.DataArray:
        ci = self.indicator.compute(x)
        sc : xr.DataArray = xr.apply_ufunc(self.func.map, ci, vectorize=True, dask='parallelized')
        sc = sc.where(sc != 9999)
        self._ci_method = ci.attrs
        sc.attrs = self.attrs
        return sc
    
    @property
    def attrs(self):
        return {
            'name': self.name,
            'weight': self.weight,
            'category': self.category,
            'long_name': self.long_name,
            'description': self.description,
            'func_method': self._func_method,
            'ci_method': self._ci_method if hasattr(self, '_ci_method') else ''

        }
    
    @attrs.setter
    def attrs(self, value: dict[str, Any]):
        self.name = value.get('name', '')
        self._ci_method = value.get('ci_method', {})
        self._func_method = value.get('func_method', {})
        self.weight = value.get('weight', 1.0)
        self.category = value.get('category', None)
        self.long_name = value.get('long_name', 'Suitability')
        self.description = value.get('description', None)
