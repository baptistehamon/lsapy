
from typing import Optional, Any
import xarray as xr

from landsuitability.criteria import SuitabilityCriteria

__all__ = ["LandSuitability"]


class LandSuitability:
    
    def __init__(
            self,
            name: str,
            data: xr.Dataset,
            criteria: dict[str, SuitabilityCriteria],
            short_name: Optional[str] = None,
            long_name: Optional[str] = None,
            description: Optional[str] = None,
    ) -> None:
        self.name = name
        self.data = data
        self.criteria = criteria
        self.short_name = short_name
        self.long_name = long_name
        self.description = description
        self._criteria_name_list = list(criteria.keys())
        self._category_list = list(set([sc.category for sc in criteria.values()]))

        self._get_params_by_category()


    def __str__(self) -> str:
        return f"{self.name}"
    

    # def __getitem__(self, key: str) -> SuitabilityCriteria:
    #     return self.criteria[key]


    def compute(self):
        return self.compute_criteria_suitability()


    def compute_criteria_suitability(self, inplace: Optional[bool] = False) -> xr.Dataset:
        sc_list = []
        for sc_name, sc in self.criteria.items():
            print(f'Computing {sc_name}...')
            res = sc.compute(self.data)
            res.name = sc_name
            sc_list.append(res)
        ls = xr.merge(sc_list, compat='override')
        ls.attrs = {'criteria': self._criteria_name_list, 'categories': self._category_list}
        if inplace:
            self.suitability = ls
        else:
            return ls
    

    def compute_category_suitability(
            self, mean_method: str = 'weighted_mean',
            inplace: Optional[bool] = False,
            keep_criteria: Optional[bool] = False
    ) -> xr.Dataset:
        if not hasattr(self, 'suitability'):
            self.compute_criteria_suitability(inplace=True)
        

        cat_list = []
        for category in self._category_list:
            print(f'Computing {category}...')
            sc_list = [sc for sc in self.criteria.values() if sc.category == category]
            res = sum([sc.compute(self.data) * sc.weight for sc in sc_list]) / self._category_weights[category]
            res.name = category
            cat_list.append(res)
        ls = xr.merge(cat_list, compat='override')
        ls.attrs = {'criteria': self._criteria_name_list, 'categories': self._category_list}
        return ls
    

    def compute_suitability(self) -> xr.Dataset:
        ls = self.compute_category_suitability()
        total_weight = sum(self._category_weights.values())
        ls = sum([ls[category] * self._category_weights[category] for category in self._category_list]) / total_weight
        return ls
    

    def _get_params_by_category(self):
        self._get_criteria_by_category()
        self._get_weight_by_category()
    
    
    def _get_criteria_by_category(self) -> dict[str, list[str]]:
        self._category_criteria = {category: [] for category in self._category_list}
        for sc_name, sc in self.criteria.items():
            self._category_criteria[sc.category].append(sc_name)
    

    def _get_weight_by_category(self) -> dict[str, float | int]:
        self._category_weights = {category: [] for category in self._category_list}
        for category in self._category_list:
            self._category_weights[category] = sum([sc.weight for sc in self.criteria.values() if sc.category == category])

