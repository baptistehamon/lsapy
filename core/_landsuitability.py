
from typing import Optional, Any, Union
import xarray as xr
import numpy as np

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


    def compute_criteria_suitability(self, inplace: Optional[bool] = False) -> None | xr.Dataset:
        sc_list = []
        for sc_name, sc in self.criteria.items():
            print(f'Computing {sc_name}...')
            res = sc.compute(self.data)
            res.name = sc_name
            sc_list.append(res)
        ls = xr.merge(sc_list, compat='override')
        ls.attrs = {'criteria': self._criteria_name_list, 'categories': self._category_list, 'compute': 'criteria'}
        if inplace:
            self.suitability = ls
        else:
            return ls
    

    def compute_category_suitability(
            self, method: Union[str, list[str]] = 'weighted_geomean',
            keep_criteria: Optional[bool] = False,
            inplace: Optional[bool] = False,
            limit_var: Optional[bool] = False,
    ) -> xr.Dataset:
        if not hasattr(self, 'suitability') or self.suitability.attrs.get('compute', '') != 'criteria':
            ci_ds = self.compute_criteria_suitability()
        else:
            ci_ds = self.suitability
        
        # if isinstance(method, str):
        #     method = [method]
        
        cat_list = []
        for category in self._category_list:
            print(f'Computing {category}...')
            sc_list = [sc for sc in self.criteria.values() if sc.category == category]
            res = _compute_vars_suitability(ci_ds[self._category_criteria[category]], method=method,
                                            weights=[sc.weight for sc in sc_list], limit_var=limit_var)
            res = res.rename({'Suitability': f'{category}'})
            if method == 'limit_factor':
                res = res.rename({'limiting_var': f'{category}_limiting_var'})
            cat_list.append(res)
        ls = xr.merge(cat_list, compat='override')
        ls.attrs = {'criteria': self._criteria_name_list, 'categories': self._category_list, 'compute': 'category'}
        if keep_criteria:
            ls = xr.merge([ci_ds, ls], compat='override')
        if inplace:
            self.suitability = ls
        else:
            return ls
    

    def compute_suitability(self, method: str = 'weighted_mean',
                            keep_category: Optional[bool] = False,
                            inplace: Optional[bool] = False,
                            limit_var: Optional[bool] = False
    ) -> xr.Dataset:
        if not hasattr(self, 'suitability') or self.suitability.attrs.get('compute', '') != 'category':
            raise ValueError("Category suitability must be computed first.")
        else:
            ls = self.suitability
        
        cat_weights = [self._category_weights[category] for category in self._category_list]
        ls = _compute_vars_suitability(ls, method=method, vars=self._category_list, weights=cat_weights, limit_var=limit_var)
        ls.attrs = {'criteria': self._criteria_name_list, 'categories': self._category_list, 'compute': 'suitability'}
        if keep_category:
            ls = xr.merge([self.suitability, ls], compat='override')
        if inplace:
            self.suitability = ls
        else:
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


# Utility functions

def _vars_weighted_mean(ds: xr.Dataset, vars = None, weights = None) -> xr.DataArray:
    if vars is None:
        vars = list(ds.data_vars)
    if weights is None:
        weights = np.ones(len(vars))
    
    s = sum([ds[v] * w for v, w in zip(vars, weights)])
    da: xr.DataArray = s / sum(weights)
    return da.assign_attrs({
        'method': 'Weighted Mean', 
        'description': f"Weighted Mean of variables: {', '.join([f'{v} ({w})' for v, w in zip(vars, weights)])}."
    })

def _vars_mean(ds: xr.Dataset, vars = None) -> xr.DataArray:
    if vars is None: vars = list(ds.data_vars)
    da = _vars_weighted_mean(ds, vars=vars)
    return da.assign_attrs({'method': 'Mean', 'description': f"Mean of variables: {', '.join(vars)}."})

def _vars_weighted_geomean(ds: xr.Dataset, vars = None, weights = None) -> xr.DataArray:
    if vars is None:
        vars = list(ds.data_vars)
    if weights is None:
        weights = np.ones(len(vars))

    s = sum([np.log(ds[v]) * w for v, w in zip(vars, weights)])
    da : xr.DataArray = np.exp(s / sum(weights))
    return da.assign_attrs({
        'method': 'Weighted Geometric Mean',
        'description': f"Weighted Geometric Mean of variables: {', '.join([f'{v} ({w})' for v, w in zip(vars, weights)])}."
    })

def _vars_geomean(ds: xr.Dataset, vars = None) -> xr.DataArray:
    if vars is None: vars = list(ds.data_vars)
    da = _vars_weighted_geomean(ds, vars=vars)
    return da.assign_attrs({'method': 'Geometric Mean', 'description': f"Geometric Mean of variables: {', '.join(vars)}."})

def _limiting_vars(ds: xr.Dataset, vars = None, limiting_var: Optional[bool] = True) -> xr.Dataset:
    if vars is None:
        vars = list(ds.data_vars)
    
    da = ds[vars].to_array()
    mask = da.notnull().all(dim='variable')

    lim = da.min(dim='variable', skipna=True).where(mask)
    lim.name = 'Suitability'
    lim = lim.assign_attrs({'method': 'Limiting Factor', 'description': f"Value of limiting factor among variables: {', '.join(vars)}."})
    if limiting_var:
        lim_var = da.fillna(2).argmin(dim='variable', skipna=True).where(mask)
        lim_var.attrs = {'method': 'Limiting Factor',
                         'description': f"Limiting factor among: {', '.join(vars)}.",
                         'legend': {f'{i}': v for i, v in enumerate(vars)}}
        lim_var.name = 'limiting_var'
        return xr.merge([lim, lim_var], compat='override')
    return lim.to_dataset()

def _compute_vars_suitability(ds : xr.Dataset, method: str = 'mean', vars = None, weights = None, limit_var = True) -> xr.Dataset:
    if method.lower() == 'mean':
        return _vars_mean(ds, vars=vars).to_dataset(name='Suitability')
    elif method.lower() == 'weighted_mean':
        return _vars_weighted_mean(ds, vars=vars, weights=weights).to_dataset(name='Suitability')
    elif method.lower() == 'geomean':
        return _vars_geomean(ds, vars=vars).to_dataset(name='Suitability')
    elif method.lower() == 'weighted_geomean':
        return _vars_weighted_geomean(ds, vars=vars, weights=weights).to_dataset(name='Suitability')
    elif method.lower() == 'limit_factor':
        return _limiting_vars(ds, vars=vars, limiting_var=limit_var)
    else:
        raise ValueError(f"Method '{method}' not recognized.")
