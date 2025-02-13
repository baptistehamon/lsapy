
from typing import Optional, Any, Union
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping
import rioxarray
import regionmask

from lsapy.criteria import SuitabilityCriteria

__all__ = ["LandSuitability"]


class LandSuitability:
    
    def __init__(
            self,
            name: str,
            criteria: dict[str, SuitabilityCriteria],
            short_name: Optional[str] = None,
            long_name: Optional[str] = None,
            description: Optional[str] = None,
    ) -> None:
        self.name = name
        self.criteria = criteria
        self.short_name = short_name
        self.long_name = long_name
        self.description = description

        self._sort_criteria_by_weight()
        self._criteria_name_list = [sc.name for sc in self.criteria.values()]
        self._category_list = list(set([sc.category for sc in self.criteria.values()]))
        self._get_params_by_category()


    def __repr__(self) -> str:
        if hasattr(self, 'data'):
            return self.data.__repr__()
        else:
            attrs = []
            for k, v in self.attrs.items():
                if isinstance(v, str):
                    v = f"'{v}'"
                attrs.append(f"{k}={v}")
            return f"{self.__class__.__name__}({', '.join(attrs) if attrs else ''})"
    

    @property
    def attrs(self):
        return {k: v for k, v in {
                    'name': self.name,
                    'criteria': self._criteria_name_list,
                    'short_name': self.short_name,
                    'long_name': self.long_name,
                    'description': self.description,
                }.items() if v is not None}


    # def __getitem__(self, key: str) -> SuitabilityCriteria:
    #     return self.criteria[key]


    def compute_criteria_suitability(self, inplace: Optional[bool] = False) -> None | xr.Dataset:
        sc_list = []
        for sc_name, sc in self.criteria.items():
            print(f'Computing {sc_name}...')
            if sc.indicator.attrs.get('compute', '') == 'done':
                sc_list.append(sc.indicator.rename(sc.name))
            else:
                sc_list.append(sc.compute())
        ls = xr.merge(sc_list, compat='override', combine_attrs='drop')
        for sc in sc_list:
            ls[sc.name].attrs = sc.attrs
        ls.attrs = self.attrs
        if inplace:
            self.data = ls
        else:
            return ls
    

    def compute_category_suitability(
            self, method: str,
            keep_criteria: Optional[bool] = False,
            inplace: Optional[bool] = False,
            limit_var: Optional[bool] = False,
    ) -> xr.Dataset:
        if not hasattr(self, 'data'):
            ds = self.compute_criteria_suitability()
        else:
            ds = self.data
        
        out = []
        out_attrs = []
        for category in self._category_list:
            print(f'Computing {category}...')
            sc_list = [sc for sc in self.criteria.values() if sc.category == category]
            res = _aggregate_vars(ds[self._criteria_by_category[category]], method=method,
                                  weights=[sc.weight for sc in sc_list], limit_var=limit_var)
            if isinstance(res, xr.Dataset):
                res = res.rename({'limiting_factor': f'{category}', 'limiting_var': f'{category}_var'})
            else:
                res = res.rename(f'{category}')
            res.attrs.update({'long_name': f'{category} Suitability'})
            out.append(res)
            out_attrs.append(res.attrs)
        out = xr.merge(out, compat='override', combine_attrs='drop')
        if keep_criteria:
            out = xr.merge([ds, out], compat='override', combine_attrs='drop')
            for sc in ds.data_vars: # add attributes of criteria
                out[sc].attrs = ds[sc].attrs
        for i, category in enumerate(self._category_list): # add attributes of category suitability
            out[f'{category}'].attrs = out_attrs[i]
        
        out.attrs.update(self.attrs)
        if inplace:
            self.data = out
        else:
            return out
    

    def compute_suitability(self,
                            method: Union[str, dict[str,str]] = 'mean',
                            by_category: Optional[bool] = False,
                            keep_all: Optional[bool] = False,
                            inplace: Optional[bool] = False,
                            limit_var: Optional[bool] = False
    ) -> xr.Dataset:
        if isinstance(method, str):
            cat_method, suit_method = method, method
        elif isinstance(method, dict):
            cat_method = method.get('category', 'mean')
            suit_method = method.get('overall', 'mean')
        else:
            raise ValueError("Method must be a string or a dictionary.")
        
        if not hasattr(self, 'data'):
            if by_category:
                ds = self.compute_category_suitability(method=cat_method, keep_criteria=True, inplace=False, limit_var=limit_var)
            else:
                ds = self.compute_criteria_suitability(inplace=False)
        else:
            ds = self.data
        
        if by_category:
            weights = [self.weights_by_category[category] for category in self._category_list]
            on_vars = [f'{category}' for category in self._category_list]
        else :
            weights = [sc.weight for sc in self.criteria.values()]
            on_vars = self._criteria_name_list
        
        print(f'Computing suitability...')
        out = _aggregate_vars(ds[on_vars], method=suit_method, weights=weights, limit_var=limit_var).rename('suitability')
        out.attrs.update({'long_name': 'Suitability'})
        out_attrs = out.attrs

        if keep_all:
            out = xr.merge([ds, out], compat='override', combine_attrs='drop')
            for sc in ds.data_vars: # add attributes of criteria
                out[sc].attrs = ds[sc].attrs
            out['suitability'].attrs = out_attrs
        
        out.attrs.update(self.attrs)
        if inplace:
            self.data = out
        else:
            return out
    

    def _sort_criteria_by_weight(self) -> dict[str, SuitabilityCriteria]:
        self.criteria = dict(sorted(self.criteria.items(), key=lambda item: item[1].weight, reverse=True))


    def _get_params_by_category(self):
        self._get_criteria_by_category()
        self._get_weights_by_category()
    
    
    def _get_criteria_by_category(self) -> dict[str, list[str]]:
        self._criteria_by_category = {category: [] for category in self._category_list}
        for sc in self.criteria.values():
            self._criteria_by_category[sc.category].append(sc.name)
    

    def _get_weights_by_category(self) -> dict[str, float | int]:
        self.weights_by_category = {category: [] for category in self._category_list}
        for category in self._category_list:
            self.weights_by_category[category] = sum([sc.weight for sc in self.criteria.values() if sc.category == category])

    
    def mask(self, mask : Union[xr.DataArray, gpd.GeoDataFrame], inplace : Optional[bool] = False,
          spatial_dims : Optional[tuple[str, str]] = None, crs : Optional[str] = None, invert : bool = False, **kwargs) -> xr.DataArray | xr.Dataset:
        
        if not hasattr(self, 'data'):
            raise ValueError("Suitability must be computed first.")
        
        if inplace:
            self.data = _mask_data(self.data, mask, spatial_dims=spatial_dims, crs=crs, invert=invert, **kwargs)
        else:     
            return _mask_data(self.data, mask, spatial_dims=spatial_dims, crs=crs, invert=invert, **kwargs)
    

    def statistics(self,
                   on_vars: Optional[list] = None,
                   on_dims: Optional[list] = None,
                   on_dim_values: Optional[dict[str, Any]] = None,
                   bins : Optional[np.ndarray] = None,
                   all_bins : Optional[bool] = False,
                   cell_area : Optional[tuple[Union[float, str], str]] = None,
                   **kwargs) -> pd.DataFrame:
        
        if not hasattr(self, 'data'):
            raise ValueError("Suitability must be computed first.")
        
        return _statistics(self.data, on_vars=on_vars, on_dims=on_dims, on_dims_values=on_dim_values, bins=bins, all_bins=all_bins, cell_area=cell_area, **kwargs)
    

    def spatial_statistics(
            self,
            areas : gpd.GeoDataFrame,
            name : Optional[str] = 'area',
            on_vars: Optional[list] = None,
            on_dims: Optional[list] = None,
            on_dim_values: Optional[dict[str, Any]] = None,
            bins : Optional[np.ndarray] = None,
            all_bins : Optional[bool] = False,
            cell_area : Optional[tuple[Union[float, str], str]] = None,
            mask_kwargs : dict = {},
            stats_kwargs: dict = {}) -> pd.DataFrame:
        
        if not hasattr(self, 'data'):
            raise ValueError("Suitability must be computed first.")
        
        return _spatial_statistics(self.data, areas, name=name, on_vars=on_vars, on_dims=on_dims, on_dim_values=on_dim_values,
                                   bins=bins, all_bins=all_bins, cell_area=cell_area, mask_kwargs=mask_kwargs, stats_kwargs=stats_kwargs)

        
        
        

    # def _write_to_netcdf(self, path: str, vars: Optional[Union[str, list[str]]] = None) -> None:
    #     if not hasattr(self, 'data'):
    #         raise ValueError("Suitability must be computed first.")

    #     if vars is None:
    #         vars = list(self.suitability.data_vars)
        
    #     self.suitability[vars].to_netcdf(path)

    
    # def _write_to_geotiff(self, path: str, vars: Optional[Union[str, list[str]]] = None) -> None:
    #     pass
        # if not hasattr(self, 'suitability'):
        #     raise ValueError("Suitability must be computed first.")

        # if vars is None:
        #     vars = list(self.suitability.data_vars)
        
        # for var in vars:
        #     pass
        #     self.suitability[var].rio.to_raster(path, driver='GTiff')


def _mask_data(data : xr.DataArray | xr.Dataset, mask : Union[xr.DataArray, gpd.GeoDataFrame],
          spatial_dims : Optional[tuple[str, str]] = None,
          crs : Optional[str] = None, invert : bool = False, **kwargs) -> xr.DataArray | xr.Dataset:
    
    if isinstance(mask, gpd.GeoDataFrame):
        mask = mask.to_crs(crs)
        data = data.rio.set_spatial_dims(*spatial_dims).rio.write_crs(crs)
        return data.rio.clip(mask.geometry.apply(mapping), invert=invert, **kwargs)
    elif isinstance(mask, xr.DataArray):
        if invert:
            mask = ~mask
        return data.where(mask, **kwargs)
    else:
        raise ValueError('mask must be a GeoDataFrame or DataArray')
    

def _select_dims_values(ds: xr.Dataset, on_dim_values: dict[str, Any]) -> xr.Dataset:
    for dim, value in on_dim_values.items():
        ds = ds.sel({dim: value})
    return ds


def _statistics(
        data: xr.DataArray | xr.Dataset,
        on_vars: Optional[list] = None,
        on_dims: Optional[list] = None,
        on_dim_values: Optional[dict[str, Any]] = None,
        bins : Optional[np.ndarray] = None,
        all_bins : Optional[bool] = False,
        cell_area : Optional[tuple[Union[float, str], str]] = None,
        **kwargs) -> pd.DataFrame:
    
    if on_dim_values is not None:
        data = _select_dims_values(data, on_dim_values)
    
    if on_vars is None:
        on_vars = list(data.data_vars)
    data = data[on_vars]
    if on_dims is None:
        on_dims = list(data.dims)
        on_dims = [d for d in on_dims if d not in ['lat', 'lon', 'x', 'y']]
    if cell_area:
        cell_area, cell_unit = cell_area

    df = data.to_dataframe().reset_index().drop(columns=[c for c in data.coords if c not in on_dims])
    if len(on_dims) > 0:
        df = df.melt(id_vars=on_dims)
    else:
        df = df.melt()
    _dims = ['variable'] + on_dims

    if bins is not None:
        df['bins'] = pd.cut(df['value'], bins=bins, **kwargs)
        _dims.append('bins')
        if all_bins:
            df = pd.concat([df, df.drop(columns=['bins']).assign(bins='all')])

    df_out = df.groupby(_dims, observed=False).describe().droplevel(0, axis=1).reset_index()
    if cell_area:
        df_out[f'area_{cell_unit}'] = df_out['count'] * cell_area

    return df_out.dropna()


def _spatial_statistics(
        data: xr.DataArray | xr.Dataset,
        areas : gpd.GeoDataFrame,
        name : Optional[str] = 'area',
        on_vars: Optional[list] = None,
        on_dims: Optional[list] = None,
        on_dim_values: Optional[dict[str, Any]] = None,
        bins : Optional[np.ndarray] = None,
        all_bins : Optional[bool] = False,
        cell_area : Optional[tuple[Union[float, str], str]] = None,
        mask_kwargs : dict = {},
        stats_kwargs: dict = {}) -> pd.DataFrame:
    
    regions = regionmask.from_geopandas(areas, name=name, **mask_kwargs)
    mask = regions.mask_3D(data)

    out = []
    for r in mask.region.values:
        df = _statistics(data, on_vars=on_vars, on_dims=on_dims, on_dim_values=on_dim_values, bins=bins, all_bins=all_bins, cell_area=cell_area, **stats_kwargs)
        df.insert(0, name, regions[r].name)
        out.append(df)
    return pd.concat(out)

####################################################################################################
# VARIABLES AGGREGATION FUNCTIONS
####################################################################################################

def vars_weighted_mean(ds: xr.Dataset, vars = None, weights = None) -> xr.DataArray:
    if vars is None:
        vars = list(ds.data_vars)
    if weights is None:
        weights = np.ones(len(vars))
    
    s = sum([ds[v] * w for v, w in zip(vars, weights)])
    da: xr.DataArray = s / sum(weights)
    return da.assign_attrs({
        'method': 'Weighted Mean', 
        'description': f"Weighted Mean of variables: {', '.join([f'{v} ({w})' for v, w in zip(vars, weights)])}."
    }).rename('weighted_mean')

def vars_mean(ds: xr.Dataset, vars = None) -> xr.DataArray:
    if vars is None: vars = list(ds.data_vars)
    da = vars_weighted_mean(ds, vars=vars)
    return da.assign_attrs({'method': 'Mean', 'description': f"Mean of variables: {', '.join(vars)}."}).rename('mean')

def vars_weighted_geomean(ds: xr.Dataset, vars = None, weights = None) -> xr.DataArray:
    if vars is None:
        vars = list(ds.data_vars)
    if weights is None:
        weights = np.ones(len(vars))

    s = sum([np.log(ds[v]) * w for v, w in zip(vars, weights)])
    da : xr.DataArray = np.exp(s / sum(weights))
    return da.assign_attrs({
        'method': 'Weighted Geometric Mean',
        'description': f"Weighted Geometric Mean of variables: {', '.join([f'{v} ({w})' for v, w in zip(vars, weights)])}."
    }).rename('weighted_geometric_mean')

def vars_geomean(ds: xr.Dataset, vars = None) -> xr.DataArray:
    if vars is None: vars = list(ds.data_vars)
    da = vars_weighted_geomean(ds, vars=vars)
    return da.assign_attrs({'method': 'Geometric Mean', 'description': f"Geometric Mean of variables: {', '.join(vars)}."}).rename('geometric_mean')

def limiting_factor(ds: xr.Dataset, vars = None, limiting_var: Optional[bool] = True) -> xr.DataArray | xr.Dataset:
    if vars is None:
        vars = list(ds.data_vars)
    
    da = ds[vars].to_array()
    mask = da.notnull().all(dim='variable')

    lim = da.min(dim='variable', skipna=True).where(mask).rename('limiting_factor')
    lim = lim.assign_attrs({'method': 'Limiting Factor', 'description': f"Value of limiting factor among variables: {', '.join(vars)}."})
    if limiting_var:
        lim_var = da.fillna(9999).argmin(dim='variable', skipna=True).where(mask).rename('limiting_var')
        lim_var.attrs = {'method': 'Limiting Factor',
                         'description': f"Limiting factor among: {', '.join(vars)}.",
                         'legend': {f'{i}': v for i, v in enumerate(vars)}}
        return xr.merge([lim, lim_var])
    return lim.to_dataset()

def _aggregate_vars(ds : xr.Dataset, method: str = 'mean', vars = None, weights = None, **kwargs) -> xr.DataArray | xr.Dataset:
    if method.lower() == 'mean':
        return vars_mean(ds, vars=vars)
    elif method.lower() == 'weighted_mean':
        return vars_weighted_mean(ds, vars=vars, weights=weights)
    elif method.lower() == 'geomean':
        return vars_geomean(ds, vars=vars)
    elif method.lower() == 'weighted_geomean':
        return vars_weighted_geomean(ds, vars=vars, weights=weights)
    elif method.lower() == 'limit_factor':
        return limiting_factor(ds, vars=vars, **kwargs)
    else:
        raise ValueError(f"Method '{method}' not recognized.")
