from typing import Any, Optional, Union
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import regionmask


def statistics_summary(
        data: xr.DataArray | xr.Dataset,
        on_vars: Optional[list] = None,
        on_dims: Optional[list] = None,
        on_dim_values: Optional[dict[str, Any]] = None,
        bins : Optional[Union[list, np.ndarray]] = None,
        bins_labels : Optional[list] = None,
        all_bins : Optional[bool] = False,
        cell_area : Optional[tuple[Union[float, str], str]] = None,
        dropna: Optional[bool] = False,
        **kwargs) -> pd.DataFrame:
    
    def _correct_lowest_cut_interal(x: pd.Series) -> pd.Series:
            first_cat = x.cat.categories[0]
            lf = first_cat.left + (abs(first_cat.left) - bins[0])
            return x.cat.rename_categories({first_cat: pd.Interval(lf, first_cat.right, closed='both')})
    
    
    if bins is not None and bins_labels is not None and (len(bins)-1 != len(bins_labels)):
        raise ValueError('bins and bins_labels must have the same length')
    
    if on_dim_values is not None:
        for dim, value in on_dim_values.items():
            data = data.sel({dim: value})
    
    if on_vars is None:
        on_vars = list(data.data_vars)
    data = data[on_vars]
    if on_dims is None:
        on_dims = list(data.dims)
        on_dims = [d for d in on_dims if d not in ['lat', 'lon', 'x', 'y']] # remove spatial dims
    if cell_area:
        cell_area, cell_unit = cell_area


    df = data.to_dataframe().reset_index()
    if len(on_dims) > 0:
        df = df.drop(columns=[c for c in data.coords if c not in on_dims])
    
    df = df.melt(id_vars=on_dims)
    _dims = ['variable'] + on_dims

    if bins is not None:
        df['bin'] = pd.cut(df['value'], bins=pd.Index(bins), **kwargs)
        if 'include_lowest' in kwargs and kwargs['include_lowest'] == True:
            df['bin'] = _correct_lowest_cut_interal(df['bin'])

        if bins_labels is not None:
            lab_mapping = dict(zip(df['bin'].cat.categories.astype(str), bins_labels))
        _dims.append('bin')
        if all_bins:
            all_bins_inter = pd.Interval(df['bin'].cat.categories[0].left, df['bin'].cat.categories[-1].right, closed='both')
            df_ = df.drop(columns=['bin']).assign(bin=all_bins_inter)
            df_.loc[df['value'].isnull(), 'bin'] = np.nan
            if bins_labels is not None:
                lab_mapping.update({str(all_bins_inter) : 'all'})
            df = pd.concat([df, df_])
        df['bin'] = df['bin'].astype(str)

    df_out = df.groupby(_dims, observed=False).describe().droplevel(0, axis=1).reset_index()

    if bins_labels is not None:
        bin_idx = np.where(df_out.columns == 'bin')[0][0]
        df_out.insert(bin_idx + 1, 'bin_label', df_out['bin'].map(lab_mapping).values)

    if cell_area:
        df_out[f'area_{cell_unit}'] = df_out['count'] * cell_area
    
    if dropna:
        return df_out.dropna()
    return df_out


def spatial_statistics_summary(
        data: xr.DataArray | xr.Dataset,
        areas : gpd.GeoDataFrame,
        name : Optional[str] = 'area',
        on_vars: Optional[list] = None,
        on_dims: Optional[list] = None,
        on_dim_values: Optional[dict[str, Any]] = None,
        bins : Optional[np.ndarray] = None,
        bins_labels : Optional[list] = None,
        all_bins : Optional[bool] = False,
        cell_area : Optional[tuple[Union[float, str], str]] = None,
        dropna: Optional[bool] = False,
        mask_kwargs : dict = {},
        stats_kwargs: dict = {}) -> pd.DataFrame:
    
    regions = regionmask.from_geopandas(areas, name=name, **mask_kwargs)
    mask = regions.mask_3D(data)

    out = []
    for r in mask.region.values:
        df = statistics_summary(
            data.where(mask.sel(region=r)), on_vars=on_vars, on_dims=on_dims, on_dim_values=on_dim_values,
            bins=bins, bins_labels=bins_labels, all_bins=all_bins,
            cell_area=cell_area ,dropna=dropna, **stats_kwargs
        )
        df.insert(0, name, regions[r].name)
        out.append(df)
    return pd.concat(out)