"""Statistics Module."""

from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import regionmask
import xarray as xr


def stats_summary(
    data: xr.DataArray | xr.Dataset,
    on_vars: list | None = None,
    on_dims: list | None = None,
    on_dim_values: dict[str, Any] | None = None,
    bins: list | np.ndarray | None = None,
    bins_labels: list | None = None,
    all_bins: bool | None = False,
    cell_area: tuple[float | int, str] | None = None,
    dropna: bool | None = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Generate a descriptive statistics summary of the data.

    Returns a pandas DataFrame of data according to the given parameters.
    The statistics includes count, mean, std, min, max, and 25%, 50%, and 75% percentiles.
    Bins can be provided to further group the data into intervals.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        The input data.
    on_vars : list, optional
        Variables for which the statistics are calculated. If None (default), all variables are kept.
    on_dims : list, optional
        Dimensions for which the statistics are calculated. If None (default), all dimensions except
        spatial ones (i.e., `lon` or `x` and `lat` or `y`) are kept.
    on_dim_values : sequence, optional
        Values of dimensions to be kept in the summary. If None (default), all values are kept.
    bins : list or np.ndarray, optional
        Bins defining data intervals. If None (default), no binning is performed.
    bins_labels : list, optional
        Labels for the bins. If None (default), bins values are used as labels.
        The length of the list must be equal to the number of bins. Ignored if `bins` is None.
    all_bins : bool, optional
        If True, a additional bin corresponding to the bounds of `bins` is added. Default is False.
        Ignored if `bins` is None.
    cell_area : tuple of float or int and str, optional
        Add a column to the summary with the given associated area calculated based on the count statistic
        variable. The tuple must contain the area value and the unit of the area.
    dropna : bool, optional
        If True, dimensions with NaN values are removed. Default is False.
    **kwargs : dict, optional
        Additional keyword arguments passed to `pd.cut` used to bin the data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the statistics for the defined dimensions and variables, including:
        count, mean, std, min, max, and 25%, 50%, and 75% percentiles.

    Examples
    --------
    >>> # define your LandSuitabilityAnalysis
    >>> lsa.run(inplace=True)
    >>> # perform simple statistics summary
    >>> stats_summary(lsa.data)

    `on_vars`, `on_dims`, and `on_dim_values` parameters can be used to filter the data. Assuming
    that `lsa.data` contains as the LSA criteria suitability and the overall suitability as variables (
    `criteria1`, `criteria2`, and `suitability`), and as dimensions the time for 10 years (2000-2009).
    If we want to get the statistics summary for criteria1 and criteria2, and only for the five first years,
    we can do:

    >>> stats_summary(
    ...     lsa.data,
    ...     on_vars=["criteria1", "criteria2"],  # select variables
    ...     on_dim_values={"time": slice(2000, 2004)},  # select values of the time dimension
    ... )

    This will compute the statistics for the two criteria and for the 2000-2004 period (i.e., one value for
    the 5 years). However, if we want to get the statistics for each year, we can specify to keep the time
    dimension:

    >>> stats_summary(
    ...     lsa.data,
    ...     on_vars=["criteria1", "criteria2"],
    ...     on_dims=["time"],  # keep the time dimension
    ...     on_dim_values={"time": slice(2000, 2004)},
    ... )

    We can also provide bins to group the data into intervals. For example, if we want to get the statistics
    for four bins (0-0.25, 0.25-0.5, 0.5-0.75, 0.75-1), we can do:

    >>> stats_summary(
    ...     lsa.data,
    ...     bins=[0, 0.25, 0.5, 0.75, 1], # define bins
    ...     bins_labels=['unsuitable', 'poorly suitable', 'moderately suitable', 'highly suitable'] # define labels
    ...     all_bins=True # add an additional bin for the overall range (i.e., 0-1)
    ... )

    Finally, we can get the area associated with each bin by providing the area of each cell in the data.
    Assuming that each cell has an area of 5 hectares (ha), we can do:
    >>> stats_summary(
    ...     lsa.data,
    ...     bins=[0, 0.25, 0.5, 0.75, 1],
    ...     bins_labels=["unsuitable", "poorly suitable", "moderately suitable", "highly suitable"],
    ...     all_bins=True,
    ...     cell_area=(5, "ha"),  # define the area of each cell
    ... )
    """

    def _close_lowest_bin(x: pd.Series, bins) -> pd.Series:
        first_cat = x.cat.categories[0]
        lf = first_cat.left + (abs(first_cat.left) - bins[0])
        return x.cat.rename_categories({first_cat: pd.Interval(lf, first_cat.right, closed="both")})

    if bins is not None and bins_labels is not None and (len(bins) - 1 != len(bins_labels)):
        raise ValueError("bins and bins_labels must have the same length")

    if on_dim_values is not None:
        for dim, value in on_dim_values.items():
            data = data.sel({dim: value})

    if on_vars is None:
        on_vars = list(data.data_vars)
    data = data[on_vars]
    if on_dims is None:
        on_dims = [d for d in data.dims if d not in ["lat", "lon", "x", "y"]]  # remove spatial dims
    if cell_area:
        cell_value, cell_unit = cell_area

    df = data.to_dataframe().reset_index()
    if len(on_dims) > 0:
        df = df.drop(columns=[c for c in data.coords if c not in on_dims])

    df = df.melt(id_vars=on_dims)
    _dims = ["variable"] + on_dims

    if bins is not None:
        df["bin"] = pd.cut(df["value"], bins=pd.Index(bins), **kwargs)
        if "include_lowest" in kwargs and kwargs["include_lowest"]:
            df["bin"] = _close_lowest_bin(df["bin"], bins)

        if bins_labels is not None:
            lab_mapping = dict(zip(df["bin"].cat.categories.astype(str), bins_labels, strict=False))
        _dims.append("bin")
        if all_bins:
            all_bins_inter = pd.Interval(
                df["bin"].cat.categories[0].left, df["bin"].cat.categories[-1].right, closed="both"
            )
            df_ = df.drop(columns=["bin"]).assign(bin=all_bins_inter)
            df_.loc[df["value"].isnull(), "bin"] = np.nan
            if bins_labels is not None:
                lab_mapping.update({str(all_bins_inter): "bins_range"})
            df = pd.concat([df, df_])
        df["bin"] = df["bin"].astype(str)

    df_out = df.groupby(_dims, observed=False).describe().droplevel(0, axis=1).reset_index()

    if bins_labels is not None:
        bin_idx = np.where(df_out.columns == "bin")[0][0]
        df_out.insert(bin_idx + 1, "bin_label", df_out["bin"].map(lab_mapping).values)

    if cell_area:
        df_out[f"area_{cell_unit}"] = df_out["count"] * cell_value

    if dropna:
        return df_out.dropna()
    return df_out


def spatial_statistics_summary(
    data: xr.DataArray | xr.Dataset,
    areas: gpd.GeoDataFrame,
    name: str = "area",
    on_vars: list | None = None,
    on_dims: list | None = None,
    on_dim_values: dict[str, Any] | None = None,
    bins: np.ndarray | None = None,
    bins_labels: list | None = None,
    all_bins: bool | None = False,
    cell_area: tuple[float | int, str] | None = None,
    dropna: bool | None = False,
    mask_kwargs: dict[str, Any] | None = None,
    stats_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Generate a spatial summary statistics of the data.

    Returns a pandas DataFrame of data consireding the given areas based on `statistic_summary` function.
    The summary includes count, mean, std, min, 25%, 50%, 75%, and max.
    Bins can be provided to further group the data into intervals.

    Parameters
    ----------
    data : xr.DataArray | xr.Dataset
        The input data.
    areas : gpd.GeoDataFrame
        Areas to be used as spatial masks.
    name : str, optional
        Name of the area column in the output DataFrame. Default is 'area'.
    on_vars : list, optional
        Variables on which the statistics are calculated. If None (default), all variables are kept.
    on_dims : list, optional
        Dimensions on which the statistics are calculated. If None (default), all dimensions are kept.
        Spatial dimensions (i.e., `lon` or `x` and `lat` or `y`) are removed by default.
    on_dim_values : sequence, optional
        Values of dimensions to be kept in the summary. If None (default), all values are kept.
    bins : list or np.ndarray, optional
        Bins defining data intervals. If None (default), no binning is performed.
    bins_labels : list, optional
        Labels for the bins. If None (default), bins values are used as labels.
        The length of the list must be equal to the number of bins. Ignored if `bins` is None.
    all_bins : bool, optional
        If True, a additional bin corresponding to the bounds of `bins` is added to the summary. Default is False.
        Ignored if `bins` is None.
    cell_area : tuple of float or int and str, optional
        Add a column to the summary with the given associated area calculated based on the count statistic
        variable. The tuple must contain the area value and the unit of the area.
    dropna : bool, optional
        If True, dimensions with NaN values are removed. Default is False.
    mask_kwargs : dict, optional
        Additional keyword arguments passed to `regionmask.from_geopandas`.
    stats_kwargs : dict, optional
        Additional keyword arguments passed to `statistics_summary`.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the statistics for the defined areas, dimensions and variables, including:
        count, mean, std, min, 25%, 50%, 75%, and max.
    """
    if mask_kwargs is None:
        mask_kwargs = {}
    if stats_kwargs is None:
        stats_kwargs = {}

    regions = regionmask.from_geopandas(areas, name=name, **mask_kwargs)
    mask = regions.mask_3D(data)

    out = []
    for r in mask.region.values:
        df = stats_summary(
            data.where(mask.sel(region=r)),
            on_vars=on_vars,
            on_dims=on_dims,
            on_dim_values=on_dim_values,
            bins=bins,
            bins_labels=bins_labels,
            all_bins=all_bins,
            cell_area=cell_area,
            dropna=dropna,
            **stats_kwargs,
        )
        df.insert(0, name, regions[r].name)
        out.append(df)
    return pd.concat(out)
