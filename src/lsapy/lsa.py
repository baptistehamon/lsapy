"""Land Suitability definition."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import xarray as xr

from lsapy.criteria import SuitabilityCriteria

__all__ = ["LandSuitabilityAnalysis"]


class LandSuitabilityAnalysis:
    """
    Data structure to define and run land suitability analysis.

    The land suitability analysis is defined by a set of suitability criteria that are combined to
    compute the suitability.

    Parameters
    ----------
    land_use : str
        A name for the land use.
    criteria : dict[str, SuitabilityCriteria]
        A dictionary of suitability criteria where the key is the name of the criteria.
    short_name : str | None, optional
        A short name for the land suitability analysis. The default is None.
    long_name : str | None, optional
        A long name for the land suitability analysis. The default is None.
    description : str | None, optional
        A description for the land suitability analysis. The default is None.
    comment : str | None, optional
        Additional information about the land suitability analysis.

    Examples
    --------
    Let first define the ``SuitabilityCriteria`` (we use `xclim` package for the GDD computation):

    >>> from lsapy.utils import load_soil_data, load_climate_data
    >>> from lsapy.functions import SuitabilityFunction
    >>> from xclim.indicators.atmos import growing_degree_days

    >>> soil_data = load_soil_data()
    >>> climate_data = load_climate_data()
    >>> sc = {
    ...     "drainage_class": SuitabilityCriteria(
    ...         name="drainage_class",
    ...         long_name="Drainage Class Suitability",
    ...         weight=3,
    ...         category="soilTerrain",
    ...         indicator=soil_data["DRC"],
    ...         func=SuitabilityFunction(
    ...             name="discrete", params={"rules": {"1": 0, "2": 0.1, "3": 0.5, "4": 0.9, "5": 1}}
    ...         ),
    ...     ),
    ...     "growing_degree_days": SuitabilityCriteria(
    ...         name="growing_degree_days",
    ...         long_name="Growing Degree Days Suitability",
    ...         weight=1,
    ...         category="climate",
    ...         indicator=growing_degree_days(climate_data["tas"], thresh="10 degC", freq="YS-JUL"),
    ...         func=SuitabilityFunction(name="vetharaniam2022_eq5", params={"a": -1.41, "b": 801}),
    ...     ),
    ... }

    Now we can define the ``LandSuitabilityAnalysis`` :

    >>> lsa = LandSuitabilityAnalysis(
    ...     land_use="land_use",
    ...     short_name="land_suitability_analysis",
    ...     long_name="Land Suitability Analysis",
    ...     criteria=sc,
    ... )

    The land suitability analysis can now be run:

    >>> lsa.run(inplace=True)
    """

    def __init__(
        self,
        land_use: str,
        criteria: dict[str, SuitabilityCriteria],
        short_name: str | None = None,
        long_name: str | None = None,
        description: str | None = None,
        comment: str | None = None,
    ) -> None:
        self.land_use = land_use
        self.criteria = criteria
        self.short_name = short_name
        self.long_name = long_name
        self.description = description
        self.comment = comment

        self._sort_criteria_by_weight()  # important if suitability as limited factor
        self._criteria_list = [sc.name for sc in self.criteria.values()]
        self.category = list(dict.fromkeys([sc.category for sc in self.criteria.values()]))
        self._get_params_by_category()

    def __repr__(self) -> str:
        """Return a string representation of the land suitability."""
        if hasattr(self, "data"):
            return self.data.__repr__()
        else:
            attrs = []
            for k, v in self.attrs.items():
                if isinstance(v, str):
                    v_ = f"'{v}'"
                else:
                    v_ = v
                attrs.append(f"{k}={v_}")
            return f"{self.__class__.__name__}({', '.join(attrs) if attrs else ''})"

    @property
    def attrs(self):
        """
        Dictionary of attributes of the land suitability analysis.

        Returns
        -------
        dict
            A dictionary of attributes of the land suitability analysis, including name, criteria, short_name,
            long_name, and description.
        """
        return {
            k: v
            for k, v in {
                "land_use": self.land_use,
                "criteria": self._criteria_list,
                "short_name": self.short_name,
                "long_name": self.long_name,
                "description": self.description,
            }.items()
            if v is not None
        }

    def run(
        self,
        suitability_type: str = "overall",
        agg_methods: str | dict[str, str] = "mean",
        by_category: bool | None = None,
        keep_vars: bool | None = True,
        inplace=False,
    ):
        """
        Run the land suitability analysis.

        Parameters
        ----------
        suitability_type : str, optional
            The type of suitability to compute. Options are 'criteria', 'category', or 'overall'.
            The default is 'overall'.
        agg_methods : str | dict[str, str], optional
            The aggregation method to use for the suitability computation. If a string, it applies the same method
            to compute the category and overall suitability. If a dictionary, the keys 'category' and 'overall'
            are used to specify the aggregation method to use for each type of suitability. The default is 'mean'.
        by_category : bool | None, optional
            If True, compute the overall suitability aggregating categories suitability. If False, use the criteria
            suitability. The default behavior uses categories suitability if categories are found in criteria, otherwise
            it uses the criteria suitability.
        keep_vars : bool | None, optional
            If True, return all the variables computed as part of the computation process, otherwise return only the
            data defined by the `suitability_type`. The default is True.
        inplace : bool, optional
            If True, compute the suitability in place. The default is False.

        Returns
        -------
        None | xr.Dataset
            If `inplace` is False, return the computed suitability as a Dataset. If `inplace` is True, return None.

        Notes
        -----
        To avoid biais in LSA categories outputs, it was decided to apply the same aggregation method to all categories.

        Examples
        --------
        Let first define the ``SuitabilityCriteria`` (we use `xclim` package for the GDD computation):

        >>> from lsapy.utils import load_soil_data, load_climate_data
        >>> from lsapy.functions import SuitabilityFunction
        >>> from xclim.indicators.atmos import growing_degree_days

        >>> soil_data = load_soil_data()
        >>> climate_data = load_climate_data()
        >>> sc = {
        ...     "drainage_class": SuitabilityCriteria(
        ...         name="drainage_class",
        ...         long_name="Drainage Class Suitability",
        ...         weight=3,
        ...         category="soilTerrain",
        ...         indicator=soil_data["DRC"],
        ...         func=SuitabilityFunction(
        ...             name="discrete", params={"rules": {"1": 0, "2": 0.1, "3": 0.5, "4": 0.9, "5": 1}}
        ...         ),
        ...     ),
        ...     "growing_degree_days": SuitabilityCriteria(
        ...         name="growing_degree_days",
        ...         long_name="Growing Degree Days Suitability",
        ...         weight=1,
        ...         category="climate",
        ...         indicator=growing_degree_days(climate_data["tas"], thresh="10 degC", freq="YS-JUL"),
        ...         func=SuitabilityFunction(name="vetharaniam2022_eq5", params={"a": -1.41, "b": 801}),
        ...     ),
        ... }

        Now we can define the ``LandSuitabilityAnalysis`` :

        >>> lsa = LandSuitabilityAnalysis(
        ...     land_use="land_use",
        ...     short_name="land_suitability_analysis",
        ...     long_name="Land Suitability Analysis",
        ...     criteria=sc,
        ... )

        The land suitability analysis can now be run:

        >>> lsa.run(
        ...     suitability_type="overall",
        ...     agg_methods={"category": "weighted_geomean", "overall": "mean"},
        ...     by_category=True,
        ...     inplace=True,
        ... )
        """

        def _pre_agg(suitability_type, by_category):
            """Prepare the aggregation variables and methods based on defined parameters."""
            agg_on = {}

            if suitability_type == "overall":
                if by_category is None and self.category == [None]:
                    by_category = False
                elif by_category is None:
                    by_category = True

                if by_category:
                    agg_on = {"suitability": self.category}
                else:
                    agg_on = {"suitability": self._criteria_list}

            if suitability_type == "category" or by_category:
                if self.category == [None] and suitability_type == "overall":
                    warnings.warn(
                        "No categories defined. Computing suitability on criteria instead.",
                        UserWarning,
                        stacklevel=2,
                    )
                    agg_on.update({"suitability": self._criteria_list})
                elif self.category == [None]:
                    warnings.warn(
                        "No categories defined. Skipping category suitability computation.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    agg_on = {**self.criteria_by_category, **agg_on}

            return agg_on, by_category

        suitability_type = suitability_type.lower()

        if suitability_type not in ["criteria", "category", "overall"]:
            raise ValueError(
                f"'suitability_type' must be one of 'criteria', 'category', or 'overall'. Got '{suitability_type}'."
            )

        ds = self._run_criteria()

        if suitability_type in ["category", "overall"]:
            agg_on, by_category = _pre_agg(suitability_type, by_category)

            if agg_on != {}:
                if isinstance(agg_methods, str):
                    agg_methods = {"category": agg_methods, "suitability": agg_methods}
                elif "overall" in agg_methods.keys():
                    suit_method = agg_methods.pop("overall")
                    agg_methods = {**agg_methods, "suitability": suit_method}
                elif not isinstance(agg_methods, dict):
                    raise ValueError(f"'agg_method' must be a string or a dictionary. Got {type(agg_methods)}.")

                if "category" not in agg_methods.keys() and (by_category or suitability_type == "category"):
                    raise ValueError("No aggregation method provided for 'category'.")
                elif suitability_type == "category" or by_category:
                    cat_method = agg_methods.pop("category")
                    agg_methods = {
                        **{k: cat_method for k in self.category},
                        **agg_methods,
                    }

                if "suitability" not in agg_methods.keys() and suitability_type == "overall":
                    raise ValueError("No aggregation method provided for 'overall'.")

                ds = self._aggregate(
                    ds,
                    agg_on=agg_on,
                    methods=agg_methods,
                    keep_vars=keep_vars,
                    kwargs=self._format_agg_kwargs(agg_methods, agg_on),
                )

        if not inplace:
            return ds
        else:
            if hasattr(self, "data"):
                warnings.warn("Existing data found and will be overwritten.", UserWarning, stacklevel=2)
            self.data = ds

    def _run_criteria(
        self,
    ) -> xr.Dataset:
        """
        Compute the suitability of each criteria.

        Returns
        -------
        xr.Dataset
            A Dataset containing the computed suitability for each criteria.
        """
        out = []
        for sc in self.criteria.values():
            out.append(sc.compute())
        out = xr.merge(out, compat="override", combine_attrs="drop")

        # Reassign attributes to each criteria
        for sc in out.data_vars:
            out[sc].attrs = {k: v for k, v in self.criteria[sc].attrs.items() if k != "is_computed"}
        out.attrs.update(self.attrs)
        return out

    def _sort_criteria_by_weight(self) -> dict[str, SuitabilityCriteria]:
        self.criteria = dict(sorted(self.criteria.items(), key=lambda item: item[1].weight, reverse=True))

    def _get_params_by_category(self):
        self.criteria_by_category = {category: [] for category in self.category}
        for sc in self.criteria.values():
            self.criteria_by_category[sc.category].append(sc.name)

        self.weights_by_category = {category: [] for category in self.category}
        for category in self.category:
            self.weights_by_category[category] = sum(
                [sc.weight for sc in self.criteria.values() if sc.category == category]
            )

    def _format_agg_kwargs(self, agg_methods: str | dict[str, str], agg_on: dict[str, list[str]]) -> dict[str, Any]:
        """
        Format the keyword arguments for the reduction function based on the LandSuitabilityAnalysis object and dataset.

        Parameters
        ----------
        agg_methods : str | dict[str, str]
            The aggregation methods to apply. If a string, it applies the same method for all aggregations.
            If a dictionary, keys are aggregated variable names and values are the associated aggregation methods.
        agg_on : dict[str, list[str]]
            A dictionary where keys are new variable names and values are lists of variable names to aggregate.

        Returns
        -------
        kwargs : dict[str, Any]
            A dictionary of keyword arguments to pass to the aggregation function for each variable.
        """
        kwargs = {}
        for k, v in agg_on.items():
            k_method = agg_methods if isinstance(agg_methods, str) else agg_methods[k]

            if k_method in ["mean", "geomean"]:
                kwargs[k] = {}
            elif k_method in ["weighted_mean", "weighted_geomean"]:
                if k not in self.category and any([i in self.category for i in v]):
                    kwargs[k] = {"weights": [self.weights_by_category[cat] for cat in v]}
                else:
                    kwargs[k] = {"weights": [self.criteria[sc].weight for sc in v]}
            elif k_method == "limiting_factor":
                kwargs[k] = {"limiting_var": True}

        return kwargs

    @staticmethod
    def _aggregate(
        ds: xr.Dataset,
        agg_on: dict[str, list[str]],
        methods: str | dict[str, str],
        keep_vars: bool | None = False,
        kwargs: dict[str, Any] | None = None,
    ):
        """
        Aggregate variables based on specified methods.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset.
        agg_on : dict[str, list[str]]
            A dictionary where keys are new variable names and values are lists of variable names to aggregate.
        methods : str or dict[str, str]
            The aggregation methods to apply. If a string, it applies the same method for all aggregations.
            If a dictionary, keys are aggregated variable names and values are the associated aggregation methods.
        keep_vars : bool, optional
            If True, keeps the original variables in the output dataset.
        kwargs : dict[str, Any], optional
            Additional keyword arguments to pass to the aggregation function for each variable.

        Returns
        -------
        xr.Dataset
            A dataset with the aggregated variables. If `keep_vars` is True, all original variables are kept.
            Otherwise, only the aggregated variables are returned.
        """
        if methods is None:
            raise ValueError("Method must be specified.")
        elif isinstance(methods, str):
            methods = {k: methods for k in agg_on.keys()}
        elif not isinstance(methods, dict):
            raise TypeError("Method must be a string or a dictionary of strings.")

        for k, v in agg_on.items():
            if kwargs and k in kwargs:
                kwargs_k = kwargs[k]
            else:
                kwargs_k = {}

            out = _aggregate_vars(ds, method=methods[k], vars=v, **kwargs_k)

            if methods[k] == "limiting_factor" and isinstance(out, xr.Dataset):
                ds[k] = out["limiting_factor"]
                ds[f"{k}_limvar"] = out["limiting_var"].assign_attrs(
                    {"long_name": f"{k.capitalize()} Limiting Factor Variable"}
                )

            else:
                ds[k] = out

            ds[k].attrs.update({"long_name": f"{k.capitalize()} Suitability"})

        if keep_vars:
            return ds

        vars_to_keep = [k for k in agg_on.keys() if k not in [i for e in agg_on.values() for i in e]]
        return ds[[i for v in vars_to_keep for i in ([v, f"{v}_limvar"] if methods[v] == "limiting_factor" else [v])]]


def vars_weighted_mean(ds: xr.Dataset, vars=None, weights=None) -> xr.DataArray:
    """
    Compute the weighted mean of the variables.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the variables to compute the weighted mean.
    vars : list[str] | None, optional
        List of variable names to include in the weighted mean.
        If None, all data variables are used.
    weights : list[float] | None, optional
        List of weights corresponding to the variables.
        If None, equal weights are used (default is 1 for each variable).

    Returns
    -------
    xr.DataArray
        A DataArray containing the weighted mean of the specified variables.
    """
    if vars is None:
        vars = list(ds.data_vars)
    if weights is None:
        weights = np.ones(len(vars))

    s = sum([ds[v] * w for v, w in zip(vars, weights, strict=False)])
    da: xr.DataArray = s / sum(weights)
    return da.assign_attrs(
        {
            "method": "Weighted Mean",
            "description": (
                f"Weighted Mean of variables: {', '.join([f'{v} ({w})' for v, w in zip(vars, weights, strict=False)])}."
            ),
        }
    ).rename("weighted_mean")


def vars_mean(ds: xr.Dataset, vars=None) -> xr.DataArray:
    """
    Compute the mean of the variables.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the variables to compute the mean.
    vars : list[str] | None, optional
        List of variable names to include in the mean.
        If None, all data variables are used.

    Returns
    -------
    xr.DataArray
        A DataArray containing the mean of the specified variables.
    """
    if vars is None:
        vars = list(ds.data_vars)
    da = vars_weighted_mean(ds, vars=vars)
    return da.assign_attrs({"method": "Mean", "description": f"Mean of variables: {', '.join(vars)}."}).rename("mean")


def vars_weighted_geomean(ds: xr.Dataset, vars=None, weights=None) -> xr.DataArray:
    """
    Compute the weighted geometric mean of the variables.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the variables to compute the weighted geometric mean.
    vars : list[str] | None, optional
        List of variable names to include in the weighted geometric mean.
        If None, all data variables are used.
    weights : list[float] | None, optional
        List of weights corresponding to the variables.
        If None, equal weights are used (default is 1 for each variable).

    Returns
    -------
    xr.DataArray
        A DataArray containing the weighted geometric mean of the specified variables.
    """
    if vars is None:
        vars = list(ds.data_vars)
    if weights is None:
        weights = np.ones(len(vars))

    s = sum([np.log(ds[v]) * w for v, w in zip(vars, weights, strict=False)])
    da: xr.DataArray = np.exp(s / sum(weights))
    return da.assign_attrs(
        {
            "method": "Weighted Geometric Mean",
            "description": (
                "Weighted Geometric Mean of variables: "
                f"{', '.join([f'{v} ({w})' for v, w in zip(vars, weights, strict=False)])}."
            ),
        }
    ).rename("weighted_geometric_mean")


def vars_geomean(ds: xr.Dataset, vars=None) -> xr.DataArray:
    """
    Compute the geometric mean of the variables.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the variables to compute the geometric mean.
    vars : list[str] | None, optional
        List of variable names to include in the geometric mean.
        If None, all data variables are used.

    Returns
    -------
    xr.DataArray
        A DataArray containing the geometric mean of the specified variables.
    """
    if vars is None:
        vars = list(ds.data_vars)
    da = vars_weighted_geomean(ds, vars=vars)
    return da.assign_attrs(
        {"method": "Geometric Mean", "description": f"Geometric Mean of variables: {', '.join(vars)}."}
    ).rename("geometric_mean")


def limiting_factor(ds: xr.Dataset, vars=None, limiting_var: bool | None = True) -> xr.DataArray | xr.Dataset:
    """
    Compute the limiting factor among the variables.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset containing the variables to compute the limiting factor.
    vars : list[str] | None, optional
        List of variable names to include in the limiting factor computation.
        If None, all data variables are used.
    limiting_var : bool, optional
        If True, return the variable that is the limiting factor. The default is True.

    Returns
    -------
    xr.DataArray | xr.Dataset
        A DataArray containing the value of the limiting factor among the specified variables.
        If `limiting_var` is True, a Dataset is returned containing both the limiting factor
        and the variable that is the limiting factor.
    """
    if vars is None:
        vars = list(ds.data_vars)

    da = ds[vars].to_array()
    mask = da.notnull().all(dim="variable")

    lim = da.min(dim="variable", skipna=True).where(mask).rename("limiting_factor")
    lim = lim.assign_attrs(
        {"method": "Limiting Factor", "description": f"Value of limiting factor among variables: {', '.join(vars)}."}
    )
    if limiting_var:
        lim_var = da.fillna(9999).argmin(dim="variable", skipna=True).where(mask).rename("limiting_var")
        lim_var.attrs = {
            "method": "Limiting Factor",
            "description": f"Limiting factor among: {', '.join(vars)}.",
            "legend": {f"{i}": v for i, v in enumerate(vars)},
        }
        return xr.merge([lim, lim_var])
    return lim.to_dataset()


def _aggregate_vars(
    ds: xr.Dataset, method: str = "mean", vars=None, weights=None, **kwargs
) -> xr.DataArray | xr.Dataset:
    if method.lower() == "mean":
        return vars_mean(ds, vars=vars)
    elif method.lower() == "weighted_mean":
        return vars_weighted_mean(ds, vars=vars, weights=weights)
    elif method.lower() == "geomean":
        return vars_geomean(ds, vars=vars)
    elif method.lower() == "weighted_geomean":
        return vars_weighted_geomean(ds, vars=vars, weights=weights)
    elif method.lower() == "limiting_factor":
        return limiting_factor(ds, vars=vars, **kwargs)
    else:
        raise ValueError(f"Aggregation method '{method}' not recognized.")
