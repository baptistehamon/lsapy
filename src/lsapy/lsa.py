"""Land Suitability definition."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import xarray as xr

from lsapy.aggregate import aggregate
from lsapy.core.formatting import lsa_repr
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
    short_name : str, optional
        A short name for the land suitability analysis. The default is None. If provided,
        it will be stored as an attribute.
    long_name : str, optional
        A long name for the land suitability analysis. The default is None. If provided,
        it will be stored as an attribute.
    description : str, optional
        A description for the land suitability analysis. The default is None. If provided,
        it will be stored as an attribute.
    comment : str, optional
        Additional information about the land suitability analysis. The default is None.
        If provided, it will be stored as an attribute.
    attrs : Mapping[Any, Any], optional
        Arbitrary metadata to store with the land suitability analysis, in addition to the attributes
        `short_name`, `long_name`, `description`, and `comment`. The default is None.

    Examples
    --------
    Let first define the ``SuitabilityCriteria`` (we use `xclim` package for the GDD computation):

    >>> from lsapy.utils import open_data
    >>> from lsapy.functions import SuitabilityFunction
    >>> from xclim.indicators.atmos import growing_degree_days

    >>> drainage = open_data("land", variables="drainage")
    >>> tas = open_data("climate", variables="tas")
    >>> sc = {
    ...     "drainage_class": SuitabilityCriteria(
    ...         name="drainage_class",
    ...         long_name="Drainage Class Suitability",
    ...         weight=3,
    ...         category="soilTerrain",
    ...         indicator=drainage,
    ...         func="discrete",
    ...         fparams={"rules": {0: 0, 1: 0.1, 2: 0.5, 3: 0.9, 4: 1}},
    ...     ),
    ...     "growing_degree_days": SuitabilityCriteria(
    ...         name="growing_degree_days",
    ...         long_name="Growing Degree Days Suitability",
    ...         weight=1,
    ...         category="climate",
    ...         indicator=growing_degree_days(tas, thresh="10 degC", freq="YS-JUL"),
    ...         func="vetharaniam2022_eq5",
    ...         fparams={"a": -1.41, "b": 801},
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
        attrs: Mapping[Any, Any] | None = None,
    ) -> None:
        self.land_use = land_use
        self.criteria = criteria

        _attrs = {}
        if short_name:
            _attrs.update({"short_name": short_name})
        if long_name:
            _attrs.update({"long_name": long_name})
        if description:
            _attrs.update({"description": description})
        if comment:
            _attrs.update({"comment": comment})
        if attrs:
            _attrs.update(dict(attrs))
        self.attrs = _attrs

        self._sort_criteria_by_weight()  # important if suitability as limited factor
        self._criteria_list = [sc.name for sc in self.criteria.values()]
        self.category = list(dict.fromkeys([sc.category for sc in self.criteria.values()]))
        self._get_params_by_category()

    def __repr__(self) -> str:
        """Return a string representation of the land suitability."""
        return lsa_repr(self)

    @property
    def land_use(self) -> str:
        """
        Name of the land use.

        Returns
        -------
        str
            Name of the land use.
        """
        return self._land_use

    @land_use.setter
    def land_use(self, value: str) -> None:
        """
        Set the name of the land use.

        Parameters
        ----------
        value : str
            Name of the land use.
        """
        self._land_use = value

    @property
    def criteria(self) -> dict[str, SuitabilityCriteria]:
        """
        Dictionary of the suitability criteria.

        Returns
        -------
        dict
            Dictionary containing the suitability criteria.
        """
        return self._criteria

    @criteria.setter
    def criteria(self, value: dict[str, SuitabilityCriteria]) -> None:
        """
        Set the suitability criteria.

        Parameters
        ----------
        value : dict[str, SuitabilityCriteria]
            Dictionary of suitability criteria where the key is the name of the criteria.
        """
        self._criteria = value

    @property
    def data(self) -> xr.Dataset:
        """
        Dataset containing the computed suitability.

        Returns
        -------
        xr.Dataset
            Dataset containing the computed suitability.
        """
        return self._data

    @data.setter
    def data(self, value: xr.Dataset) -> None:
        """
        Set the computed suitability dataset.

        Parameters
        ----------
        value : xr.Dataset
            Dataset containing the computed suitability.
        """
        self._data = value

    @property
    def category(self) -> list[str | None]:
        """
        List of categories defined in the suitability criteria.

        Returns
        -------
        list[str | None]
            List of categories defined in the suitability criteria.
        """
        return self._category

    @category.setter
    def category(self, value: list[str | None]) -> None:
        """
        Set the list of categories defined in the suitability criteria.

        Parameters
        ----------
        value : list[str | None]
            List of categories defined in the suitability criteria.
        """
        self._category = value

    @property
    def criteria_by_category(self) -> dict[str | None, list[str]]:
        """
        Dictionary of criteria names grouped by category.

        Returns
        -------
        dict
            Dictionary where keys are categories and values are lists of criteria names.
        """
        return self._criteria_by_category

    @criteria_by_category.setter
    def criteria_by_category(self, value: dict[str | None, list[str]]) -> None:
        """
        Set the dictionary of criteria names grouped by category.

        Parameters
        ----------
        value : dict[str | None, list[str]]
            Dictionary where keys are categories and values are lists of criteria names.
        """
        self._criteria_by_category = value

    @property
    def weights_by_category(self) -> dict[str | None, float]:
        """
        Dictionary of total weights grouped by category.

        Returns
        -------
        dict
            Dictionary where keys are categories and values are total weights.
        """
        return self._weights_by_category

    @weights_by_category.setter
    def weights_by_category(self, value: dict[str | None, float]) -> None:
        """
        Set the dictionary of total weights grouped by category.

        Parameters
        ----------
        value : dict[str | None, float]
            Dictionary where keys are categories and values are total weights.
        """
        self._weights_by_category = value

    @property
    def attrs(self) -> dict[Any, Any]:
        """
        Dictionary of the Land Suitability Analysis attributes.

        Returns
        -------
        dict
            Dictionary containing the attributes of the Land Suitability Analysis.
        """
        return self._attrs

    @attrs.setter
    def attrs(self, value: Mapping[Any, Any]) -> None:
        """
        Set the attributes of the Land Suitability Analysis.

        Parameters
        ----------
        value : Mapping[Any, Any]
            Mapping of attributes to set for the Land Suitability Analysis.
        """
        self._attrs = dict(value)

    def run(
        self,
        suitability_type: str = "overall",
        agg_methods: str | dict[str, str] = "mean",
        by_category: bool | None = None,
        keep_vars: bool | None = True,
        inplace=False,
        **kwargs,
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
        **kwargs : dict
            Additional keyword arguments to pass to the suitability criteria compute method.

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

        >>> from lsapy.utils import open_data
        >>> from lsapy.functions import SuitabilityFunction
        >>> from xclim.indicators.atmos import growing_degree_days

        >>> drainage = open_data("land", variables="drainage")
        >>> tas = open_data("climate", variables="tas")
        >>> sc = {
        ...     "drainage_class": SuitabilityCriteria(
        ...         name="drainage_class",
        ...         long_name="Drainage Class Suitability",
        ...         weight=3,
        ...         category="soilTerrain",
        ...         indicator=drainage,
        ...         func="discrete",
        ...         fparams={"rules": {0: 0, 1: 0.1, 2: 0.5, 3: 0.9, 4: 1}},
        ...     ),
        ...     "growing_degree_days": SuitabilityCriteria(
        ...         name="growing_degree_days",
        ...         long_name="Growing Degree Days Suitability",
        ...         weight=1,
        ...         category="climate",
        ...         indicator=growing_degree_days(tas, thresh="10 degC", freq="YS-JUL"),
        ...         func="vetharaniam2022_eq5",
        ...         fparams={"a": -1.41, "b": 801},
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

        ds = self._run_criteria(**kwargs)

        if suitability_type in ["category", "overall"]:
            agg_on, by_category = _pre_agg(suitability_type, by_category)

            if agg_on != {}:
                if isinstance(agg_methods, str):
                    agg_methods = {"category": agg_methods, "suitability": agg_methods}
                elif not isinstance(agg_methods, dict):
                    raise TypeError(f"'agg_methods' must be a string or a dictionary. Got {type(agg_methods)}.")
                elif "overall" in agg_methods.keys():
                    suit_method = agg_methods.pop("overall")
                    agg_methods = {**agg_methods, "suitability": suit_method}

                if "category" not in agg_methods.keys() and (by_category or suitability_type == "category"):
                    raise ValueError("No aggregation method provided for 'category'.")
                elif suitability_type == "category" or by_category:
                    cat_method = agg_methods.pop("category")
                    agg_methods = {
                        **{k: cat_method for k in self.category if k is not None},
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
        **kwargs,
    ) -> xr.Dataset:
        """
        Compute the suitability of each criteria.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the suitability criteria compute method.

        Returns
        -------
        xr.Dataset
            A Dataset containing the computed suitability for each criteria.
        """
        out: Any = []
        for sc in self.criteria.values():
            out.append(sc.compute(**kwargs))
        out = xr.merge(out, compat="override", combine_attrs="drop")

        # Reassign attributes to each criteria
        for sc in out.data_vars:
            out[sc].attrs = self.criteria[sc].attrs
        out.attrs["land_use"] = self.land_use
        out.attrs["criteria"] = self._criteria_list
        out.attrs.update(self.attrs)
        return out

    def _sort_criteria_by_weight(self) -> None:
        self.criteria = dict(
            sorted(
                self.criteria.items(),
                key=lambda item: item[1].weight if item[1].weight is not None else 0,
                reverse=True,
            )
        )

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

            if k_method in ["wmean", "wgmean"]:
                kwargs[k] = {
                    "weights": [
                        self.weights_by_category[_v] if _v in self.category else self.criteria[_v].weight for _v in v
                    ]
                }
            else:
                kwargs[k] = {}

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
        if isinstance(methods, str):
            methods = {k: methods for k in agg_on.keys()}
        elif not isinstance(methods, dict):
            raise TypeError("'methods' must be a string or a dictionary of strings.")

        for k, v in agg_on.items():
            if kwargs and k in kwargs:
                kwargs_k = kwargs[k]
            else:
                kwargs_k = {}

            out = aggregate(ds, method=methods[k], variables=v, **kwargs_k)

            if methods[k] == "limfactor" and isinstance(out, xr.Dataset):
                ds[k] = out["limiting_factor"]
                ds[f"{k}_limvar"] = out["limiting_variable"].assign_attrs(
                    {"long_name": f"{k.capitalize()} Limiting Factor Variable"}
                )

            else:
                ds[k] = out

            ds[k].attrs.update({"long_name": f"{k.capitalize()} Suitability"})

        if keep_vars:
            return ds

        vars_to_keep = [k for k in agg_on.keys() if k not in [i for e in agg_on.values() for i in e]]
        return ds[[i for v in vars_to_keep for i in ([v, f"{v}_limvar"] if methods[v] == "limfactor" else [v])]]
