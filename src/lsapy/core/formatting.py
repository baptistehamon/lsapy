"""String formatting routines for __repr__."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import xarray as xr
from xarray.core.formatting import (
    _calculate_col_width,  # noqa: PLC2701
    _mapping_repr,  # noqa: PLC2701
    attrs_repr,
    dim_summary_limited,
    inline_variable_array_repr,
    maybe_truncate,
    pretty_print,
    render_human_readable_nbytes,
)

from lsapy.core.options import OPTIONS

if TYPE_CHECKING:
    from lsapy import LandSuitabilityAnalysis, SuitabilityCriteria, SuitabilityFunction  # type: ignore[attr-defined]


def sf_repr(sf: SuitabilityFunction) -> str:
    """Return a short string representation of a SuitabilityFunction."""
    func = f"{sf.func.__name__}"
    if not sf.params:
        return f"{func}()"
    return f"{func}({', '.join(f'{k}={v}' for k, v in sf.params.items())})"


def sc_func_repr(func) -> str:
    """Return a string representation of a suitability criteria standardization function."""
    if isinstance(func, functools.partial):
        f = f"{func.func.__name__}"
        if not func.args and not func.keywords:
            return f"{f}()"
        args = [repr(a) for a in func.args] if func.args else []
        kwargs = [f"{k}={v!r}" for k, v in func.keywords.items()] if func.keywords else []
        return f"{f}({', '.join(args + kwargs)})"
    else:
        return repr(func)


def sc_params_repr(sc: SuitabilityCriteria) -> str:
    """Format "weight" and "category" for SuitabilityCriteria."""
    summary = [f"weight: {sc.weight}"]
    if sc.category:
        summary.append(f"category: {sc.category}")
    return f"({', '.join(summary)})"


def data_repr(obj: xr.DataArray, col_width: int, max_width: int) -> str:
    """Format indicator data for SuitabilityCriteria."""
    first_col = pretty_print("    Data", col_width)
    nbytes_str = f" {render_human_readable_nbytes(obj.nbytes)}"
    front_str = f"{first_col}{obj.dtype}{nbytes_str} "

    values_width = max_width - len(front_str)
    values_str = inline_variable_array_repr(obj.variable, values_width)

    return front_str + values_str


def sc_repr(sc: SuitabilityCriteria) -> str:
    """Return a string representation of a SuitabilityCriteria."""
    max_rows = OPTIONS["display_max_rows"]
    max_width = OPTIONS["display_width"]

    col_width = _calculate_col_width([f"{k}:" for k in sc.attrs.keys()] + ["Dimensions"])

    if not sc.name:
        name = ""
    else:
        name = f" {sc.name!r}"

    summary = [f"<SuitabilityCriteria>{name} {sc_params_repr(sc)}"]

    if sc.func:
        summary.extend(["Function:", f"    {maybe_truncate(sc_func_repr(sc.func), max_width)}"])

    if sc.indicator is not None:
        dims = pretty_print("    Dimensions", col_width)
        summary.extend(
            [
                "Indicator:",
                f"{pretty_print('    Name', col_width)}{sc.indicator.name} ",
                data_repr(sc.indicator, col_width, max_width),
                f"{dims}{dim_summary_limited(sc.indicator.sizes, len(dims) + 1, max_rows)} ",
            ]
        )

    if sc.attrs:
        summary.append(attrs_repr(sc.attrs, col_width=col_width, max_rows=max_rows))

    if not sc.func and not sc.indicator and not sc.attrs:
        summary.append("    *undefined*")

    return "\n".join(summary)


def summarize_criteria(
    name: str,
    criteria: SuitabilityCriteria,
    col_width: int,
    max_width: int | None = None,
) -> str:
    """Summarize a criteria in one line, e.g., for the LandSuitabilityAnalysis.__repr__."""
    if max_width is None:
        max_width_options = OPTIONS["display_width"]
        if not isinstance(max_width_options, int):
            raise TypeError(f"`max_width` value of `{max_width}` is not a valid int")
        else:
            max_width = max_width_options

    first_col = pretty_print(f"    {criteria.name} ", col_width)
    wgt = f"(w={criteria.weight}) "

    front_str = f"{first_col}{wgt}"

    if criteria.category:
        cat = f"{criteria.category} "
    else:
        cat = ""

    if criteria.func:
        func = f"{sc_func_repr(criteria.func)} "
    else:
        func = ""

    details_width = max_width - len(front_str)
    details_str = maybe_truncate(f"{cat}{func}", details_width)

    return front_str + details_str


criteria_repr = functools.partial(
    _mapping_repr,
    title="Criteria",
    summarizer=summarize_criteria,
    expand_option_name="display_expand_data_vars",
)


def lsa_repr(lsa: LandSuitabilityAnalysis) -> str:
    """Return a string representation of a LandSuitabilityAnalysis."""
    max_rows = OPTIONS["display_max_rows"]

    col_width = _calculate_col_width(
        [sc.name for sc in lsa.criteria.values()] + [k for k in lsa.attrs.keys()],
    )

    summary = [f"<LandSuitabilityAnalysis> {lsa.land_use!r}"]

    summary.append(criteria_repr(lsa.criteria, col_width=col_width, max_rows=max_rows))

    summary.append(attrs_repr(lsa.attrs, col_width=col_width, max_rows=max_rows))
    return "\n".join(summary)
