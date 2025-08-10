"""String formatting routines for __repr__."""

from __future__ import annotations

from xarray.core.formatting import (
    _calculate_col_width,  # noqa: PLC2701
    attrs_repr,
    dim_summary_limited,
    inline_variable_array_repr,
    maybe_truncate,
    pretty_print,
    render_human_readable_nbytes,
)

from lsapy.core.options import OPTIONS


def sf_repr(sf):
    """Return a string representation of a SuitabilityFunction."""
    return f"SuitabilityFunction(func={sf.func.__name__}, params={sf.params})"


def sf_short_repr(sf):
    """Return a short string representation of a SuitabilityFunction."""
    return f"{sf.func.__name__}({', '.join(f'{k}={v}' for k, v in sf.params.items())})"


def sc_params_repr(sc):
    """Format "weight" and "category" for SuitabilityCriteria."""
    summary = [f"weight: {sc.weight}"]
    if sc.category:
        summary.append(f"category: {sc.category}")
    return f"({', '.join(summary)})"


def data_repr(obj, col_width, max_width) -> str:
    """Format indicator data for SuitabilityCriteria."""
    first_col = pretty_print("    Data", col_width)
    nbytes_str = f" {render_human_readable_nbytes(obj.nbytes)}"
    front_str = f"{first_col}{obj.dtype}{nbytes_str} "

    values_width = max_width - len(front_str)
    values_str = inline_variable_array_repr(obj.variable, values_width)

    return front_str + values_str


def sc_repr(sc) -> str:
    """Return a string representation of a SuitabilityCriteria."""
    max_rows = OPTIONS["display_max_rows"]
    max_width = OPTIONS["display_width"]

    exclude_attrs = ["name", "weight", "category", "from_indicator", "is_computed"]
    attrs = {k: v for k, v in sc.attrs.items() if k not in exclude_attrs}

    col_width = _calculate_col_width([f"{k}:" for k in attrs.keys()] + ["Dimensions"])
    name_col = pretty_print("    Name", col_width)

    summary = [f"<SuitabilityCriteria> {sc.name!r}{sc_params_repr(sc)}"]

    if sc.func:
        summary.extend(
            [
                "Function:",
                f"    {maybe_truncate(sf_short_repr(sc.func), max_width)}",
                # f"{name_col}{sc.func.func.__name__} "
                # f"{pretty_print("    Parameters", col_width)}{sc.func.params!r} "
            ]
        )

    dims = pretty_print("    Dimensions", col_width)
    summary.extend(
        [
            "Indicator:",
            f"{name_col}{sc.indicator.name} ",
            data_repr(sc.indicator, col_width, max_width),
            f"{dims}{dim_summary_limited(sc.indicator.sizes, len(dims) + 1, max_rows)} ",
        ]
    )

    summary.append(attrs_repr(attrs, max_rows=max_rows))

    return "\n".join(summary)
