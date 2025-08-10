"""String formatting routines for __repr__."""

from __future__ import annotations


def sf_repr(sf):
    """Return a string representation of a SuitabilityFunction."""
    return f"SuitabilityFunction(func={sf.func.__name__}, params={sf.params})"


def sf_short_repr(sf):
    """Return a short string representation of a SuitabilityFunction."""
    return f"{sf.func.__name__}({', '.join(f'{k}={v}' for k, v in sf.params.items())})"
