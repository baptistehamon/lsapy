"""Suitability Function Utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

equations: dict[str, dict[str, Callable[[Any], Any]]] = {}
_alt_names: dict[str, str] = {}

F = TypeVar("F", bound=Callable[..., Any])


def declare_equation(ftype: str, alt_name: str | None = None) -> Callable[[F], F]:
    """
    Register an equation in the `equations` mapping under the specified type.

    Parameters
    ----------
    ftype : str
        The type of equation to register.
    alt_name : str, optional
        An alternative name for the equation (not currently supported).

    Returns
    -------
    Callable
        A decorator function that registers the equation.
    """

    def _wrapper(func: F) -> F:
        if ftype not in equations:
            equations[ftype] = {}
        equations[ftype].update({func.__name__: func})
        if alt_name is not None:
            _alt_names.update({alt_name: func.__name__})
        return func

    return _wrapper


def get_function_from_name(name: str) -> Callable[[Any], Any]:
    """
    Retrieve a function by its name from the registered equations.

    Parameters
    ----------
    name : str
        The name of the function to retrieve.

    Returns
    -------
    Callable
        The function corresponding to the given name.
    """
    if name in _alt_names:
        name = _alt_names[name]
    for _, funcs in equations.items():
        if name in funcs:
            return funcs[name]
    raise ValueError(f"Equation `{name}` not implemented.")
