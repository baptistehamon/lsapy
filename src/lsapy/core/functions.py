"""Suitability Function Utilities."""

from collections.abc import Callable

equations: dict[str, dict] = {}


def declare_equation(type: str, alt_name: str | None = None) -> Callable:
    """
    Register an equation in the `equations` mapping under the specified type.

    Parameters
    ----------
    type : str
        The type of equation to register.
    alt_name : str, optional
        An alternative name for the equation (not currently supported).

    Returns
    -------
    Callable
        A decorator function that registers the equation.
    """

    # TODO: Support alternative names for equations
    def _decorator(func: callable):
        if type not in equations:
            equations[type] = {}
        equations[type].update({func.__name__: func})
        return func

    return _decorator


def get_function_from_name(name: str) -> Callable:
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
    for _, funcs in equations.items():
        if name in funcs:
            return funcs[name]
    raise ValueError(f"Equation `{name}` not implemented.")
