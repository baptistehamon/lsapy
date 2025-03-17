"""Suitability Functions definitions."""

import warnings
from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "SuitabilityFunction",
    "MembershipSuitFunction",
    "DiscreteSuitFunction"
]


class SuitabilityFunction:

    def __init__(
            self,
            func: Callable | None = None,
            func_method: str | None = None,
            func_params: dict[str, Any] = None
    ):
        if func_params is not None:
            if func is None and func_method is None:
                raise ValueError("If `func_params` is provided, `func` or `func_method` must also be provided.")
        else:
            func_params = {}

        self.func = func
        self.func_method = func_method
        self.func_params = func_params
        if func is None and func_method is not None:
            self.func = _get_function_from_name(func_method)

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"func={self.func.__name__}, "
                f"func_method='{self.func_method}', "
                f"func_params={self.func_params})")

    def __call__(self, x):
        if self.func is None:
            raise ValueError("No function has been provided.")
        return self.func(x, **self.func_params)

    def map(self, x):
        return self(x)

    def plot(self, x) -> None:
        plt.plot(x, self(x))

    @property
    def attrs(self):
        if self.func_method is None and self.func_params is None:
            return {}
        return {k: v for k, v in {
                    'func_method': self.func_method,
                    'func_params': self.func_params
                }.items() if v is not None}


# ---------------------------------------------------------------------------- #
# ------------------------ Membership functions ------------------------------ #
# ---------------------------------------------------------------------------- #


class MembershipSuitFunction(SuitabilityFunction):

    def __init__(
            self,
            func: Callable | None = None,
            func_method: str | None = None,
            func_params: dict[str, int | float] | None = None
    ):
        super().__init__(func, func_method, func_params)

    @staticmethod
    def fit(x, y=None, methods: str | list[str] = 'all', plot: bool = False):
        if y is None:
            y = [0, .25, .5, .75, 1]
        return _fit_mbs_functions(x, np.array(y), methods, plot)


def _prepare_for_fitting(methods:  str | list[str] = 'all'):
    _types = ['sigmoid_like', 'gaussian_like']
    _skipped = []

    if methods == 'all':
        methods = [f for t in _types for f in equations[t.replace('_like', '')]]
    elif isinstance(methods, list) or isinstance(methods, str):
        if isinstance(methods, str):
            methods = [methods]

        _methods = []
        for method in methods:
            if method in _types:
                [_methods.append(m) for m in equations[method.replace('_like', '')].keys()]
            else:
                try:
                    _get_function_from_name(method)
                    _methods.append(method)
                except Exception:
                    _skipped.append(method)
                    warnings.warn(f"`{method}` not found in equations. Skipped.", stacklevel=2)
        methods = _methods
        for m in ['sigmoid', 'vetharaniam2024_eq8']:
            if m in methods:
                methods.remove(m)
                _skipped.append(m)
                if m == 'sigmoid':
                    warnings.warn("No parameters to determine for `sigmoid`. Skipped.", stacklevel=2)
                if m == 'vetharaniam2024_eq8':
                    warnings.warn("Fitting method does not support `vetharaniam2024_eq8`. Skipped.", stacklevel=2)
    return methods, _skipped


def _get_function_p0(method: str, x: np.ndarray) -> list[float]:
    if method in equations['sigmoid']:
        return [1, np.median(x)]
    if method in equations['gaussian']:
        return [1, np.median(x), 1]
    return []


def _fit_mbs_functions(x, y, methods: str | list[str] = 'all', plot: bool = False):

    skipped = []
    methods, _skipped = _prepare_for_fitting(methods)
    skipped.extend(_skipped)

    if len(methods) == 0:
        print(f"Skipped fitting for the following methods: {', '.join(skipped)}.")
        raise ValueError("No methods to fit.")
    else:
        x_ = np.linspace(min(x), max(x), 100)
        rms_errors = []
        f_params = []
        for method in methods:
            try:
                f = _get_function_from_name(method)
                p0 = _get_function_p0(method, x)
                popt, _ = curve_fit(f, x, y, p0=p0, maxfev=15000)
                y_ = f(x_, *popt)
                f_params.append(popt)
                rmse = _rms_error(y, f(x, *popt))
                rms_errors.append(rmse)
                if plot:
                    plt.plot(x_, y_, label=method + f' (RMSE={rmse:.2f})')
            except Exception:
                skipped.append(method)
                warnings.warn(f"Failed to fit `{method}`. Skipped.", stacklevel=2)
        if plot:
            plt.scatter(x, y, c='r')
            plt.legend()
            plt.show()

        if len(skipped) > 0:
            print(f"Skipped fitting for the following methods: {', '.join(skipped)}.")
    f_best, p_best = _get_best_fit(methods, rms_errors, f_params)
    return _get_function_from_name(f_best), p_best


# ---------------------------------------------------------------------------- #
# --------------------------- Discrete functions ----------------------------- #
# ---------------------------------------------------------------------------- #

class DiscreteSuitFunction(SuitabilityFunction):

    def __init__(
            self,
            func_params: dict[str, int | float] | None = None
    ):
        self.func = discrete
        self.func_method = 'discrete'
        self.func_params = func_params


# ---------------------------------------------------------------------------- #
# ---------------------------- Utility functions ----------------------------- #
# ---------------------------------------------------------------------------- #

equations: dict[str, dict] = {}


def _get_function_from_name(name: str) -> callable:
    for _type, funcs in equations.items():
        if name in funcs:
            return funcs[name]
    raise ValueError(f"Equation `{name}` not implemented.")


def equation(type: str):
    """
    Register an equation in the `equations` mapping under the specified type.

    Parameters
    ----------
    type : str
        The type of equation to register.

    Returns
    -------
    decorator
        The decorator function.
    """

    def decorator(func: callable):
        if type not in equations:
            equations[type] = {}

        equations[type].update({func.__name__: func})
        return func
    return decorator


@equation('discrete')
def discrete(x, rules: dict[str | int, int | float]) -> float:
    return np.vectorize(rules.get, otypes=[np.float32])(x, np.nan)


@equation('sigmoid')
def logistic(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))


@equation('sigmoid')
def sigmoid(x):
    return logistic(x, 1, 0)


@equation('sigmoid')
def vetharaniam2022_eq3(x, a, b):
    return np.exp(a * (x - b)) / (1 + np.exp(a * (x - b)))


@equation('sigmoid')
def vetharaniam2022_eq5(x, a, b):
    return 1 / (1 + np.exp(a * (np.sqrt(x) - np.sqrt(b))))


@equation('gaussian')
def vetharaniam2024_eq8(x, a, b, c):
    return np.exp(-a * np.power(x - b, c))


@equation('gaussian')
def vetharaniam2024_eq10(x, a, b, c):
    return 2 / (1 + np.exp(a * np.power(np.power(x, c) - np.power(b, c), 2)))


def _rms_error(y_true, y_pred):
    diff = abs(y_true - y_pred)
    return np.sqrt(np.mean(diff**2))


def _get_best_fit(methods, rmse, params, verbose=True):
    best_fit = np.nanargmin(rmse)
    if verbose:
        print(f"""
Best fit: {methods[best_fit]}
RMSE: {rmse[best_fit]:.5f}
Params: a={params[best_fit][0]}, b={params[best_fit][1]}
""")
    return methods[best_fit], params[best_fit]
