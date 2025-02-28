"""Suitability Functions definitions."""

from typing import Optional, Callable, Union, Any
import warnings
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit


__all__ = [
    "SuitabilityFunction",
    "MembershipSuitFunction",
    "DiscreteSuitFunction"
]


class SuitabilityFunction:

    def __init__(
            self,
            func: Optional[Callable] = None,
            func_method: Optional[str] = None,
            func_params: Union[dict[str, Any]] = {}
    ):
        if func_params is not None:
            if func is None and func_method is None:
                raise ValueError("If `func_params` is provided, `func` or `func_method` must also be provided.")

        self.func = func
        self.func_method = func_method
        self.func_params = func_params
        if func is None and func_method is not None:
            self.func = _get_from_equations(func_method)
            # try:
            #     self.func = _get_from_equations(func_method)
            # except ValueError as e:
            #     raise ValueError(f"Error in initializing function from method '{func_method}': {e}")
    

    def __repr__(self):
        return f"{self.__class__.__name__}(func={self.func.__name__}, func_method='{self.func_method}', func_params={self.func_params})"
    

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
        if self.func_method == None and self.func_params == None:
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
            func: Optional[Callable] = None,
            func_method: Optional[str] = None,
            func_params: Optional[dict[str, int | float]] = None
    ):
        super().__init__(func, func_method, func_params)


    @staticmethod
    def fit(x, y = np.array([0,.25,.5,.75,1]), methods: str | list[str] = 'all', plot: bool = False):
        return _fit_mbs_functions(x, y, methods, plot)


def _fit_mbs_functions(x, y, methods: str | list[str] = 'all', plot: bool = False):
    _types = ['sigmoid', 'gaussian']

    if methods == 'all':
        methods = [f for t in _types for f in equations[t]]
    elif isinstance(methods, list) or isinstance(methods, str):
        if isinstance(methods, str):
            methods = [methods]
        
        _methods = []
        for method in methods:
            if method in _types:
                [_methods.append(m) for m in equations[method].keys()]
            else:
                try:
                    _get_from_equations(method)
                    _methods.append(method)
                except :
                    warnings.warn(f"`{method}` not found in equations. Skipped.")
        methods = _methods
    
    else:
        raise ValueError(f"'methods' must be a string or a list of strings. Got `{type(methods)}`.")
    
    x_ = np.linspace(min(x), max(x), 100)
    rms_errors = []
    f_params = []
    for method in methods:
        f = _get_from_equations(method)
        popt, _ = curve_fit(f, x, y, p0=[1, np.median(x)], maxfev=15000)
        y_ = f(x_, *popt)
        f_params.append(popt)
        rmse = _rms_error(y,f(x, *popt))
        rms_errors.append(rmse)
        if plot:
            plt.plot(x_, y_, label=method + f' (RMSE={rmse:.2f})')
    if plot:
        plt.scatter(x, y, c='r')
        plt.legend()
        plt.show()
    
    f_best, p_best = _get_best_fit(methods, rms_errors, f_params)
    return _get_from_equations(f_best), p_best


# ---------------------------------------------------------------------------- #
# --------------------------- Discrete functions ----------------------------- #
# ---------------------------------------------------------------------------- #

class DiscreteSuitFunction(SuitabilityFunction):

    def __init__(
            self,
            func_params: Optional[dict[str, int | float]] = None
    ):
        self.func = discrete
        self.func_method = 'discrete'
        self.func_params = func_params


# ---------------------------------------------------------------------------- #
# ---------------------------- Utility functions ----------------------------- #
# ---------------------------------------------------------------------------- #

equations : dict[str, dict] = {}


def _get_from_equations(name: str) -> callable:
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
def discrete(x, rules: dict[str|int, int|float]) -> float:
    return np.vectorize(rules.get, otypes=[np.float32])(x, np.nan)


@equation('sigmoid')
def logistic(x, a, b):
    return 1 / (1 + np.exp(-a*(x - b)))


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


@equation('discrete')
def discrete(x, rules: dict[str|int, int|float]) -> float:
    return np.vectorize(rules.get, otypes=[np.float32])(x, np.nan)


# TODO: Check if a general logistic working with positive and negative values exists
# def _general_logistic(x, c, d, m):
#     return 1 / (np.power(1 + np.power(x/c, d), m))


def _rms_error(y_true, y_pred):
    diff = abs(y_true - y_pred)
    return np.sqrt(np.mean(diff**2))


def _get_best_fit(methods, rmse, params, verbose=True):
    best_fit = np.argmin(rmse) #TODO: fix when nan in rmse
    if verbose:
        print(f"""
Best fit: {methods[best_fit]}
RMSE: {rmse[best_fit]:.5f}
Params: a={params[best_fit][0]}, b={params[best_fit][1]}
""")
    return methods[best_fit], params[best_fit]
