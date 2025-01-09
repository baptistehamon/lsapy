"Suitability function definitions"

from typing import Optional, Callable, Union, Any
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit


class SuitabilityFunction:

    def __init__(
            self,
            func: Optional[Callable] = None,
            func_method: Optional[str] = None,
            func_params: Union[dict[str, Any]] = {}
    ):
        if func_params is not None:
            if func is None and func_method is None:
                raise ValueError("If 'func_params' is provided, 'func' or 'func_method' must also be provided.")

        self.func = func
        self._func_method = func_method
        self._func_params = func_params
        if func is None and func_method is not None:
            self.func = _get_function_from_name(func_method)
    

    def __str__(self):
        params_str = ', '.join([f'{k}={v}' for k, v in self.func_params.items()])
        return f'{self.func_method}(x, {params_str})'
    
    
    def __repr__(self):
        return f"SuitabilityFunction(func={self.func}, func_method='{self.func_method}', func_params={self.func_params})"
    

    def __call__(self, x):
        if self.func is None:
            raise ValueError("No function has been provided.")
        return self.func(x, **self.func_params)
    
    
    def map(self, x):
        return self(x)
    
    
    @property
    def func_method(self):
        if self._func_method is None:
            return ''
        return self._func_method
    
    
    @func_method.setter
    def func_method(self, value: str):
        self._func_method = value if value else None

    
    @property
    def func_params(self):
        if self._func_params is None:
            return {}
        return self._func_params
    
    
    @func_params.setter
    def func_params(self, value: dict[str, Any]):
        self._func_params = value if value else None


    @property
    def attrs(self):
        if self.func_method == '' and self.func_params == {}:
            return {}
        return {'func_method': self.func_method, 'func_params': self.func_params}
    
    
    @attrs.setter
    def attrs(self, value: dict[str, Any]):
        self.func_method = value.get('func_method', '')
        self.func_params = value.get('func_params', {})

    
    def plot(self, x) -> None:
        plt.plot(x, self(x))

# ---------------------------------------------------------------------------- #
# ------------------------ Membership functions ------------------------------ #
# ---------------------------------------------------------------------------- #
_MEMBERSHIP_FUNCTIONS = [
    'logistic',
    'vetharaniam2022_eq3',
    'vetharaniam2022_eq5'
]

class MembershipSuitFunction(SuitabilityFunction):
    _methods = _MEMBERSHIP_FUNCTIONS

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


def logistic(x, a, b):
    return 1 / (1 + np.exp(-a*(x - b)))

def sigmoid(x):
    return logistic(x, 1, 0)

def logistic_vetharaniam2022_eq3(x, a, b):
    return np.exp(a * (x - b)) / (1 + np.exp(a * (x - b)))

def logistic_vetharaniam2022_eq5(x, a, b):
    return 1 / (1 + np.exp(a * (np.sqrt(x) - np.sqrt(b))))


# TODO: Check if a general logistic working with positive and negative values exists
# def _general_logistic(x, c, d, m):
#     return 1 / (np.power(1 + np.power(x/c, d), m))

# TODO: Add a general normal distribution function


def _get_mbs_function_from_name(name: str) -> Callable:
    if name.lower() not in _MEMBERSHIP_FUNCTIONS:
        raise ValueError(f"Method must be one of {_MEMBERSHIP_FUNCTIONS}. Got {name}")

    if name.lower() == 'logistic':
        return logistic
    elif name.lower() == 'vetharaniam2022_eq3':
        return logistic_vetharaniam2022_eq3
    elif name.lower() == 'vetharaniam2022_eq5':
        return logistic_vetharaniam2022_eq5


def _fit_mbs_functions(x, y, methods: str | list[str] = 'all', plot: bool = False):
    if methods == 'all':
        methods = _MEMBERSHIP_FUNCTIONS
    elif isinstance(methods, str):
        methods = [methods]
    else:
        raise ValueError(f"'methods' must be a string or a list of strings. Got {methods}")
    
    x_ = np.linspace(min(x), max(x), 100)
    rms_errors = []
    f_params = []
    for method in methods:
        f = _get_mbs_function_from_name(method)
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
    return _get_mbs_function_from_name(f_best), p_best


# ---------------------------------------------------------------------------- #
# --------------------------- Discrete functions ----------------------------- #
# ---------------------------------------------------------------------------- #
_DISCRETE_FUNCTIONS = [
    'discrete'
]

class DiscreteSuitFunction(SuitabilityFunction):
    _methods = _DISCRETE_FUNCTIONS

    def __init__(
            self,
            func: Optional[Callable] = None,
            func_method: Optional[str] = None,
            func_params: Optional[dict[str, int | float]] = None
    ):
        super().__init__(func, func_method, func_params)


def discrete(x, rules: dict[str|int, int|float]) -> float:
    return rules.get(x, 9999) # 9999 as default to avoid dtype issues


def _get_discrete_function_from_name(name: str) -> Callable:
    if name.lower() not in _DISCRETE_FUNCTIONS:
        raise ValueError(f"Method must be one of {_DISCRETE_FUNCTIONS}. Got {name}")

    if name.lower() == 'discrete':
        return discrete

# ---------------------------------------------------------------------------- #
# ---------------------------- Utility functions ----------------------------- #
# ---------------------------------------------------------------------------- #
_FUNCTIONS = _MEMBERSHIP_FUNCTIONS + _DISCRETE_FUNCTIONS

def _get_function_from_name(name: str):
    if name.lower() not in _FUNCTIONS:
        raise ValueError(f"Method must be one of {_FUNCTIONS}. Got {name}")
    
    if name.lower() in _MEMBERSHIP_FUNCTIONS:
        return _get_mbs_function_from_name(name)
    elif name.lower() in _DISCRETE_FUNCTIONS:
        return _get_discrete_function_from_name(name)


def _rms_error(y_true, y_pred):
    diff = abs(y_true - y_pred)
    return np.sqrt(np.mean(diff**2))


def _get_best_fit(methods, rmse, params, verbose=True):
    best_fit = np.argmin(rmse)
    if verbose:
        print(f"""
Best fit: {methods[best_fit]}
RMSE: {rmse[best_fit]:.5f}
Params: a={params[best_fit][0]}, b={params[best_fit][1]}
""")
    return methods[best_fit], params[best_fit]
