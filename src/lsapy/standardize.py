"""Standardization Functions Module."""

import operator
import warnings

import numpy as np
from scipy.optimize import curve_fit

from lsapy.core.functions import declare_equation, equations, get_function_from_name

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    warnings.warn("Matplotlib not found. Plotting functionality will be disabled.", stacklevel=2)

__all__ = [
    "boolean",
    "discrete",
    "logistic",
    "sigmoid",
    "vetharaniam2022_eq3",
    "vetharaniam2022_eq5",
    "vetharaniam2024_eq8",
    "vetharaniam2024_eq10",
    "fit",
]

FIT_TYPES = ["sigmoid", "gaussian"]  # only sigmoid and gaussian need to be fitted
binary_ops = {">": "gt", "<": "lt", ">=": "ge", "<=": "le", "==": "eq", "!=": "ne"}


@declare_equation("boolean")
def boolean(x, op: str, thresh: int | float, skipna: bool = True):
    """
    Boolean function.

    This function applies a boolean operation to the values based on a threshold.

    Parameters
    ----------
    x : any
        Input values.
    op : {">", "gt", "<", "lt", ">=", "ge", "<=", "le", "==", "eq", "!=", "ne"}
        Logical operator.
    thresh : any
        Threshold value.
    skipna : bool, optional
        Whether to skip NaN values. If True, NaN values in `x` will remain NaN in the output. Default is True.

    Returns
    -------
    np.ndarray
        Boolean mask of the operation.
    """
    if op in binary_ops:
        op = binary_ops[op]
    elif op in binary_ops.values():
        pass
    else:
        raise ValueError(f"Operator '{op}' not recognized.")

    res = getattr(operator, op)(x, thresh)
    if skipna:
        mask = np.isnan(x)
        return np.where(mask, np.nan, res)
    return res


@declare_equation("categorical")
def discrete(x, rules: dict[str | int, int | float]) -> np.ndarray:
    """
    Discrete function.

    This function maps input values to a set of rules that define the output values.

    Parameters
    ----------
    x : any
        Input values to map.
    rules : dict[str | int, int | float]
        Rules to map the input values to output values. The keys correspond to the input values and the
        values to its associated output values.

    Returns
    -------
    np.ndarray
        Mapped output values.
    """
    return np.vectorize(rules.get)(x, np.nan)


@declare_equation("sigmoid")
def logistic(x, a, b) -> np.ndarray:
    r"""
    Logistic function capped to 1.

    This function should be used on sigmoid-like suitability data. The function can be used on both
    positive and negative values of `x`, as well as on increasing and decreasing sigmoid-like data.

    Parameters
    ----------
    x : any
        Input values.
    a : float | int
        Steepness of the function parameter.
    b : float | int
        Value of the function's midpoint.

    Returns
    -------
    float
        Output values.

    Notes
    -----
    The function is defined as:

    .. math::

        f(x) = \frac{1}{1 + e^{-a(x - b)}}
    """
    return 1 / (1 + np.exp(-a * np.subtract(x, b)))


@declare_equation("sigmoid")
def sigmoid(x) -> np.ndarray:
    r"""
    Logistic sigmoid function.

    This function is a special case of the logistic function with `a=1` and `b=0`, thus can be used on both
    positive and negative values of `x`, however only for increasing sigmoid-like data.

    Parameters
    ----------
    x : any
        Input values.

    Returns
    -------
    float
        Output values.

    See Also
    --------
    :func:`logistic`

    Notes
    -----
    The sigmoid function is defined as:

    .. math::

        f(x) = \frac{1}{1 + e^{-x}}
    """
    return logistic(x, 1, 0)


@declare_equation("sigmoid", "VTR22_eq3")
def vetharaniam2022_eq3(x, a, b) -> np.ndarray:
    r"""
    Sigmoid like function.

    This function is equivalent to the logistic function and thus can be used on both positive and negative values of
    `x`, as well as on increasing and decreasing sigmoid-like data.

    Parameters
    ----------
    x : any
        Input values.
    a : float | int
        Steepness of the function parameter.
    b : float | int
        Value of the function's midpoint.

    Returns
    -------
    float
        Output values.

    See Also
    --------
    :func:`logistic`

    Notes
    -----
    Alternative name: `VTR22_eq3`.
    This function has been implemented to support reproductiblity of the original paper. However, as it is equivalent to
    the more commonly used `logistic` function, it is recommended to use the `logistic` function instead.
    This function is defined as:

    .. math::

        f(x) = \frac{e^{a(x - b)}}{1 + e^{a(x - b)}}

    References
    ----------
    :cite:cts:`vetharaniam_lsa_2022`
    """
    return np.exp(a * np.subtract(x, b)) / (1 + np.exp(a * np.subtract(x, b)))


@declare_equation("sigmoid", "VTR22_eq5")
def vetharaniam2022_eq5(x, a, b) -> np.ndarray:
    r"""
    Sigmoid like function.

    This function is a modified version of the logistic function that can for both increasing and decreasing
    sigmoid-like data, but only for positive values of `x`.

    Parameters
    ----------
    x : any
        Input values.
    a : float | int
        Steepness of the function parameter.
    b : float | int
        Value of the function's midpoint.

    Returns
    -------
    float
        Output values.

    Notes
    -----
    Alternative name: `VTR22_eq5`.
    The sigmoid like function is defined as:

    .. math::

        f(x) = \frac{1}{1 + e^{a(\sqrt{x} - \sqrt{b})}}

    References
    ----------
    :cite:cts:`vetharaniam_lsa_2022`
    """
    return 1 / (1 + np.exp(a * (np.sqrt(x) - np.sqrt(b))))


@declare_equation("gaussian", "VTR24_eq8")
def vetharaniam2024_eq8(x, a, b, c) -> np.ndarray:
    r"""
    Gaussian like function.

    This function should be used on Gaussian-like data, either positive or negative, and allows
    to have a plateau at around the midpoint.

    Parameters
    ----------
    x : any
        Input values to map.
    a : float | int
        Steepness of the function parameter. Should be a positive number.
    b : float | int
        Value of the function's midpoint.
    c : float | int
        Scaling parameter. Should be a even number. If negative, the function will be flipped.

    Returns
    -------
    float
        Suitability values.

    Notes
    -----
    Alternative name: `VTR24_eq8`.
    The Gaussian like function is defined as:

    .. math::

        f(x) = e^{-a(x - b)^c}

    References
    ----------
    :cite:cts:`vetharaniam_lsa_2024`
    """
    return np.exp(-a * np.power(np.subtract(x, b), c))


@declare_equation("gaussian", "VTR24_eq10")
def vetharaniam2024_eq10(x, a, b, c) -> np.ndarray:
    r"""
    Gaussian like function.

    This function should be used on Gaussian-like data and allows asymmetric distribution. However,
    it only works for positive values of `x`.

    Parameters
    ----------
    x : any
        Input values.
    a : float | int
        Steepness of the function parameter.
    b : float | int
        Value of the function's midpoint.
    c : float | int
        Scaling parameter. Should be a positive number.

    Returns
    -------
    float
        Output values.

    Notes
    -----
    Alternative name: `VTR24_eq10`.
    The Gaussian like function is defined as:

    .. math::

        f(x) = e^{-a(x^c - b^c)}

    References
    ----------
    :cite:cts:`vetharaniam_lsa_2024`
    """
    return 2 / (1 + np.exp(a * np.power(np.power(x, c) - np.power(b, c), 2)))


def fit(x, y=None, kind: str | list[str] = "all", plot: bool = False, verbose: bool = False):
    """
    Fit standardization functions to data.

    This function fits membership functions to the provided data. It helps to determine the best membership function
    to use on the data.

    Parameters
    ----------
    x : any
        Input values to fit the functions on.
    y : any, optional
        Target values to fit the functions. Should be the same length as `x`. If not provided,
        the default values are used (0, 0.25, 0.5, 0.75, 1).
    kind : str | list[str], optional
        List of functions or function types to fit. If '{TYPES}_like', all equations corresponding to the
        type are fitted (available types: 'sigmoid', 'gaussian'). If 'all', all available equations are fitted.
    plot : bool, optional
        Whether to plot the fitted functions. Default is False.
    verbose : bool, optional
        Whether to print the fitting results. Default is False.

    Returns
    -------
    tuple
        A tuple containing the best fitting function and its parameters.
    """
    if y is None:
        y = [0, 0.25, 0.5, 0.75, 1]
    y = np.array(y)
    functions, skipped = _check_fitting(kind)

    x_ = np.linspace(min(x), max(x), 100)
    rms_errors = []
    f_params = []
    for func in functions:
        try:
            f = get_function_from_name(func)
            p0 = _get_function_p0(func, x)
            popt, _ = curve_fit(f, x, y, p0=p0, maxfev=15000)
            y_ = f(x_, *popt)
            f_params.append(popt)
            rmse = _rmse(y, f(x, *popt))
            rms_errors.append(rmse)
            if plot:
                plt.plot(x_, y_, label=func + f" (RMSE={rmse:.2f})")
        except Exception:
            skipped.append(func)
            warnings.warn(f"Failed to fit `{func}`. Skipped.", stacklevel=2)

    if all([f in skipped for f in functions]):
        warnings.warn(f"No methods to fit. Skipping: {', '.join(skipped)}.", stacklevel=2)
        return None, None

    if plot and plt is not None:
        plt.scatter(x, y, c="r")
        plt.legend()

    f_best, p_best = _get_best_fit([m for m in functions if m not in skipped], rms_errors, f_params, verbose=verbose)
    return get_function_from_name(f_best), p_best


def _check_fitting(kind="all") -> tuple[list[str], list[str]]:
    _types = [t + "_like" for t in FIT_TYPES]
    _skipped = []

    if not isinstance(kind, str) and not isinstance(kind, list):
        raise ValueError(f"`kind` should be a str or a list of string. Got {type(kind)}")

    functions: list[str] = []
    if isinstance(kind, str):
        if kind == "all":
            for t in FIT_TYPES:
                functions.extend(equations[t].keys())
                kind = None
        else:
            kind = [kind]
    if kind is not None:
        for func in kind:
            if not isinstance(func, str):
                continue
            if func in _types:
                for f in equations[func.replace("_like", "")].keys():
                    if f not in functions:
                        functions.append(f)
            else:
                try:
                    get_function_from_name(func)
                    if func not in functions:
                        functions.append(func)
                except Exception:
                    _skipped.append(func)
                    warnings.warn(f"`{func}` not found in implemented equations. Skipped.", stacklevel=3)

    for f in ["sigmoid", "vetharaniam2024_eq8"]:
        if f in functions:
            functions.remove(f)
            _skipped.append(f)
            if f == "sigmoid":
                warnings.warn("No parameters to determine for `sigmoid`. Skipped.", stacklevel=3)
            if f == "vetharaniam2024_eq8":
                warnings.warn("Fitting does not support `vetharaniam2024_eq8`. Skipped.", stacklevel=3)

    if len(functions) == 0:
        raise ValueError("No functions to fit. Try to modify `kind` parameter.")
    return functions, _skipped


def _get_function_p0(func: str, x: np.ndarray) -> list[float]:
    if func in equations["sigmoid"]:
        return [1, np.median(x)]
    if func in equations["gaussian"]:
        return [1, np.median(x), 1]
    return []


def _rmse(y_true, y_pred) -> float:
    diff = abs(y_true - y_pred)
    return np.sqrt(np.mean(diff**2))


def _get_best_fit(functions, rmse, params, verbose=True) -> tuple[str, list[float]]:
    best_fit = np.nanargmin(rmse)
    if verbose:
        print(f"""
Best fit: {functions[best_fit]}
RMSE: {rmse[best_fit]:.5f}
Params: a={params[best_fit][0]}, b={params[best_fit][1]}
""")
    return functions[best_fit], params[best_fit]
