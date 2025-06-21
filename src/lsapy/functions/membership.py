"""Membership Function Module."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from lsapy.core.functions import declare_equation, equations, get_function_from_name

__all__ = [
    "fit_membership",
    "logistic",
    "sigmoid",
    "vetharaniam2022_eq3",
    "vetharaniam2022_eq5",
    "vetharaniam2024_eq8",
    "vetharaniam2024_eq10",
]

EQUATION_TYPES = ["sigmoid", "gaussian"]


@declare_equation("sigmoid")
def logistic(x, a, b):
    r"""
    Logistic function.

    Parameters
    ----------
    x : any
        Input values to map.
    a : float | int
        Steepness of the function parameter.
    b : float | int
        Value of the function's midpoint.

    Returns
    -------
    float
        Suitability values.

    Notes
    -----
    The logistic function is defined as:

    .. math::

        f(x) = \frac{1}{1 + e^{-a(x - b)}}
    """
    return 1 / (1 + np.exp(-a * (x - b)))


@declare_equation("sigmoid")
def sigmoid(x):
    r"""
    Sigmoid function.

    Parameters
    ----------
    x : any
        Input values to map.

    Returns
    -------
    float
        Suitability values.

    Notes
    -----
    The sigmoid function is defined as:

    .. math::

        f(x) = \frac{1}{1 + e^{-x}}
    """
    return logistic(x, 1, 0)


@declare_equation("sigmoid", "VTR22_3")
def vetharaniam2022_eq3(x, a, b):
    r"""
    Sigmoid like function.

    # TODO: add a more detailed description.

    Parameters
    ----------
    x : any
        Input values to map.
    a : float | int
        Steepness of the function parameter.
    b : float | int
        Value of the function's midpoint.

    Returns
    -------
    float
        Suitability values.

    Notes
    -----
    The sigmoid like function is defined as:

    .. math::

        f(x) = \frac{e^{a(x - b)}}{1 + e^{a(x - b)}}

    References
    ----------
    :cite:cts:`vetharaniam_lsa_2022`
    """
    return np.exp(a * (x - b)) / (1 + np.exp(a * (x - b)))


@declare_equation("sigmoid", "VTR22_5")
def vetharaniam2022_eq5(x, a, b):
    r"""
    Sigmoid like function.

    # TODO: add a more detailed description.

    Parameters
    ----------
    x : any
        Input values to map.
    a : float | int
        Steepness of the function parameter.
    b : float | int
        Value of the function's midpoint.

    Returns
    -------
    float
        Suitability values.

    Notes
    -----
    The sigmoid like function is defined as:

    .. math::

        f(x) = \frac{1}{1 + e^{a(\sqrt{x} - \sqrt{b})}}

    References
    ----------
    :cite:cts:`vetharaniam_lsa_2022`
    """
    return 1 / (1 + np.exp(a * (np.sqrt(x) - np.sqrt(b))))


@declare_equation("gaussian", "VTR24_8")
def vetharaniam2024_eq8(x, a, b, c):
    r"""
    Gaussian like function.

    # TODO: add a more detailed description.

    Parameters
    ----------
    x : any
        Input values to map.
    a : float | int
        Steepness of the function parameter.
    b : float | int
        Value of the function's midpoint.
    c : float | int
        Scaling parameter.

    Returns
    -------
    float
        Suitability values.

    Notes
    -----
    The Gaussian like function is defined as:

    .. math::

        f(x) = e^{-a(x - b)^c}

    References
    ----------
    :cite:cts:`vetharaniam_lsa_2024`
    """
    return np.exp(-a * np.power(x - b, c))


@declare_equation("gaussian", "VTR24_10")
def vetharaniam2024_eq10(x, a, b, c):
    r"""
    Gaussian like function.

    # TODO: add a more detailed description.

    Parameters
    ----------
    x : any
        Input values to map.
    a : float | int
        Steepness of the function parameter.
    b : float | int
        Value of the function's midpoint.
    c : float | int
        Scaling parameter.

    Returns
    -------
    float
        Suitability values.

    Notes
    -----
    The Gaussian like function is defined as:

    .. math::

        f(x) = e^{-a(x^c - b^c)}

    References
    ----------
    :cite:cts:`vetharaniam_lsa_2024`
    """
    return 2 / (1 + np.exp(a * np.power(np.power(x, c) - np.power(b, c), 2)))


def fit_membership(x, y=None, fit_on: str | list[str] = "all", plot: bool = False, verbose: bool = False):
    """
    Fit membership function to data.

    This function fits membership functions to the provided data. It helps to determine the best membership function
    to use on the data.

    Parameters
    ----------
    x : any
        Input values to fit the functions on.
    y : any, optional
        Target suitability values to fit the functions. Should be the same length as `x`. If not provided,
        the default values are used (0, 0.25, 0.5, 0.75, 1).
    fit_on : str | list[str], optional
        List of equation or equation types to fit. `all, `sigmoid_like` and `gaussian_like` can also be used.
        If 'all', all available equations are fitted. If '{TYPES}_like', all equations corresponding to the
        type are fitted. Default is 'all'.
    plot : bool, optional
        Whether to plot the fitted functions. Default is False.
    verbose : bool, optional
        Whether to print the fitting results. Default is False.

    Returns
    -------
    tuple
        A tuple containing the best fitting function and its parameters.

    Examples
    --------
    >>> from lsapy.functions.membership import fit_membership
    >>> fit_membership([1, 3, 5, 7, 10])  # doctest: +ELLIPSIS
    (<function vetharaniam2022_eq5 at 0x...>, array([-2.78959209,  4.86485647]))

    By default, the function will fit all available membership equations. If you want to fit only specific equations,
    you can specify it using the `fit_on` parameter: "all", "sigmoid_like", "gaussian_like", or a list of equations.
    The default `y` values can also be changed.

    >>> fit_membership(x=[1, 3, 5, 5, 7, 9], y=[0, 0.5, 1, 1, 0.5, 0], fit_on="gaussian_like")  # doctest: +ELLIPSIS
    (<function vetharaniam2024_eq10 at 0x...>, array([0.38213219, 4.97273138, 0.93922461]))
    """
    if y is None:
        y = [0, 0.25, 0.5, 0.75, 1]
    y = np.array(y)
    functions, skipped = _check_fitting(fit_on)

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
        warnings.warn(f"No methods to fit. fitting for the following methods: {', '.join(skipped)}.", stacklevel=2)
        return None, None

    if plot:
        plt.scatter(x, y, c="r")
        plt.legend()
        plt.show()

    f_best, p_best = _get_best_fit([m for m in functions if m not in skipped], rms_errors, f_params, verbose=verbose)
    return get_function_from_name(f_best), p_best


def _check_fitting(fit_on: str | list[str] = "all"):
    _types = [t + "_like" for t in EQUATION_TYPES]
    _skipped = []

    if not isinstance(fit_on, str) and not isinstance(fit_on, list):
        raise ValueError(f"`fit_on` should be a str or a list of string. Got {type(fit_on)}")

    functions = []
    if isinstance(fit_on, str):
        if fit_on == "all":
            for t in EQUATION_TYPES:
                functions.extend(equations[t].keys())
                fit_on = None
        else:
            fit_on = [fit_on]
    if fit_on is not None:
        for func in fit_on:
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
            raise ValueError("No functions to fit. Try to modify `fit_on` parameter.")
        return functions, _skipped


def _get_function_p0(func: str, x: np.ndarray) -> list[float]:
    if func in equations["sigmoid"]:
        return [1, np.median(x)]
    if func in equations["gaussian"]:
        return [1, np.median(x), 1]
    return []


def _rmse(y_true, y_pred):
    diff = abs(y_true - y_pred)
    return np.sqrt(np.mean(diff**2))


def _get_best_fit(functions, rmse, params, verbose=True):
    best_fit = np.nanargmin(rmse)
    if verbose:
        print(f"""
Best fit: {functions[best_fit]}
RMSE: {rmse[best_fit]:.5f}
Params: a={params[best_fit][0]}, b={params[best_fit][1]}
""")
    return functions[best_fit], params[best_fit]
