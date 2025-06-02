"""Membership Suitability Function definition."""

import warnings
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from lsapy.core.functions import SuitabilityFunction

__all__ = [
    "MembershipFunction",
    "logistic",
    "sigmoid",
    "vetharaniam2022_eq3",
    "vetharaniam2022_eq5",
    "vetharaniam2024_eq8",
    "vetharaniam2024_eq10",
]


class MembershipFunction(SuitabilityFunction):
    """
    Membership Suitability Function.

    Membership functions are used to transform continuous indicator values into suitability values.
    The membership converts the indicator values into a suitability value between 0 and 1.

    Parameters
    ----------
    func : Callable | None, optional
        Function to compute the suitability value.
    name : str | None, optional
        Name of the function to compute the suitability value. By default, the name of the function is used if provided,
        otherwise it is set to `None`.
    params : dict[str, int | float] | None, optional
        Parameters of the function.

    See Also
    --------
    DiscreteFunction : Discrete Suitability Function.

    Examples
    --------
    >>> mf = MembershipFunction(name="logistic", params={"a": 1, "b": 5})
    >>> mf(3)
    array(0.11920292, dtype=float32)
    """

    def __init__(
        self,
        func: Callable | None = None,
        name: str | None = None,
        params: dict[str, int | float] | None = None,
    ):
        super().__init__(func, name, params)
        if func is None and name is not None:
            try:
                self.func = _get_function_from_name(name)
            except Exception:
                warnings.warn("`name` not found in implemented equations. Setting `func` to None.", stacklevel=2)

    @staticmethod
    def fit_functions(x, y=None, methods: str | list[str] = "all", plot: bool = False):
        """
        Fit the membership functions to data.

        This method help to identify the best membership function to use on the data by fitting
        the available functions.
        # TODO: check if results should be print or return

        Parameters
        ----------
        x : any
            Input values to fit the functions on.
        y : any, optional
            Target suitability values to fit the functions. Should be the same length as `x`. If not provided,
            the default values are used (0, 0.25, 0.5, 0.75, 1).
        methods : str | list[str], optional
            List of methods to fit. If 'all', all available methods are fitted. If a list of methods, only the specified
            methods are fitted. Default is 'all'.
        plot : bool, optional
            Whether to plot the fitted functions. Default is False.

        Returns
        -------
        tuple
            A tuple containing the best fitting function and its parameters.

        Examples
        --------
        >>> MembershipFunction.fit_functions([1, 3, 5, 7, 10])  # doctest: +SKIP
        Skipped fitting for the following methods: sigmoid, vetharaniam2024_eq8.
        <BLANKLINE>
        Best fit: logistic
        RMSE: 0.04863
        Params: a=0.6772100495121773, b=4.999999998691947
        <BLANKLINE>
        (<function logistic at 0x0000015722C73C40>, array([0.67721005, 5.        ]))

        By default, the function will fit all available methods. If you want to fit only specific methods, you can
        specify the methods to fit: "all", "sigmoid_like", "gaussian_like", or a list of methods.

        >>> MembershipFunction.fit_functions(
        ...     x=[1, 3, 5, 5, 7, 9], y=[0, 0.5, 1, 1, 0.5, 0], methods="gaussian_like"
        ... )  # doctest: +SKIP
        Skipped fitting for the following methods: vetharaniam2024_eq8.
        <BLANKLINE>
        Best fit: vetharaniam2024_eq10
        RMSE: 0.01329
        Params: a=0.38213218843552715, b=4.972731378762913
        <BLANKLINE>
        (<function vetharaniam2024_eq10 at 0x0000015722C73F60>, array([0.38213219, 4.97273138, 0.93922462]))
        """
        if y is None:
            y = [0, 0.25, 0.5, 0.75, 1]
        return _fit_membership_func(x, np.array(y), methods, plot)


def _prepare_for_fitting(methods: str | list[str] = "all"):
    _types = ["sigmoid_like", "gaussian_like"]
    _skipped = []

    if methods == "all":
        methods = [f for t in _types for f in equations[t.replace("_like", "")]]
    elif isinstance(methods, list) or isinstance(methods, str):
        if isinstance(methods, str):
            methods = [methods]

        _methods = []
        for method in methods:
            if method in _types:
                [_methods.append(m) for m in equations[method.replace("_like", "")].keys()]
            else:
                try:
                    _get_function_from_name(method)
                    _methods.append(method)
                except Exception:
                    _skipped.append(method)
                    warnings.warn(f"`{method}` not found in equations. Skipped.", stacklevel=2)
        methods = _methods
        for m in ["sigmoid", "vetharaniam2024_eq8"]:
            if m in methods:
                methods.remove(m)
                _skipped.append(m)
                if m == "sigmoid":
                    warnings.warn("No parameters to determine for `sigmoid`. Skipped.", stacklevel=2)
                if m == "vetharaniam2024_eq8":
                    warnings.warn("Fitting method does not support `vetharaniam2024_eq8`. Skipped.", stacklevel=2)
    return methods, _skipped


def _get_function_p0(method: str, x: np.ndarray) -> list[float]:
    if method in equations["sigmoid"]:
        return [1, np.median(x)]
    if method in equations["gaussian"]:
        return [1, np.median(x), 1]
    return []


def _fit_membership_func(x, y, methods: str | list[str] = "all", plot: bool = False):
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
                    plt.plot(x_, y_, label=method + f" (RMSE={rmse:.2f})")
            except Exception:
                skipped.append(method)
                warnings.warn(f"Failed to fit `{method}`. Skipped.", stacklevel=2)
        if plot:
            plt.scatter(x, y, c="r")
            plt.legend()
            plt.show()

        if len(skipped) > 0:
            print(f"Skipped fitting for the following methods: {', '.join(skipped)}.")
    f_best, p_best = _get_best_fit([m for m in methods if m not in skipped], rms_errors, f_params)
    return _get_function_from_name(f_best), p_best


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

    def _decorator(func: callable):
        if type not in equations:
            equations[type] = {}

        equations[type].update({func.__name__: func})
        return func

    return _decorator


@equation("sigmoid")
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


@equation("sigmoid")
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


@equation("sigmoid")
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


@equation("sigmoid")
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


@equation("gaussian")
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


@equation("gaussian")
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
