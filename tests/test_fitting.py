"""Tests for the functions related to fitting."""

from __future__ import annotations

import numpy as np
import pytest

import lsapy.standardize as std


class TestRMSE:
    def test_zero(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        assert std._rmse(y_true, y_pred) == 0.0

    def test_non_zero(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])
        assert std._rmse(y_true, y_pred) == 1.0
        y_pred = np.array([11, 12, 13, 14, 15])
        assert std._rmse(y_true, y_pred) == 10.0

    def test_nan(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([np.nan, 2, 3, 4, 5])
        assert np.isnan(std._rmse(y_true, y_pred))


class TestCheckFitting:
    def test_all(self):
        # should skip sigmoid and vetharaniam2024_eq8
        valid, skipped = std._check_fitting()
        assert valid == [
            "logistic",
            "vetharaniam2022_eq3",
            "vetharaniam2022_eq5",
            "vetharaniam2024_eq10",
        ]
        assert skipped == ["sigmoid", "vetharaniam2024_eq8"]

    def test_type_like(self):
        valid, skipped = std._check_fitting("sigmoid_like")
        assert valid == ["logistic", "vetharaniam2022_eq3", "vetharaniam2022_eq5"]
        assert skipped == ["sigmoid"]

    def test_list(self):
        # should not str type in list
        valid, skipped = std._check_fitting(["logistic", 1])
        assert valid == ["logistic"]
        assert skipped == []

    def test_invalid(self):
        # test for invalid type
        with pytest.raises(ValueError, match="`kind` should be a str or a list of string. Got <class 'int'>"):
            std._check_fitting(1)
        # test for invalid name
        with pytest.raises(ValueError, match="No functions to fit. Try to modify `kind` parameter."):
            std._check_fitting("invalid")


class TestGetFunctionP0:
    def test_sigmoid(self):
        p0 = std._get_function_p0("vetharaniam2022_eq5", np.array([1, 2, 3, 4, 5]))
        assert p0 == [1, 3]

    def test_gaussian(self):
        p0 = std._get_function_p0("vetharaniam2024_eq10", np.array([1, 2, 3, 4, 5]))
        assert p0 == [1, 3, 1]

    def test_invalid(self):
        p0 = std._get_function_p0("invalid", np.array([1, 2, 3, 4, 5]))
        assert p0 == []


class TestGetBestFit:
    def test_simple(self):
        functions = ["logistic", "vetharaniam2022_eq3", "vetharaniam2022_eq5"]
        rmse = np.array([3, 1, 2])
        params = [(1, 1), (2, 1), (3, 1)]
        func, param = std._get_best_fit(functions, rmse, params)
        assert func == "vetharaniam2022_eq3"
        assert param == (2, 1)

    def test_nan(self):
        functions = ["logistic", "vetharaniam2022_eq3", "vetharaniam2022_eq5"]
        rmse = np.array([3, np.nan, 2])
        params = [(1, 1), np.nan, (3, 1)]
        f, p = std._get_best_fit(functions, rmse, params)
        assert f == "vetharaniam2022_eq5"
        assert p == (3, 1)


class TestFit:
    def test_simple(self):
        # test simple cas with plotting
        x = [-10, -6.5, -5, -3.5, 0]
        f, p = std.fit(x, kind="all", plot=True)
        assert f == std.logistic
        np.testing.assert_array_almost_equal(p, [0.76, -5.0], decimal=2)

    def test_skipped(self):
        with pytest.warns(UserWarning, match="No methods to fit. Skipping: vetharaniam2024_eq8, vetharaniam2024_eq10."):
            std.fit([-10, -6.5, -5, -3.5, 0], kind="gaussian_like")
