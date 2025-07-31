"""Tests for discrete, membership and suitability functions."""

from __future__ import annotations

import numpy as np
import pytest

import lsapy.functions as func
from lsapy.core.functions import _alt_names, equations  # noqa: PLC2701

EQUATIONS = [v for t in equations.values() for v in t.keys()]


class TestDiscrete:
    def test_simple(self):
        # str as key
        rules = {"a": 0, "b": 0.25, "c": 0.5, "d": 0.75, "e": 1}
        assert func.discrete("a", rules) == 0
        assert func.discrete("e", rules) == 1

        # int as key
        rules = {1: 0, 2: 0.25, 3: 0.5, 4: 0.75, 5: 1}
        assert func.discrete(1, rules) == 0
        assert func.discrete(5, rules) == 1

        # mixed types as key
        rules = {"a": 0, 1: 0.25, "c": 0.5, 4: 0.75, "e": 1}
        assert func.discrete("a", rules) == 0
        assert func.discrete(1, rules) == 0.25

    def test_nan(self):
        # key not in rules
        rules = {"a": 0, "b": 0.25, "c": 0.5, "d": 0.75, "e": 1}
        assert np.isnan(func.discrete("x", rules))
        assert np.isnan(func.discrete(10, rules))

        # empty rules
        rules = {}
        assert np.isnan(func.discrete("a", rules))
        assert np.isnan(func.discrete(1, rules))


class TestLogistic:
    def test_midpoint(self):
        # if x < b (midpoint), then logistic(x) < 0.5 for a > 0
        assert func.logistic(-0.5, 1, 0) < 0.5
        assert func.logistic(-0.5, -1, 0) > 0.5  # if a < 0
        # if x > b (midpoint), then logistic(x) > 0.5 for a > 0
        assert func.logistic(0.5, 1, 0) > 0.5
        assert func.logistic(0.5, -1, 0) < 0.5  # if a < 0
        # if x == b (midpoint), then logistic(x) == 0.5
        assert func.logistic(0, 1, 0) == 0.5

    def test_steepness(self):
        # if a = 0, then logistic(x) = 0.5
        assert func.logistic(0, 0, 0) == 0.5
        assert func.logistic(-10, 0, 25) == 0.5


class TestSigmoid:
    def test_simple(self):
        # if x < 0, then sigmoid(x) < 0.5
        assert func.sigmoid(-1) < 0.5
        # if x > 0, then sigmoid(x) > 0.5
        assert func.sigmoid(1) > 0.5
        # if x == 0, then sigmoid(x) == 0.5
        assert func.sigmoid(0) == 0.5

        # close to bounds
        np.testing.assert_allclose(func.sigmoid(10), 1, atol=0.001)
        np.testing.assert_allclose(func.sigmoid(-10), 0, atol=0.001)


class TestVetharaniam22Eq3:
    def test_midpoint(self):
        # if x < b (midpoint), then vetharaniam2022_eq3(x) < 0.5 for a > 0
        assert func.vetharaniam2022_eq3(-0.5, 1, 0) < 0.5
        assert func.vetharaniam2022_eq3(-0.5, -1, 0) > 0.5  # if a < 0
        # if x > b (midpoint), then vetharaniam2022_eq3(x) > 0.5 for a > 0
        assert func.vetharaniam2022_eq3(0.5, 1, 0) > 0.5
        assert func.vetharaniam2022_eq3(0.5, -1, 0) < 0.5  # if a < 0
        # if x == b (midpoint), then vetharaniam2022_eq3(x) == 0.5
        assert func.vetharaniam2022_eq3(0, 1, 0) == 0.5

    def test_steepness(self):
        # if a = 0, then vetharaniam2022_eq3(x) = 0.5
        assert func.vetharaniam2022_eq3(0, 0, 0) == 0.5
        assert func.vetharaniam2022_eq3(-10, 0, 25) == 0.5


class TestVetharaniam22Eq5:
    def test_negative(self):
        # if x < 0, vetharaniam2022_eq5(x) = nan for any a, b
        assert np.isnan(func.vetharaniam2022_eq5(-5, 1, 0))
        assert np.isnan(func.vetharaniam2022_eq5(-5, 0, 0))
        assert np.isnan(func.vetharaniam2022_eq5(-5, -1, 0))
        assert np.isnan(func.vetharaniam2022_eq5(-5, 1, 100))

    def test_midpoint(self):
        # if x < b (midpoint), then vetharaniam2022_eq5(x) > 0.5 for a > 0
        assert func.vetharaniam2022_eq5(2.5, 1, 5) > 0.5
        assert func.vetharaniam2022_eq5(2.5, -1, 5) < 0.5  # if a < 0
        # if x > b (midpoint), then vetharaniam2022_eq5(x) > 0.5 for a > 0
        assert func.vetharaniam2022_eq5(7.5, 1, 5) < 0.5
        assert func.vetharaniam2022_eq5(7.5, -1, 5) > 0.5  # if a < 0
        # if x == b (midpoint), then vetharaniam2022_eq5(x) == 0.5
        assert func.vetharaniam2022_eq5(5, 1, 5) == 0.5

    def test_steepness(self):
        # if a = 0, then vetharaniam2022_eq5(x) = 0.5
        assert func.vetharaniam2022_eq5(0, 0, 5) == 0.5
        assert func.vetharaniam2022_eq5(2.5, 0, 5) == 0.5


class TestVetharaniam24Eq8:
    def test_midpoint(self):
        # if x == b (midpoint), then vetharaniam2024_eq8(x) == 1 for any a, c if c > 0
        assert func.vetharaniam2024_eq8(5, 1, 5, 2) == 1
        assert func.vetharaniam2024_eq8(0, 2, 0, 3) == 1
        assert func.vetharaniam2024_eq8(-5, 0.5, -5, 1) == 1
        # for c < 0, then vetharaniam2024_eq8(x) == 0 if x == b
        assert func.vetharaniam2024_eq8(5.0, 1, 5, -2) == 0
        assert func.vetharaniam2024_eq8(0.0, 2, 0, -3) == 0
        assert func.vetharaniam2024_eq8(-5.0, 0.5, -5, -1) == 0

    def test_steepness(self):
        # increasing a should make the function decrease faster as |x-b| increases for a > 0
        val1 = func.vetharaniam2024_eq8(2, 0.5, 5, 2)  # x < b
        val2 = func.vetharaniam2024_eq8(2, 2, 5, 2)
        assert val2 < val1
        val1 = func.vetharaniam2024_eq8(7, 0.5, 5, 2)  # x > b
        val2 = func.vetharaniam2024_eq8(7, 2, 5, 2)
        assert val2 < val1
        # if a < 0, the function grows without bound as |x-b| increases
        val1 = func.vetharaniam2024_eq8(2, -0.5, 5, 2)
        val2 = func.vetharaniam2024_eq8(3, -0.5, 5, 2)
        assert val2 < val1
        # if a == 0, then vetharaniam2024_eq8(x) == 1
        assert func.vetharaniam2024_eq8(-100, 0, 0, 2) == 1
        assert func.vetharaniam2024_eq8(0, 0, 0, 2) == 1
        assert func.vetharaniam2024_eq8(100, 0, 0, 2) == 1

    def test_scaling(self):
        # if c == 0, then vetharaniam2024_eq8(x) == exp(-a)
        assert np.isclose(func.vetharaniam2024_eq8(5, 1, 5, 0), np.exp(-1))
        assert np.isclose(func.vetharaniam2024_eq8(0, 2, 0, 0), np.exp(-2))
        assert np.isclose(func.vetharaniam2024_eq8(-5, 0.5, -5, 0), np.exp(-0.5))
        # for c > 0, increasing c should make the function decrease slower as |x-b| increases
        # for vetharaniam2024_eq8(x) > exp(-a), else decrease faster
        x, a, b = 4.5, 1, 5
        val1 = func.vetharaniam2024_eq8(x, a, b, 2)
        val2 = func.vetharaniam2024_eq8(x, a, b, 4)
        if val1 > np.exp(-a):
            assert val2 > val1
        else:
            assert val2 < val1
        # for c < 0, inverse behavior
        val1 = func.vetharaniam2024_eq8(x, a, b, -2)
        val2 = func.vetharaniam2024_eq8(x, a, b, -4)
        if val1 < np.exp(-a):
            assert val2 < val1
        else:
            assert val2 > val1

    def test_symmetry(self):
        # the function should be symmetric around the midpoint b
        # The function should be symmetric around the midpoint b
        assert np.isclose(func.vetharaniam2024_eq8(2, 1, 5, 2), func.vetharaniam2024_eq8(8, 1, 5, 2))
        assert np.isclose(func.vetharaniam2024_eq8(-1, 0.5, -2, 2), func.vetharaniam2024_eq8(-3, 0.5, -2, 2))


class TestVetharaniam24Eq10:
    def test_midpoint(self):
        # if x == b (midpoint), then vetharaniam2024_eq10(x) == 1 for any a, c if c > 0
        assert func.vetharaniam2024_eq10(5, 1, 5, 2) == 1
        assert func.vetharaniam2024_eq10(0, 2, 0, 3) == 1
        assert func.vetharaniam2024_eq10(-5, 0.5, -5, 1) == 1

    def test_steepness(self):
        # increasing a should make the function decrease faster as |x-b| increases for a > 0
        val1 = func.vetharaniam2024_eq8(2, 0.5, 5, 2)  # x < b
        val2 = func.vetharaniam2024_eq8(2, 2, 5, 2)
        assert val2 < val1
        val1 = func.vetharaniam2024_eq8(7, 0.5, 5, 2)  # x > b
        val2 = func.vetharaniam2024_eq8(7, 2, 5, 2)
        assert val2 < val1
        # if a == 0, then vetharaniam2024_eq10(x) == 1
        assert func.vetharaniam2024_eq8(-100, 0, 0, 2) == 1
        assert func.vetharaniam2024_eq8(0, 0, 0, 2) == 1
        assert func.vetharaniam2024_eq8(100, 0, 0, 2) == 1

    def test_scaling(self):
        # if c == 0, then vetharaniam2024_eq8(x) = 1
        assert func.vetharaniam2024_eq10(5, 1, 5, 0) == 1
        assert func.vetharaniam2024_eq10(0, 2, 0, 0) == 1
        assert func.vetharaniam2024_eq10(-5, 0.5, -5, 0) == 1
        # for c > 0, increasing c should make the function decrease faster as |x-b| increases
        x, a, b = 4.5, 0.2, 5
        val1 = func.vetharaniam2024_eq10(x, a, b, 2)
        val2 = func.vetharaniam2024_eq10(x, a, b, 4)
        assert val2 < val1


class TestSuitabilityFunction:
    def test_names(self):
        # test equations names
        for name in EQUATIONS:
            sf = func.SuitabilityFunction(name=name)
            assert callable(sf.func), f"Function {name} is not callable"
            assert sf.func.__name__ == name, f"Function name mismatch for {name}"
        # test alternative names
        for k, v in _alt_names.items():
            sf = func.SuitabilityFunction(name=k)
            assert sf.func.__name__ == v, f"wrong function returned for {k} alternative name"

    def test_invalid_function(self):
        for f in [1, "string", {}]:
            with pytest.raises(TypeError):
                func.SuitabilityFunction(func=f)

    def test_callable(self):
        sf = func.SuitabilityFunction(name="discrete", params={"rules": {1: 0, 2: 0.1, 3: 0.5, 4: 0.9, 5: 1}})
        assert sf(3) == 0.5
        sf = func.SuitabilityFunction(name="logistic", params={"a": 1, "b": 5})
        assert sf(5) == 0.5
