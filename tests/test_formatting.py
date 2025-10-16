"""Tests for repr formatting functions."""

from __future__ import annotations

from functools import partial

import lsapy.standardize as std
from lsapy import SuitabilityCriteria, SuitabilityFunction
from lsapy.core.formatting import sc_func_repr, summarize_criteria


class TestSummarizeCriteria:
    def test_output(self, criteria_gdd, growing_degree_days):
        res = summarize_criteria("criteria", criteria_gdd, 20, 80)
        expected_res = "    growing_degre...(w=3.0) climate vetharaniam2022_eq5(a=-0.55, b=1350) "
        assert res == expected_res
        # test without category and function
        sc = SuitabilityCriteria(
            name="test_criteria",
            indicator=growing_degree_days,
        )
        res = summarize_criteria("criteria", sc, 20, 80)
        expected_res = "    test_criteria   (w=1.0) "
        assert res == expected_res

    def test_col_width(self, criteria_gdd):
        # col_width=0
        res = summarize_criteria("criteria", criteria_gdd, col_width=0, max_width=80)
        expected_res = "    growing_degree_da...(w=3.0) climate vetharaniam2022_eq5(a=-0.55, b=1350) "
        assert res == expected_res
        # col_width=max_width
        res = summarize_criteria("criteria", criteria_gdd, col_width=50, max_width=80)
        expected_res = "    growing_degree_days                           (w=3.0) climate vetharaniam..."
        assert res == expected_res

    def test_max_width(self, criteria_gdd):
        # default max_width
        res = summarize_criteria("criteria", criteria_gdd, col_width=20)
        expected_res = "    growing_degre...(w=3.0) climate vetharaniam2022_eq5(a=-0.55, b=1350) "
        assert res == expected_res


class TestRepr:
    def test_sf(self, sf_anpr):
        # test repr for function without params
        sf = SuitabilityFunction(name="sigmoid")
        assert repr(sf) == "sigmoid()"
        # test repr for function with params
        assert repr(sf_anpr) == "vetharaniam2022_eq5(a=-0.71, b=1100)"

    def test_sc_func(self):
        sc = SuitabilityCriteria("test", func="logistic")
        assert sc_func_repr(sc.func) == repr(std.logistic)
        sc.func = partial(std.logistic)
        assert sc_func_repr(sc.func) == "logistic()"
