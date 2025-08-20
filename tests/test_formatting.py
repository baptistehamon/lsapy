"""Tests for repr formatting functions."""

from __future__ import annotations

from lsapy import SuitabilityCriteria, SuitabilityFunction
from lsapy.core.formatting import sf_short_repr, summarize_criteria


class TestSummarizeCriteria:
    def test_output(self, criteria_gdd, growing_degree_days):
        res = summarize_criteria("criteria", criteria_gdd, 20, 80)
        expected_res = "    growing_degre...(w=3) climate vetharaniam2022_eq5(a=-0.55, b=1350) "
        assert res == expected_res
        # test without category and function
        sc = SuitabilityCriteria(
            name="test_criteria",
            indicator=growing_degree_days,
        )
        res = summarize_criteria("criteria", sc, 20, 80)
        expected_res = "    test_criteria   (w=1) "
        assert res == expected_res

    def test_col_width(self, criteria_gdd):
        # col_width=0
        res = summarize_criteria("criteria", criteria_gdd, col_width=0, max_width=80)
        expected_res = "    growing_degree_da...(w=3) climate vetharaniam2022_eq5(a=-0.55, b=1350) "
        assert res == expected_res
        # col_width=max_width
        res = summarize_criteria("criteria", criteria_gdd, col_width=50, max_width=80)
        expected_res = "    growing_degree_days                           (w=3) climate vetharaniam20..."
        assert res == expected_res

    def test_max_width(self, criteria_gdd):
        # default max_width
        res = summarize_criteria("criteria", criteria_gdd, col_width=20)
        expected_res = "    growing_degre...(w=3) climate vetharaniam2022_eq5(a=-0.55, b=1350) "
        assert res == expected_res


class TestRepr:
    def test_sf(self):
        # test repr for function without params
        sf = SuitabilityFunction(name="sigmoid")
        assert repr(sf) == "SuitabilityFunction(func=sigmoid)"
        # test short repr for function without params
        assert sf_short_repr(sf) == "sigmoid()"
