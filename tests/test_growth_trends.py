"""Tests for financial growth trends calculation."""

from koreaalpha_core.stock.fundamental import calculate_growth_trends


class TestGrowthTrends:
    def test_basic(self):
        data = [
            {"year": 2022, "income_statement": {"revenue": 100, "operating_income": 20, "net_income": 15}, "ratios": {"roe": 0.10}},
            {"year": 2023, "income_statement": {"revenue": 120, "operating_income": 25, "net_income": 18}, "ratios": {"roe": 0.12}},
            {"year": 2024, "income_statement": {"revenue": 150, "operating_income": 30, "net_income": 22}, "ratios": {"roe": 0.14}},
        ]
        result = calculate_growth_trends(data)
        assert len(result["revenue"]) == 3
        assert result["revenue"][0]["year"] == 2022
        assert result["revenue_cagr"] is not None
        assert result["revenue_cagr"] > 0

    def test_single_year(self):
        data = [{"year": 2024, "income_statement": {"revenue": 100}, "ratios": {}}]
        result = calculate_growth_trends(data)
        assert result == {}

    def test_unsorted_input(self):
        data = [
            {"year": 2024, "income_statement": {"revenue": 200, "operating_income": 40, "net_income": 30}, "ratios": {"roe": 0.15}},
            {"year": 2022, "income_statement": {"revenue": 100, "operating_income": 20, "net_income": 15}, "ratios": {"roe": 0.10}},
        ]
        result = calculate_growth_trends(data)
        assert result["revenue"][0]["year"] == 2022  # sorted correctly
        assert result["revenue_cagr"] is not None

    def test_zero_revenue(self):
        data = [
            {"year": 2022, "income_statement": {"revenue": 0}, "ratios": {}},
            {"year": 2023, "income_statement": {"revenue": 100}, "ratios": {}},
        ]
        result = calculate_growth_trends(data)
        assert result["revenue_cagr"] is None  # can't compute CAGR from 0
