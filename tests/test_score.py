"""Tests for portfolio scoring."""

from koreaalpha_core.portfolio.score import calculate_portfolio_score


class TestPortfolioScore:
    def test_perfect_metrics(self):
        metrics = {
            "sharpe_ratio": 2.0,
            "mdd": -0.03,
            "cagr": 0.20,
            "sortino_ratio": 3.0,
            "calmar_ratio": 1.5,
        }
        score, grade = calculate_portfolio_score(metrics)
        assert score >= 90
        assert grade == "A+"

    def test_poor_metrics(self):
        metrics = {
            "sharpe_ratio": -0.5,
            "mdd": -0.50,
            "cagr": -0.10,
            "sortino_ratio": 0.1,
            "calmar_ratio": 0.05,
        }
        score, grade = calculate_portfolio_score(metrics)
        assert score < 40
        assert grade == "F"

    def test_moderate_metrics(self):
        metrics = {
            "sharpe_ratio": 0.8,
            "mdd": -0.12,
            "cagr": 0.08,
            "sortino_ratio": 1.2,
            "calmar_ratio": 0.6,
        }
        score, grade = calculate_portfolio_score(metrics)
        assert 50 <= score <= 80
        assert grade in ("B+", "B", "C")

    def test_with_benchmark(self):
        metrics = {"sharpe_ratio": 1.0, "mdd": -0.10, "cagr": 0.10, "sortino_ratio": 1.5, "calmar_ratio": 0.8}
        comps = [{"sharpe_diff": 0.5, "cagr_diff": 0.03}]
        score_with, _ = calculate_portfolio_score(metrics, comps)

        comps_bad = [{"sharpe_diff": -0.3, "cagr_diff": -0.05}]
        score_without, _ = calculate_portfolio_score(metrics, comps_bad)

        assert score_with > score_without

    def test_none_values(self):
        metrics = {"sharpe_ratio": None, "mdd": None, "cagr": None, "sortino_ratio": None, "calmar_ratio": None}
        score, grade = calculate_portfolio_score(metrics)
        assert 0 <= score <= 100
        assert grade in ("A+", "A", "B+", "B", "C", "D", "F")

    def test_deterministic(self):
        metrics = {"sharpe_ratio": 1.0, "mdd": -0.15, "cagr": 0.08, "sortino_ratio": 1.0, "calmar_ratio": 0.5}
        s1, g1 = calculate_portfolio_score(metrics)
        s2, g2 = calculate_portfolio_score(metrics)
        assert s1 == s2
        assert g1 == g2
