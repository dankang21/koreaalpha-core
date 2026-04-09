"""Tests for newly added metrics."""

import numpy as np
import pytest
from koreaalpha_core.portfolio.metrics import (
    rolling_sharpe, rolling_volatility, drawdown_series,
    calculate_var, calculate_cvar, calculate_skewness, calculate_kurtosis,
    calculate_alpha, calculate_information_ratio,
    calculate_omega_ratio, calculate_tail_ratio,
    monthly_returns, annual_returns, longest_streak,
    correlation_matrix,
)
from koreaalpha_core.kr_market import (
    calc_dividend_tax, calc_overseas_cgt, calc_after_tax_return,
    DEFAULT_FINANCIAL_INCOME_THRESHOLD,
)


class TestRolling:
    def test_rolling_sharpe(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 120)
        rs = rolling_sharpe(returns, window=60)
        assert len(rs) == 120
        assert np.isnan(rs[0])
        assert not np.isnan(rs[59])

    def test_rolling_volatility(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 60)
        rv = rolling_volatility(returns, window=20)
        assert len(rv) == 60
        assert not np.isnan(rv[19])
        assert rv[19] > 0


class TestDrawdown:
    def test_series(self):
        prices = [100, 120, 90, 110, 130]
        dd = drawdown_series(prices)
        assert len(dd) == 5
        assert dd[0] == 0.0  # starting point
        assert abs(dd[2] - (-0.25)) < 1e-10  # 120→90


class TestVaR:
    def test_var_95(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 252)
        var = calculate_var(returns, 0.95)
        assert var < 0

    def test_cvar(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 252)
        var = calculate_var(returns, 0.95)
        cvar = calculate_cvar(returns, 0.95)
        assert cvar <= var  # CVaR is always <= VaR


class TestDistribution:
    def test_skewness(self):
        returns = [0.01, 0.02, -0.01, 0.03, -0.02, 0.01]
        sk = calculate_skewness(returns)
        assert isinstance(sk, float)

    def test_kurtosis(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.01, 1000)
        k = calculate_kurtosis(returns)
        assert abs(k) < 1  # near-normal distribution, so close to 0


class TestAlpha:
    def test_same_returns(self):
        r = [0.01, -0.02, 0.03, -0.01, 0.02]
        alpha = calculate_alpha(r, r)
        assert abs(alpha) < 0.01

    def test_info_ratio(self):
        r = [0.02, 0.01, 0.03, 0.02, 0.01]
        b = [0.01, 0.01, 0.01, 0.01, 0.01]
        ir = calculate_information_ratio(r, b)
        assert ir > 0


class TestOmega:
    def test_positive(self):
        returns = [0.01, 0.02, -0.005, 0.03, -0.01]
        omega = calculate_omega_ratio(returns)
        assert omega > 1  # more positive returns than negative

    def test_tail_ratio(self):
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 252)
        tr = calculate_tail_ratio(returns)
        assert tr > 0


class TestReturnsMatrix:
    def test_monthly(self):
        prices = list(range(100, 160))
        dates = [f"2025{m:02d}{d:02d}" for m in [1, 2, 3] for d in range(1, 21)]
        result = monthly_returns(prices, dates)
        assert len(result) == 3
        assert result[0]["month"] == 1

    def test_annual(self):
        prices = list(range(100, 160))
        dates = [f"202501{d:02d}" for d in range(1, 31)] + [f"202601{d:02d}" for d in range(1, 31)]
        result = annual_returns(prices, dates)
        assert len(result) == 2


class TestStreak:
    def test_basic(self):
        returns = [0.01, 0.02, 0.03, -0.01, -0.02, 0.01]
        s = longest_streak(returns)
        assert s["win_streak"] == 3
        assert s["loss_streak"] == 2


class TestCorrelationMatrix:
    def test_2x2(self):
        r = correlation_matrix({
            "A": [0.01, 0.02, 0.03],
            "B": [0.01, 0.02, 0.03],
        })
        assert r["labels"] == ["A", "B"]
        assert abs(r["matrix"][0][1] - 1.0) < 1e-4  # perfect correlation


class TestTax:
    def test_dividend_tax(self):
        result = calc_dividend_tax(10_000_000)
        assert result["tax"] == round(10_000_000 * 0.154)
        assert result["is_over_threshold"] is False

    def test_dividend_over_threshold(self):
        result = calc_dividend_tax(25_000_000)
        assert result["is_over_threshold"] is True

    def test_overseas_cgt(self):
        result = calc_overseas_cgt(10_000_000)
        assert result["deduction"] == 2_500_000
        assert result["taxable"] == 7_500_000
        assert result["tax"] == round(7_500_000 * 0.22)

    def test_overseas_cgt_under_deduction(self):
        result = calc_overseas_cgt(2_000_000)
        assert result["tax"] == 0

    def test_after_tax_return(self):
        result = calc_after_tax_return(0.10, 100_000_000, dividend_income=5_000_000)
        assert result["after_tax_return"] < result["gross_return"]
        assert result["total_tax"] > 0
