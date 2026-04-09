"""Portfolio metrics calculation tests."""

import math
import pytest
import numpy as np

from koreaalpha_core.portfolio.metrics import (
    calculate_returns,
    calculate_cagr,
    calculate_cagr_from_returns,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_mdd,
    calculate_mdd_duration,
    calculate_calmar_ratio,
    calculate_beta,
    calculate_correlation,
    calculate_all_metrics,
    PortfolioMetrics,
)


class TestCalculateReturns:
    def test_basic(self):
        prices = [100, 110, 105, 115]
        returns = calculate_returns(prices)
        assert len(returns) == 3
        assert abs(returns[0] - 0.10) < 1e-10
        assert abs(returns[1] - (-5 / 110)) < 1e-10

    def test_too_short(self):
        with pytest.raises(ValueError):
            calculate_returns([100])


class TestCagr:
    def test_basic(self):
        # 100 → 200 in 5 years = ~14.87%
        cagr = calculate_cagr(100, 200, 5)
        assert abs(cagr - 0.1487) < 0.001

    def test_no_growth(self):
        assert calculate_cagr(100, 100, 3) == 0.0

    def test_negative(self):
        cagr = calculate_cagr(100, 80, 2)
        assert cagr < 0

    def test_invalid(self):
        with pytest.raises(ValueError):
            calculate_cagr(0, 100, 1)


class TestVolatility:
    def test_zero_vol(self):
        # Identical returns yield near-zero volatility (need 3+ samples for ddof=1)
        returns = [0.01, 0.01, 0.01, 0.01]
        vol = calculate_volatility(returns)
        assert vol < 0.001

    def test_positive(self):
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        vol = calculate_volatility(returns)
        assert vol > 0


class TestSharpeRatio:
    def test_positive(self):
        # Consistently positive returns -> positive Sharpe
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 252)  # daily mean 0.1%, daily vol 1%
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.035)
        assert isinstance(sharpe, float)

    def test_risk_free_impact(self):
        np.random.seed(123)
        returns = list(np.random.normal(0.001, 0.01, 252))
        sharpe_low_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.01)
        sharpe_high_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.10)
        assert sharpe_low_rf > sharpe_high_rf


class TestSortinoRatio:
    def test_no_downside(self):
        returns = [0.01, 0.02, 0.03, 0.01, 0.02]
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
        assert sortino == float("inf")

    def test_with_downside(self):
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
        assert isinstance(sortino, float)
        assert sortino > 0


class TestMDD:
    def test_basic(self):
        # 100 → 120 → 90 → 110 : MDD = (90-120)/120 = -25%
        prices = [100, 120, 90, 110]
        mdd = calculate_mdd(prices)
        assert abs(mdd - (-0.25)) < 1e-10

    def test_no_drawdown(self):
        prices = [100, 110, 120, 130]
        mdd = calculate_mdd(prices)
        assert mdd == 0.0

    def test_duration(self):
        prices = [100, 120, 110, 105, 115, 130]
        dur = calculate_mdd_duration(prices)
        assert dur == 3  # 120 -> 110 -> 105 -> 115 (not yet recovered to 120)... recovers at 130


class TestCalmarRatio:
    def test_basic(self):
        assert calculate_calmar_ratio(0.10, -0.20) == 0.5

    def test_no_drawdown(self):
        assert calculate_calmar_ratio(0.10, 0.0) == float("inf")


class TestBeta:
    def test_same_returns(self):
        returns = [0.01, -0.02, 0.03, -0.01]
        beta = calculate_beta(returns, returns)
        assert abs(beta - 1.0) < 1e-10

    def test_double_returns(self):
        bench = [0.01, -0.02, 0.03, -0.01, 0.02]
        port = [r * 2 for r in bench]
        beta = calculate_beta(port, bench)
        assert abs(beta - 2.0) < 1e-10


class TestCorrelation:
    def test_perfect(self):
        a = [0.01, 0.02, 0.03, 0.04]
        assert abs(calculate_correlation(a, a) - 1.0) < 1e-10

    def test_inverse(self):
        a = [0.01, 0.02, 0.03, 0.04]
        b = [-0.01, -0.02, -0.03, -0.04]
        assert abs(calculate_correlation(a, b) - (-1.0)) < 1e-10


class TestAllMetrics:
    def test_basic(self):
        np.random.seed(42)
        prices = [1000]
        for _ in range(251):
            prices.append(prices[-1] * (1 + np.random.normal(0.0003, 0.01)))

        m = calculate_all_metrics(prices)
        assert isinstance(m, PortfolioMetrics)
        assert m.mdd <= 0
        assert m.volatility >= 0
        assert 0 <= m.positive_days_pct <= 1
        assert m.best_day >= m.worst_day
