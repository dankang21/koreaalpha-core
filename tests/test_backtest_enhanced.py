"""Backtest trading day integration + dividend reinvestment tests."""

import numpy as np
from koreaalpha_core.portfolio.backtest import (
    run_backtest, BacktestConfig, _get_rebalance_indices, _period_key,
)


class TestPeriodKey:
    def test_monthly(self):
        assert _period_key("20260401", "monthly") == "202604"
        assert _period_key("20260501", "monthly") == "202605"

    def test_quarterly(self):
        assert _period_key("20260101", "quarterly") == "2026Q1"
        assert _period_key("20260401", "quarterly") == "2026Q2"
        assert _period_key("20261001", "quarterly") == "2026Q4"

    def test_yearly(self):
        assert _period_key("20260401", "yearly") == "2026"


class TestRebalanceIndices:
    def test_none_period(self):
        indices = _get_rebalance_indices(100, "none")
        assert indices == []

    def test_monthly_with_dates(self):
        # Jan-Mar trading days (20 days each)
        dates = [f"202601{d:02d}" for d in range(2, 22)] + \
                [f"202602{d:02d}" for d in range(2, 22)] + \
                [f"202603{d:02d}" for d in range(2, 22)]
        indices = _get_rebalance_indices(60, "monthly", dates, use_kr_trading_days=True)
        assert 0 in indices  # first day
        assert len(indices) == 3  # 3 months

    def test_simple_fallback(self):
        indices = _get_rebalance_indices(100, "monthly", dates=None, use_kr_trading_days=False)
        assert 0 in indices
        assert 21 in indices


class TestDividendReinvest:
    def test_dividend_increases_value(self):
        np.random.seed(42)
        n = 248
        prices = {"A": [100.0]}
        for _ in range(n - 1):
            prices["A"].append(prices["A"][-1] * (1 + np.random.normal(0.0003, 0.01)))

        # Without dividends
        config_no_div = BacktestConfig(
            initial_capital=10_000_000,
            rebalance_period="none",
            dividend_reinvest=False,
        )
        result_no_div = run_backtest({"A": prices["A"]}, {"A": 1.0}, config_no_div)

        # Dividend reinvestment (3% annual)
        config_div = BacktestConfig(
            initial_capital=10_000_000,
            rebalance_period="none",
            dividend_reinvest=True,
            dividend_yields={"A": 0.03},
        )
        result_div = run_backtest({"A": prices["A"]}, {"A": 1.0}, config_div)

        # Dividend reinvestment should yield higher value
        assert result_div.portfolio_values[-1] > result_no_div.portfolio_values[-1]

    def test_no_dividend_same_result(self):
        prices = {"A": [100, 101, 102, 103, 104, 105]}
        config = BacktestConfig(
            initial_capital=1_000_000,
            rebalance_period="none",
            dividend_reinvest=False,
        )
        r1 = run_backtest({"A": prices["A"]}, {"A": 1.0}, config)

        config2 = BacktestConfig(
            initial_capital=1_000_000,
            rebalance_period="none",
            dividend_reinvest=True,
            dividend_yields={},  # empty yields
        )
        r2 = run_backtest({"A": prices["A"]}, {"A": 1.0}, config2)

        assert abs(r1.portfolio_values[-1] - r2.portfolio_values[-1]) < 1


class TestKrTradingDayRebalance:
    def test_skips_holiday(self):
        # 2026-01-01 is a holiday, 2026-01-02 is Friday (trading day)
        dates = ["20251231", "20260101", "20260102", "20260105"]
        prices = {"A": [100, 100, 101, 102]}

        config = BacktestConfig(
            initial_capital=1_000_000,
            rebalance_period="monthly",
            use_kr_trading_days=True,
        )
        result = run_backtest({"A": prices["A"]}, {"A": 1.0}, config, dates=dates)

        # Verify rebalancing only occurs on trading days
        for event in result.rebalance_events:
            assert event.date != "20260101"  # no rebalancing on holidays
