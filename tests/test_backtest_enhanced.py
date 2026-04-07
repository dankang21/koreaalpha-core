"""백테스트 거래일 연동 + 배당 재투자 테스트."""

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
        # 1월~3월 거래일 (각 20일)
        dates = [f"202601{d:02d}" for d in range(2, 22)] + \
                [f"202602{d:02d}" for d in range(2, 22)] + \
                [f"202603{d:02d}" for d in range(2, 22)]
        indices = _get_rebalance_indices(60, "monthly", dates, use_kr_trading_days=True)
        assert 0 in indices  # 첫 날
        assert len(indices) == 3  # 3개월

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

        # 배당 없이
        config_no_div = BacktestConfig(
            initial_capital=10_000_000,
            rebalance_period="none",
            dividend_reinvest=False,
        )
        result_no_div = run_backtest({"A": prices["A"]}, {"A": 1.0}, config_no_div)

        # 배당 재투자 (연 3%)
        config_div = BacktestConfig(
            initial_capital=10_000_000,
            rebalance_period="none",
            dividend_reinvest=True,
            dividend_yields={"A": 0.03},
        )
        result_div = run_backtest({"A": prices["A"]}, {"A": 1.0}, config_div)

        # 배당 재투자가 더 높아야 함
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
            dividend_yields={},  # 빈 yields
        )
        r2 = run_backtest({"A": prices["A"]}, {"A": 1.0}, config2)

        assert abs(r1.portfolio_values[-1] - r2.portfolio_values[-1]) < 1


class TestKrTradingDayRebalance:
    def test_skips_holiday(self):
        # 2026-01-01은 공휴일, 2026-01-02는 금요일(거래일)
        dates = ["20251231", "20260101", "20260102", "20260105"]
        prices = {"A": [100, 100, 101, 102]}

        config = BacktestConfig(
            initial_capital=1_000_000,
            rebalance_period="monthly",
            use_kr_trading_days=True,
        )
        result = run_backtest({"A": prices["A"]}, {"A": 1.0}, config, dates=dates)

        # 리밸런싱이 거래일에만 발생하는지 확인
        for event in result.rebalance_events:
            assert event.date != "20260101"  # 공휴일에 리밸런싱 안 함
