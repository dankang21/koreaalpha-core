"""한국 시장 특화 모듈 테스트."""

from datetime import date
from koreaalpha_core.kr_market import (
    is_kr_trading_day, count_trading_days, get_trading_days,
    calc_transaction_cost, calc_round_trip_cost,
    DEFAULT_SECURITIES_TAX_RATE,
)


class TestTradingDay:
    def test_weekday(self):
        assert is_kr_trading_day(date(2026, 4, 6))  # 월요일

    def test_weekend(self):
        assert not is_kr_trading_day(date(2026, 4, 4))  # 토요일
        assert not is_kr_trading_day(date(2026, 4, 5))  # 일요일

    def test_holiday(self):
        assert not is_kr_trading_day(date(2026, 1, 1))  # 신정
        assert not is_kr_trading_day(date(2026, 3, 1))  # 삼일절
        assert not is_kr_trading_day(date(2026, 12, 25))  # 크리스마스


class TestCountTradingDays:
    def test_one_week(self):
        days = count_trading_days(date(2026, 4, 6), date(2026, 4, 10))
        assert days == 5  # 월~금

    def test_includes_weekend(self):
        days = count_trading_days(date(2026, 4, 4), date(2026, 4, 10))
        assert days == 5  # 토일 제외

    def test_get_list(self):
        days = get_trading_days(date(2026, 4, 6), date(2026, 4, 10))
        assert len(days) == 5
        assert all(d.weekday() < 5 for d in days)


class TestTransactionCost:
    def test_buy(self):
        cost = calc_transaction_cost(10_000_000, is_sell=False)
        assert cost == 10_000_000 * 0.00015  # 수수료만

    def test_sell_includes_tax(self):
        cost = calc_transaction_cost(10_000_000, is_sell=True)
        expected = 10_000_000 * (0.00015 + DEFAULT_SECURITIES_TAX_RATE)
        assert abs(cost - expected) < 1

    def test_round_trip(self):
        rt = calc_round_trip_cost(10_000_000)
        buy = calc_transaction_cost(10_000_000, is_sell=False)
        sell = calc_transaction_cost(10_000_000, is_sell=True)
        assert abs(rt - (buy + sell)) < 1
