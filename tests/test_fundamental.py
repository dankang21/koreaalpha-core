"""펀더멘탈 분석 테스트."""

import pytest

from koreaalpha_core.stock.fundamental import (
    calculate_per,
    calculate_pbr,
    calculate_roe,
    calculate_debt_ratio,
    calculate_all_fundamentals,
)


class TestPER:
    def test_basic(self):
        assert calculate_per(50000, 5000) == 10.0

    def test_negative_eps(self):
        # 한국 관행: 적자 기업도 음수 PER 반환 (기본값)
        assert calculate_per(50000, -1000) == -50.0
        # allow_negative=False이면 None
        assert calculate_per(50000, -1000, allow_negative=False) is None

    def test_zero_eps(self):
        assert calculate_per(50000, 0) is None


class TestPBR:
    def test_basic(self):
        assert calculate_pbr(50000, 25000) == 2.0

    def test_negative_bps(self):
        assert calculate_pbr(50000, -100) is None


class TestROE:
    def test_basic(self):
        roe = calculate_roe(1_000_000, 10_000_000)
        assert roe == 0.1  # 10%

    def test_negative_equity(self):
        assert calculate_roe(1_000_000, -500_000) is None


class TestDebtRatio:
    def test_basic(self):
        ratio = calculate_debt_ratio(15_000_000, 10_000_000)
        assert ratio == 1.5  # 150%


class TestAllFundamentals:
    def test_samsung_like(self):
        """삼성전자 유사 데이터로 테스트."""
        m = calculate_all_fundamentals(
            price=70000,
            eps=5000,
            bps=50000,
            net_income=30_000_000_000_000,
            equity=300_000_000_000_000,
            total_assets=450_000_000_000_000,
            total_liabilities=150_000_000_000_000,
            revenue=250_000_000_000_000,
            operating_income=40_000_000_000_000,
            fcf=20_000_000_000_000,
            market_cap=400_000_000_000_000,
            dividend_per_share=1444,
        )
        assert m.per == 14.0
        assert m.pbr == 1.4
        assert m.roe == 0.1
        assert m.debt_ratio == 0.5
        assert m.operating_margin == 0.16
        assert m.dividend_yield > 0
