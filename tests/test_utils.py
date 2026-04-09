"""Formatters / validators tests."""

from koreaalpha_core.utils.formatters import format_percentage, format_currency, format_number
from koreaalpha_core.utils.validators import (
    validate_kr_stock_code, validate_us_ticker, validate_portfolio_weights, validate_date_yyyymmdd,
)


class TestFormatters:
    def test_percentage(self):
        assert format_percentage(0.1234) == "12.34%"
        assert format_percentage(0.1, 1) == "10.0%"

    def test_currency_krw(self):
        assert "원" in format_currency(1_234_567)
        assert "억" in format_currency(1_5000_0000)
        assert "조" in format_currency(1_5000_0000_0000)

    def test_currency_usd(self):
        assert "$" in format_currency(1234.56, "USD")

    def test_number(self):
        assert format_number(1.2345) == "1.23"
        assert format_number(1000, 0) == "1,000"


class TestValidators:
    def test_kr_stock_code(self):
        assert validate_kr_stock_code("005930") is True
        assert validate_kr_stock_code("12345") is False
        assert validate_kr_stock_code("AAPL") is False

    def test_us_ticker(self):
        assert validate_us_ticker("AAPL") is True
        assert validate_us_ticker("A") is True
        assert validate_us_ticker("123456") is False
        assert validate_us_ticker("toolong") is False

    def test_portfolio_weights(self):
        assert validate_portfolio_weights([0.5, 0.3, 0.2]) is True
        assert validate_portfolio_weights([0.5, 0.5, 0.5]) is False
        assert validate_portfolio_weights([1.0]) is True

    def test_date_yyyymmdd(self):
        assert validate_date_yyyymmdd("20260401") is True
        assert validate_date_yyyymmdd("2026-04-01") is False
        assert validate_date_yyyymmdd("202604") is False
