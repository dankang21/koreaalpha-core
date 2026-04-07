"""Input validation utilities."""

import re


def validate_kr_stock_code(code: str) -> bool:
    """Validate a Korean stock code (6-digit number)."""
    return bool(re.match(r"^\d{6}$", code))


def validate_us_ticker(ticker: str) -> bool:
    """Validate a US ticker symbol (1-5 uppercase letters)."""
    return bool(re.match(r"^[A-Z]{1,5}$", ticker))


def validate_portfolio_weights(weights: list[float], tolerance: float = 0.01) -> bool:
    """Validate that portfolio weights sum to 1.0 (100%)."""
    total = sum(weights)
    return abs(total - 1.0) <= tolerance


def validate_date_yyyymmdd(date: str) -> bool:
    """Validate a date in YYYYMMDD format."""
    return bool(re.match(r"^\d{8}$", date))
