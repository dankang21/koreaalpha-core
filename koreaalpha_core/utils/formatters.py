"""Output formatting utilities."""


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a number as a percentage string. 0.1234 -> '12.34%'"""
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, currency: str = "KRW") -> str:
    """Format an amount in currency format. 1234567 -> '1,234,567원'"""
    if currency == "KRW":
        if abs(value) >= 1_0000_0000_0000:
            return f"{value / 1_0000_0000_0000:.1f}조원"
        if abs(value) >= 1_0000_0000:
            return f"{value / 1_0000_0000:.0f}억원"
        return f"{value:,.0f}원"
    elif currency == "USD":
        return f"${value:,.2f}"
    return f"{value:,.2f} {currency}"


def format_number(value: float, decimals: int = 2) -> str:
    """Format a number. 1.2345 -> '1.23'"""
    return f"{value:,.{decimals}f}"
