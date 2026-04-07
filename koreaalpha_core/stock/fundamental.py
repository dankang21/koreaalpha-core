"""Fundamental analysis module.

Calculates fundamental metrics including PER, PBR, ROE, Debt Ratio, FCF, etc.
"""

from dataclasses import dataclass


@dataclass
class FundamentalMetrics:
    """Comprehensive fundamental metrics."""

    per: float | None  # Price-to-Earnings Ratio
    pbr: float | None  # Price-to-Book Ratio
    roe: float | None  # Return on Equity
    roa: float | None  # Return on Assets
    debt_ratio: float | None  # Debt Ratio
    operating_margin: float | None  # Operating Margin
    net_margin: float | None  # Net Profit Margin
    fcf_yield: float | None  # Free Cash Flow Yield
    dividend_yield: float | None  # Dividend Yield


def calculate_per(price: float, eps: float, allow_negative: bool = True) -> float | None:
    """Calculate Price-to-Earnings Ratio (PER).

    Args:
        price: Current stock price
        eps: Earnings Per Share (EPS)
        allow_negative: If True, return negative PER for loss-making companies (Korean convention)

    Returns:
        PER (None if eps is 0)
    """
    if eps == 0:
        return None
    if eps < 0 and not allow_negative:
        return None
    return round(price / eps, 2)


def calculate_pbr(price: float, bps: float) -> float | None:
    """Calculate Price-to-Book Ratio (PBR).

    Args:
        price: Current stock price
        bps: Book value Per Share (BPS)

    Returns:
        PBR
    """
    if bps <= 0:
        return None
    return round(price / bps, 2)


def calculate_roe(net_income: float, equity: float) -> float | None:
    """Calculate Return on Equity (ROE).

    Args:
        net_income: Net income
        equity: Shareholders' equity

    Returns:
        ROE (e.g., 0.15 = 15%)
    """
    if equity <= 0:
        return None
    return round(net_income / equity, 4)


def calculate_roa(net_income: float, total_assets: float) -> float | None:
    """Calculate Return on Assets (ROA)."""
    if total_assets <= 0:
        return None
    return round(net_income / total_assets, 4)


def calculate_debt_ratio(
    total_liabilities: float, equity: float
) -> float | None:
    """Calculate Debt Ratio.

    Args:
        total_liabilities: Total liabilities
        equity: Shareholders' equity

    Returns:
        Debt Ratio (e.g., 1.5 = 150%)
    """
    if equity <= 0:
        return None
    return round(total_liabilities / equity, 4)


def calculate_operating_margin(
    operating_income: float, revenue: float
) -> float | None:
    """Calculate Operating Margin."""
    if revenue <= 0:
        return None
    return round(operating_income / revenue, 4)


def calculate_net_margin(net_income: float, revenue: float) -> float | None:
    """Calculate Net Profit Margin."""
    if revenue <= 0:
        return None
    return round(net_income / revenue, 4)


def calculate_fcf_yield(
    fcf: float, market_cap: float
) -> float | None:
    """Calculate Free Cash Flow Yield.

    Args:
        fcf: Free Cash Flow
        market_cap: Market capitalization

    Returns:
        FCF Yield
    """
    if market_cap <= 0:
        return None
    return round(fcf / market_cap, 4)


def calculate_all_fundamentals(
    price: float,
    eps: float,
    bps: float,
    net_income: float,
    equity: float,
    total_assets: float,
    total_liabilities: float,
    revenue: float,
    operating_income: float,
    fcf: float,
    market_cap: float,
    dividend_per_share: float = 0.0,
) -> FundamentalMetrics:
    """Calculate all fundamental metrics at once.

    Args:
        price: Current stock price
        eps: Earnings Per Share
        bps: Book value Per Share
        net_income: Net income
        equity: Shareholders' equity
        total_assets: Total assets
        total_liabilities: Total liabilities
        revenue: Revenue
        operating_income: Operating income
        fcf: Free Cash Flow
        market_cap: Market capitalization
        dividend_per_share: Dividend per share

    Returns:
        FundamentalMetrics
    """
    div_yield = round(dividend_per_share / price, 4) if price > 0 else None

    return FundamentalMetrics(
        per=calculate_per(price, eps),
        pbr=calculate_pbr(price, bps),
        roe=calculate_roe(net_income, equity),
        roa=calculate_roa(net_income, total_assets),
        debt_ratio=calculate_debt_ratio(total_liabilities, equity),
        operating_margin=calculate_operating_margin(operating_income, revenue),
        net_margin=calculate_net_margin(net_income, revenue),
        fcf_yield=calculate_fcf_yield(fcf, market_cap),
        dividend_yield=div_yield,
    )
