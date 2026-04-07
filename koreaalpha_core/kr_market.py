"""Korean market-specific module.

Korean trading days, transaction costs, and tax calculations.
Holiday calculations are delegated to the korean-holidays package.
Tax rates are provided as defaults but can be overridden via parameters.
"""

from datetime import date, timedelta

# Delegate holiday/trading day calculations to the korean-holidays package
from korean_holidays import (
    is_holiday as _is_holiday,
    is_trading_day as is_kr_trading_day,
    get_trading_days,
    count_trading_days,
)

# Default tax rates (as of 2026; override via parameters if changed)
DEFAULT_SECURITIES_TAX_RATE = 0.0018  # Securities transaction tax 0.18% on sell
DEFAULT_BROKERAGE_FEE_RATE = 0.00015  # Brokerage fee (one-way)
DEFAULT_DIVIDEND_TAX_RATE = 0.154  # Dividend income tax (withholding 15.4%)
DEFAULT_OVERSEAS_CGT_RATE = 0.22   # Overseas stock capital gains tax 22%
DEFAULT_OVERSEAS_CGT_DEDUCTION = 2_500_000  # Overseas stock CGT deduction 2.5M KRW
DEFAULT_FINANCIAL_INCOME_THRESHOLD = 20_000_000  # Comprehensive financial income tax threshold 20M KRW


def calc_transaction_cost(
    amount: float,
    is_sell: bool = False,
    brokerage_rate: float = DEFAULT_BROKERAGE_FEE_RATE,
    securities_tax_rate: float = DEFAULT_SECURITIES_TAX_RATE,
) -> float:
    """Calculate transaction cost.

    Args:
        amount: Transaction amount
        is_sell: Whether it is a sell order (securities transaction tax added on sell)
        brokerage_rate: Brokerage fee rate
        securities_tax_rate: Securities transaction tax rate

    Returns:
        Total transaction cost
    """
    cost = abs(amount) * brokerage_rate
    if is_sell:
        cost += abs(amount) * securities_tax_rate
    return cost


def calc_round_trip_cost(
    amount: float,
    brokerage_rate: float = DEFAULT_BROKERAGE_FEE_RATE,
    securities_tax_rate: float = DEFAULT_SECURITIES_TAX_RATE,
) -> float:
    """Calculate round-trip transaction cost (buy + sell)."""
    buy_cost = calc_transaction_cost(amount, is_sell=False, brokerage_rate=brokerage_rate)
    sell_cost = calc_transaction_cost(amount, is_sell=True, brokerage_rate=brokerage_rate, securities_tax_rate=securities_tax_rate)
    return buy_cost + sell_cost


def calc_dividend_tax(
    dividend: float,
    tax_rate: float = DEFAULT_DIVIDEND_TAX_RATE,
    threshold: float = DEFAULT_FINANCIAL_INCOME_THRESHOLD,
) -> dict:
    """Calculate dividend income tax.

    Args:
        dividend: Dividend amount
        tax_rate: Withholding tax rate (default 15.4%)
        threshold: Comprehensive financial income tax threshold (default 20M KRW)

    Returns:
        {"gross": dividend, "tax": tax, "net": after-tax, "is_over_threshold": subject to comprehensive taxation}
    """
    tax = dividend * tax_rate
    return {
        "gross": round(dividend),
        "tax": round(tax),
        "net": round(dividend - tax),
        "is_over_threshold": dividend > threshold,
    }


def calc_overseas_cgt(
    gain: float,
    tax_rate: float = DEFAULT_OVERSEAS_CGT_RATE,
    deduction: float = DEFAULT_OVERSEAS_CGT_DEDUCTION,
) -> dict:
    """Calculate overseas stock capital gains tax (annual basis).

    Args:
        gain: Annual capital gains
        tax_rate: Capital gains tax rate (default 22%)
        deduction: Basic deduction (default 2.5M KRW)

    Returns:
        {"gain": gains, "deduction": deduction, "taxable": taxable amount, "tax": tax, "net": after-tax}
    """
    if gain <= 0:
        return {"gain": round(gain), "deduction": 0, "taxable": 0, "tax": 0, "net": round(gain)}
    taxable = max(gain - deduction, 0)
    tax = taxable * tax_rate
    return {
        "gain": round(gain),
        "deduction": deduction,
        "taxable": round(taxable),
        "tax": round(tax),
        "net": round(gain - tax),
    }


def calc_after_tax_return(
    gross_return: float,
    investment: float,
    dividend_income: float = 0,
    overseas_gain: float = 0,
    dividend_tax_rate: float = DEFAULT_DIVIDEND_TAX_RATE,
    overseas_cgt_rate: float = DEFAULT_OVERSEAS_CGT_RATE,
    overseas_cgt_deduction: float = DEFAULT_OVERSEAS_CGT_DEDUCTION,
) -> dict:
    """Calculate after-tax real return.

    Args:
        gross_return: Pre-tax total return (e.g., 0.12 = 12%)
        investment: Investment principal
        dividend_income: Annual dividend income
        overseas_gain: Overseas stock capital gains
        dividend_tax_rate: Dividend income tax rate
        overseas_cgt_rate: Overseas stock capital gains tax rate
        overseas_cgt_deduction: Overseas stock CGT deduction

    Returns:
        {"gross_return": pre-tax, "after_tax_return": after-tax, "total_tax": total tax}
    """
    gross_profit = investment * gross_return

    div_tax = dividend_income * dividend_tax_rate if dividend_income > 0 else 0

    overseas_tax = 0
    if overseas_gain > 0:
        taxable = max(overseas_gain - overseas_cgt_deduction, 0)
        overseas_tax = taxable * overseas_cgt_rate

    total_tax = div_tax + overseas_tax
    net_profit = gross_profit - total_tax
    after_tax_return = net_profit / investment if investment > 0 else 0

    return {
        "gross_return": round(gross_return, 6),
        "after_tax_return": round(after_tax_return, 6),
        "total_tax": round(total_tax),
    }
