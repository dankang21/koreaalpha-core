"""Backtest engine module.

Simulates portfolio performance based on historical data.
Incorporates rebalancing, transaction costs, and slippage.
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from .metrics import PortfolioMetrics, calculate_all_metrics


@dataclass
class BacktestConfig:
    """Backtest configuration."""

    initial_capital: float = 10_000_000  # Initial capital (10 million KRW)
    rebalance_period: str = "monthly"  # monthly, quarterly, yearly, none
    transaction_cost_pct: float = 0.0015  # One-way transaction cost 0.15% (brokerage fee + tax approx.)
    slippage_pct: float = 0.001  # Slippage 0.1%
    start_date: str | None = None
    end_date: str | None = None
    use_kr_trading_days: bool = True  # Rebalancing based on Korean trading days
    dividend_reinvest: bool = False  # Whether to reinvest dividends
    dividend_yields: dict[str, float] | None = None  # {asset_name: annual_dividend_yield}


@dataclass
class RebalanceEvent:
    """Rebalancing event record."""

    date: str
    day_index: int
    trades: dict[str, float]  # asset_name -> trade amount (positive=buy, negative=sell)
    cost: float  # Transaction cost


@dataclass
class BacktestResult:
    """Backtest result."""

    config: BacktestConfig
    metrics: PortfolioMetrics
    portfolio_values: list[float]  # Daily portfolio values
    dates: list[str]  # Date list
    rebalance_events: list[RebalanceEvent]
    total_transaction_cost: float
    allocations: dict[str, float]  # Target asset allocation


def _get_rebalance_indices(
    num_days: int,
    period: str,
    dates: list[str] | None = None,
    use_kr_trading_days: bool = False,
) -> list[int]:
    """Return rebalancing point indices.

    If dates are in YYYYMMDD format and use_kr_trading_days=True,
    rebalancing occurs on the first trading day of each month/quarter/year
    based on Korean trading days.
    """
    if period == "none":
        return []

    # Date-based rebalancing (Korean trading day integration)
    if dates and len(dates) == num_days and use_kr_trading_days:
        from ..kr_market import is_kr_trading_day
        from datetime import date as dt_date

        indices = [0]  # Always rebalance on the first day
        prev_key = _period_key(dates[0], period)

        for i in range(1, num_days):
            cur_key = _period_key(dates[i], period)
            if cur_key != prev_key:
                # New period starts -> check if it's a trading day
                try:
                    d = dt_date(int(dates[i][:4]), int(dates[i][4:6]), int(dates[i][6:8]))
                    if is_kr_trading_day(d):
                        indices.append(i)
                        prev_key = cur_key
                except (ValueError, IndexError):
                    indices.append(i)
                    prev_key = cur_key
        return indices

    # Simple interval-based (legacy method)
    if period == "monthly":
        interval = 21
    elif period == "quarterly":
        interval = 63
    elif period == "yearly":
        interval = 248  # Based on Korean trading days
    else:
        raise ValueError(f"Unsupported rebalancing period: {period}")

    return list(range(0, num_days, interval))


def _period_key(date_str: str, period: str) -> str:
    """Extract a period key from a date string."""
    if period == "monthly":
        return date_str[:6]  # YYYYMM
    elif period == "quarterly":
        month = int(date_str[4:6])
        quarter = (month - 1) // 3 + 1
        return f"{date_str[:4]}Q{quarter}"
    elif period == "yearly":
        return date_str[:4]  # YYYY
    return date_str


def run_backtest(
    asset_prices: dict[str, list[float]],
    allocations: dict[str, float],
    config: BacktestConfig | None = None,
    dates: list[str] | None = None,
) -> BacktestResult:
    """Run a portfolio backtest.

    Args:
        asset_prices: {asset_name: daily_closing_prices} -- all assets must have the same length
        allocations: {asset_name: target_weight} -- must sum to 1.0
        config: Backtest configuration
        dates: Date list (optional, for display purposes)

    Returns:
        BacktestResult
    """
    if config is None:
        config = BacktestConfig()

    # Validation
    asset_names = list(asset_prices.keys())
    if not asset_names:
        raise ValueError("At least 1 asset is required.")

    num_days = len(asset_prices[asset_names[0]])
    for name, prices in asset_prices.items():
        if len(prices) != num_days:
            raise ValueError(
                f"Asset '{name}' data length ({len(prices)}) differs "
                f"from other assets ({num_days})."
            )

    alloc_sum = sum(allocations.values())
    if abs(alloc_sum - 1.0) > 0.01:
        raise ValueError(f"Asset allocation weights do not sum to 1.0: {alloc_sum:.4f}")

    # Prices -> numpy
    price_arrays = {
        name: np.asarray(prices, dtype=np.float64)
        for name, prices in asset_prices.items()
    }

    # Daily returns
    return_arrays = {}
    for name, parr in price_arrays.items():
        rets = np.zeros(num_days)
        rets[1:] = np.diff(parr) / parr[:-1]
        return_arrays[name] = rets

    # Rebalancing points (trading day integration)
    rebal_indices = _get_rebalance_indices(
        num_days, config.rebalance_period, dates, config.use_kr_trading_days,
    )

    # Dividend reinvestment setup
    daily_div_yields = {}
    if config.dividend_reinvest and config.dividend_yields:
        from .metrics import TRADING_DAYS
        for name, annual_yield in config.dividend_yields.items():
            daily_div_yields[name] = annual_yield / TRADING_DAYS

    # Simulation
    capital = config.initial_capital
    portfolio_values = np.zeros(num_days)
    holdings = {}  # Holdings per asset
    rebalance_events = []
    total_cost = 0.0

    for day in range(num_days):
        if day == 0 or day in rebal_indices:
            # On rebalancing days, apply daily return first (except day 0)
            if day > 0:
                for name in asset_names:
                    if name in holdings:
                        holdings[name] *= 1 + return_arrays[name][day]
                capital = sum(holdings.values())

            # Rebalance: redistribute to target weights
            trades = {}
            cost = 0.0

            for name in asset_names:
                target = capital * allocations.get(name, 0.0)
                current = holdings.get(name, 0.0)
                diff = target - current

                if abs(diff) > 1:  # Only trade if difference > 1 KRW
                    trade_cost = abs(diff) * (
                        config.transaction_cost_pct + config.slippage_pct
                    )
                    cost += trade_cost
                    trades[name] = diff

                holdings[name] = target

            capital -= cost
            total_cost += cost

            # Readjust holdings after cost deduction
            if capital > 0 and cost > 0:
                ratio = capital / (capital + cost)
                for name in holdings:
                    holdings[name] *= ratio

            if trades:
                rebalance_events.append(
                    RebalanceEvent(
                        date=dates[day] if dates else f"day_{day}",
                        day_index=day,
                        trades=trades,
                        cost=round(cost, 2),
                    )
                )
        else:
            # Apply daily returns
            for name in asset_names:
                if name in holdings:
                    holdings[name] *= 1 + return_arrays[name][day]

            # Dividend reinvestment (reinvest daily dividends into the same asset)
            if daily_div_yields:
                for name in asset_names:
                    if name in holdings and name in daily_div_yields:
                        div = holdings[name] * daily_div_yields[name]
                        holdings[name] += div

        capital = sum(holdings.values())
        portfolio_values[day] = capital

    # Calculate metrics
    metrics = calculate_all_metrics(portfolio_values.tolist())

    return BacktestResult(
        config=config,
        metrics=metrics,
        portfolio_values=portfolio_values.tolist(),
        dates=dates or [f"day_{i}" for i in range(num_days)],
        rebalance_events=rebalance_events,
        total_transaction_cost=round(total_cost, 2),
        allocations=allocations,
    )


def generate_backtest_report(result: BacktestResult) -> dict:
    """Convert backtest results into a report format.

    Returns:
        Report dictionary
    """
    m = result.metrics
    return {
        "summary": {
            "initial_capital": result.config.initial_capital,
            "final_value": round(result.portfolio_values[-1], 0),
            "total_return_pct": round(m.total_return * 100, 2),
            "cagr_pct": round(m.cagr * 100, 2),
            "volatility_pct": round(m.volatility * 100, 2),
            "sharpe_ratio": m.sharpe_ratio,
            "sortino_ratio": m.sortino_ratio,
            "mdd_pct": round(m.mdd * 100, 2),
            "mdd_duration_days": m.mdd_duration_days,
            "calmar_ratio": m.calmar_ratio,
        },
        "allocations": result.allocations,
        "rebalance_count": len(result.rebalance_events),
        "total_transaction_cost": result.total_transaction_cost,
        "transaction_cost_pct": round(
            result.total_transaction_cost / result.config.initial_capital * 100,
            4,
        ),
        "period": {
            "start": result.dates[0] if result.dates else None,
            "end": result.dates[-1] if result.dates else None,
            "trading_days": len(result.portfolio_values),
        },
    }
