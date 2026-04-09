"""Efficient frontier calculation via Monte Carlo simulation.

Pure computation — no API calls, no I/O, no data fetching.
Accepts pre-computed return data (numpy arrays).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import TRADING_DAYS_KR


DEFAULT_RISK_FREE_RATE = 0.035


@dataclass
class FrontierPortfolio:
    """A single portfolio on the efficient frontier."""
    expected_return: float
    volatility: float
    sharpe: float
    weights: np.ndarray


@dataclass
class FrontierResult:
    """Efficient frontier calculation result."""
    frontier_points: list[FrontierPortfolio]
    current: FrontierPortfolio
    min_variance: FrontierPortfolio
    max_sharpe: FrontierPortfolio
    individual_assets: list[dict]
    n_simulations: int
    risk_free_rate: float


def portfolio_stats(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    trading_days: int = TRADING_DAYS_KR,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> tuple[float, float, float]:
    """Calculate portfolio expected return, volatility, and Sharpe ratio.

    Args:
        weights: Asset weight array (sums to 1.0).
        mean_returns: Daily mean return per asset.
        cov_matrix: Daily return covariance matrix.
        trading_days: Annualization factor.
        risk_free_rate: Annual risk-free rate.

    Returns:
        (annualized_return, annualized_volatility, sharpe_ratio)
    """
    port_return = float(np.dot(weights, mean_returns) * trading_days)
    port_vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix * trading_days, weights))))
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0
    return port_return, port_vol, sharpe


def calculate_efficient_frontier(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    current_weights: np.ndarray,
    n_points: int = 50,
    n_simulations: int | None = None,
    trading_days: int = TRADING_DAYS_KR,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    seed: int = 42,
) -> FrontierResult:
    """Calculate efficient frontier via random portfolio simulation.

    Args:
        mean_returns: Daily mean return per asset (length N).
        cov_matrix: N×N daily return covariance matrix.
        current_weights: Current portfolio weights (length N).
        n_points: Number of frontier points to extract.
        n_simulations: Number of random portfolios. Defaults to max(n_points*200, 10000).
        trading_days: Annualization factor.
        risk_free_rate: Annual risk-free rate.
        seed: Random seed for reproducibility.

    Returns:
        FrontierResult with frontier points, current/min-var/max-sharpe portfolios.
    """
    n_assets = len(mean_returns)
    if n_simulations is None:
        n_simulations = max(n_points * 200, 10000)

    np.random.seed(seed)

    # Current portfolio
    cur_ret, cur_vol, cur_sharpe = portfolio_stats(
        current_weights, mean_returns, cov_matrix, trading_days, risk_free_rate,
    )
    current = FrontierPortfolio(cur_ret, cur_vol, cur_sharpe, current_weights)

    # Random simulations
    all_returns = np.zeros(n_simulations)
    all_vols = np.zeros(n_simulations)
    all_sharpes = np.zeros(n_simulations)
    all_weights = np.zeros((n_simulations, n_assets))

    for i in range(n_simulations):
        w = np.random.random(n_assets)
        w = w / w.sum()
        all_weights[i] = w
        all_returns[i], all_vols[i], all_sharpes[i] = portfolio_stats(
            w, mean_returns, cov_matrix, trading_days, risk_free_rate,
        )

    # Min variance
    min_var_idx = int(np.argmin(all_vols))
    min_variance = FrontierPortfolio(
        all_returns[min_var_idx], all_vols[min_var_idx],
        all_sharpes[min_var_idx], all_weights[min_var_idx],
    )

    # Max Sharpe
    max_sharpe_idx = int(np.argmax(all_sharpes))
    max_sharpe = FrontierPortfolio(
        all_returns[max_sharpe_idx], all_vols[max_sharpe_idx],
        all_sharpes[max_sharpe_idx], all_weights[max_sharpe_idx],
    )

    # Extract frontier points (efficient boundary)
    ret_min = float(min_variance.expected_return)
    ret_max = float(all_returns.max())
    ret_bins = np.linspace(ret_min, ret_max, n_points)

    frontier_points: list[FrontierPortfolio] = []
    seen_keys: set[tuple[float, float]] = set()

    for target_ret in ret_bins:
        tolerance = (ret_max - ret_min) / (n_points * 2)
        mask = np.abs(all_returns - target_ret) < tolerance
        if mask.any():
            subset_vols = all_vols[mask]
            best_idx = np.where(mask)[0][int(np.argmin(subset_vols))]
            key = (round(float(all_vols[best_idx]), 6), round(float(all_returns[best_idx]), 6))
            if key not in seen_keys:
                seen_keys.add(key)
                frontier_points.append(FrontierPortfolio(
                    all_returns[best_idx], all_vols[best_idx],
                    all_sharpes[best_idx], all_weights[best_idx],
                ))

    frontier_points.sort(key=lambda p: p.volatility)

    # Individual asset stats
    individual_assets = []
    for i in range(n_assets):
        asset_ret = float(mean_returns[i] * trading_days)
        asset_vol = float(np.sqrt(cov_matrix[i][i] * trading_days))
        individual_assets.append({
            "index": i,
            "return": asset_ret,
            "volatility": asset_vol,
        })

    return FrontierResult(
        frontier_points=frontier_points,
        current=current,
        min_variance=min_variance,
        max_sharpe=max_sharpe,
        individual_assets=individual_assets,
        n_simulations=n_simulations,
        risk_free_rate=risk_free_rate,
    )
