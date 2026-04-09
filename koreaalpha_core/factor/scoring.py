"""Factor scoring engine — momentum, value, quality, growth.

Pure computation. Accepts pre-fetched financial data as input.
No yfinance or API calls.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..portfolio.metrics import TRADING_DAYS_KR


@dataclass
class FactorScores:
    """Factor scores for a single asset."""
    momentum: float
    value: float
    quality: float
    growth: float

    @property
    def composite(self) -> float:
        """Equal-weighted composite score."""
        return (self.momentum + self.value + self.quality + self.growth) / 4

    def weighted_composite(
        self,
        w_momentum: float = 0.25,
        w_value: float = 0.25,
        w_quality: float = 0.25,
        w_growth: float = 0.25,
    ) -> float:
        """Weighted composite score."""
        total = w_momentum + w_value + w_quality + w_growth or 1.0
        return (
            self.momentum * w_momentum
            + self.value * w_value
            + self.quality * w_quality
            + self.growth * w_growth
        ) / total


def calculate_momentum(
    prices: np.ndarray,
    trading_days: int = TRADING_DAYS_KR,
) -> float:
    """Calculate momentum score from price array.

    Uses weighted 12M/6M/3M returns (40%/35%/25%).

    Args:
        prices: Daily closing prices (oldest first).
        trading_days: Annual trading days for period estimation.

    Returns:
        Momentum score (scaled).
    """
    if len(prices) < 60:
        return 0.0

    ret_full = prices[-1] / prices[0] - 1
    idx_6m = max(0, len(prices) - min(trading_days // 2, len(prices)))
    idx_3m = max(0, len(prices) - min(trading_days // 4, len(prices)))

    ret_6m = prices[-1] / prices[idx_6m] - 1
    ret_3m = prices[-1] / prices[idx_3m] - 1

    momentum = ret_full * 0.4 + ret_6m * 0.35 + ret_3m * 0.25
    return round(float(momentum) * 5, 2)


def calculate_value_score(
    per: float | None = None,
    pbr: float | None = None,
) -> float:
    """Calculate value factor score from PER and PBR.

    Lower PER/PBR → higher value score.

    Args:
        per: Price-to-Earnings ratio (trailing or forward).
        pbr: Price-to-Book ratio.

    Returns:
        Value score (0 to ~2.0).
    """
    score = 0.0
    if per and per > 0:
        score += min(1.0 / per * 15, 2.0)
    if pbr and pbr > 0:
        score += min(1.0 / pbr, 2.0)
    return round(score / 2 if score else 0.0, 2)


def calculate_quality_score(
    roe: float | None = None,
    volatility: float = 0.0,
) -> float:
    """Calculate quality factor score from ROE and volatility.

    Higher ROE and lower volatility → higher quality.

    Args:
        roe: Return on Equity (as decimal, e.g. 0.15 for 15%).
        volatility: Annualized volatility (as decimal).

    Returns:
        Quality score.
    """
    score = 0.0
    if roe and roe > 0:
        score = min(roe * 5, 2.0)
    score -= min(volatility * 0.5, 1.0)
    return round(float(score), 2)


def calculate_growth_score(
    revenue_growth: float | None = None,
    earnings_growth: float | None = None,
) -> float:
    """Calculate growth factor score.

    Args:
        revenue_growth: Revenue growth rate (as decimal).
        earnings_growth: Earnings growth rate (as decimal).

    Returns:
        Growth score.
    """
    score = 0.0
    if revenue_growth is not None:
        score += min(float(revenue_growth) * 3, 2.0)
    if earnings_growth is not None:
        score += min(float(earnings_growth) * 2, 2.0)
    return round(score / 2 if score else 0.0, 2)


def calculate_factor_scores(
    prices: np.ndarray,
    per: float | None = None,
    pbr: float | None = None,
    roe: float | None = None,
    revenue_growth: float | None = None,
    earnings_growth: float | None = None,
    trading_days: int = TRADING_DAYS_KR,
) -> FactorScores:
    """Calculate all factor scores for an asset.

    Args:
        prices: Daily closing prices (oldest first).
        per: Price-to-Earnings ratio.
        pbr: Price-to-Book ratio.
        roe: Return on Equity (decimal).
        revenue_growth: Revenue growth rate (decimal).
        earnings_growth: Earnings growth rate (decimal).
        trading_days: Annual trading days.

    Returns:
        FactorScores dataclass.
    """
    returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else np.array([])
    vol = float(np.std(returns) * np.sqrt(trading_days)) if len(returns) > 0 else 0.0

    return FactorScores(
        momentum=calculate_momentum(prices, trading_days),
        value=calculate_value_score(per, pbr),
        quality=calculate_quality_score(roe, vol),
        growth=calculate_growth_score(revenue_growth, earnings_growth),
    )
