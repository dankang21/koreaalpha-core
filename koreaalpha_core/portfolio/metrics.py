"""Portfolio performance metrics calculation module.

Calculates core metrics including Sharpe Ratio, MDD, Sortino Ratio, CAGR,
volatility, beta, correlation, etc. NumPy-based, no pandas dependency.
"""

import math
from dataclasses import dataclass

import numpy as np


TRADING_DAYS_US = 252  # US annual trading days
TRADING_DAYS_KR = 248  # Korea annual trading days
TRADING_DAYS = TRADING_DAYS_KR  # Default: Korea
DEFAULT_RISK_FREE_RATE = 0.035  # Based on Korean government bond; modify this constant to change


@dataclass
class PortfolioMetrics:
    """Comprehensive portfolio performance metrics."""

    cagr: float  # Compound Annual Growth Rate
    volatility: float  # Annualized volatility
    sharpe_ratio: float  # Sharpe Ratio
    sortino_ratio: float  # Sortino Ratio
    mdd: float  # Maximum Drawdown (negative)
    mdd_duration_days: int  # Days to recover from MDD
    calmar_ratio: float  # CAGR / |MDD|
    total_return: float  # Total return
    best_day: float  # Best daily return
    worst_day: float  # Worst daily return
    positive_days_pct: float  # Percentage of positive return days


def calculate_returns(prices: list[float] | np.ndarray) -> np.ndarray:
    """Calculate daily returns from a price time series.

    Args:
        prices: Daily closing price array (minimum 2 elements)

    Returns:
        Daily returns array (len = len(prices) - 1)
    """
    p = np.asarray(prices, dtype=np.float64)
    if len(p) < 2:
        raise ValueError("At least 2 price data points are required.")
    if np.any(p[:-1] == 0):
        raise ValueError("Prices contain zero; cannot compute returns.")
    return np.diff(p) / p[:-1]


def calculate_cagr(
    start_value: float, end_value: float, years: float
) -> float:
    """Calculate Compound Annual Growth Rate (CAGR).

    Args:
        start_value: Starting value
        end_value: Ending value
        years: Period in years

    Returns:
        CAGR (e.g., 0.07 = 7%)
    """
    if start_value <= 0 or years <= 0:
        raise ValueError("start_value and years must be positive.")
    return (end_value / start_value) ** (1 / years) - 1


def calculate_cagr_from_returns(
    returns: list[float] | np.ndarray,
) -> float:
    """Calculate CAGR from a daily returns array."""
    r = np.asarray(returns, dtype=np.float64)
    total = np.prod(1 + r)
    years = len(r) / TRADING_DAYS
    if years <= 0:
        return 0.0
    return total ** (1 / years) - 1


def calculate_volatility(
    returns: list[float] | np.ndarray,
) -> float:
    """Calculate annualized volatility.

    Args:
        returns: Daily returns array

    Returns:
        Annualized volatility (e.g., 0.15 = 15%)
    """
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        raise ValueError("At least 2 return data points are required.")
    return float(np.std(r, ddof=1) * math.sqrt(TRADING_DAYS))


def calculate_sharpe_ratio(
    returns: list[float] | np.ndarray,
    risk_free_rate: float = 0.035,
) -> float:
    """Calculate Sharpe Ratio.

    Args:
        returns: Daily returns array
        risk_free_rate: Annual risk-free rate (default 3.5%, Korean government bond)

    Returns:
        Annualized Sharpe Ratio
    """
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        raise ValueError("At least 2 return data points are required.")

    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = r - daily_rf
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(TRADING_DAYS))


def calculate_sortino_ratio(
    returns: list[float] | np.ndarray,
    risk_free_rate: float = 0.035,
) -> float:
    """Calculate Sortino Ratio.

    Uses only downside volatility to evaluate risk-adjusted returns.

    Args:
        returns: Daily returns array
        risk_free_rate: Annual risk-free rate

    Returns:
        Annualized Sortino Ratio
    """
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        raise ValueError("At least 2 return data points are required.")

    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = r - daily_rf
    downside = excess[excess < 0]

    if len(downside) == 0:
        return float("inf")  # No downside

    downside_std = np.sqrt(np.mean(downside**2))
    if downside_std == 0:
        return 0.0
    return float(np.mean(excess) / downside_std * math.sqrt(TRADING_DAYS))


def calculate_mdd(prices: list[float] | np.ndarray) -> float:
    """Calculate Maximum Drawdown (MDD).

    Args:
        prices: Daily closing price array

    Returns:
        MDD (negative, e.g., -0.15 = -15%)
    """
    p = np.asarray(prices, dtype=np.float64)
    if len(p) < 2:
        raise ValueError("At least 2 price data points are required.")

    peak = np.maximum.accumulate(p)
    # Guard against peak being 0
    safe_peak = np.where(peak == 0, 1, peak)
    drawdown = (p - peak) / safe_peak
    return float(np.min(drawdown))


def calculate_mdd_duration(prices: list[float] | np.ndarray) -> int:
    """Calculate the maximum number of days to recover from MDD.

    Args:
        prices: Daily closing price array

    Returns:
        Maximum drawdown duration (in trading days)
    """
    p = np.asarray(prices, dtype=np.float64)
    peak = np.maximum.accumulate(p)
    in_drawdown = p < peak

    max_duration = 0
    current = 0
    for underwater in in_drawdown:
        if underwater:
            current += 1
            max_duration = max(max_duration, current)
        else:
            current = 0
    return max_duration


def calculate_calmar_ratio(cagr: float, mdd: float) -> float:
    """Calculate Calmar Ratio (CAGR / |MDD|).

    Args:
        cagr: Compound Annual Growth Rate
        mdd: Maximum Drawdown (negative)

    Returns:
        Calmar Ratio
    """
    if mdd == 0:
        return float("inf") if cagr > 0 else 0.0
    return cagr / abs(mdd)


def calculate_beta(
    returns: list[float] | np.ndarray,
    benchmark_returns: list[float] | np.ndarray,
) -> float:
    """Calculate beta relative to a benchmark.

    Args:
        returns: Portfolio daily returns
        benchmark_returns: Benchmark daily returns

    Returns:
        Beta coefficient
    """
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)

    min_len = min(len(r), len(b))
    r, b = r[:min_len], b[:min_len]

    cov = np.cov(r, b)
    var_benchmark = cov[1, 1]
    if var_benchmark == 0:
        return 0.0
    return float(cov[0, 1] / var_benchmark)


def calculate_correlation(
    returns_a: list[float] | np.ndarray,
    returns_b: list[float] | np.ndarray,
) -> float:
    """Calculate the correlation coefficient between two return series.

    Returns:
        Pearson correlation coefficient (-1 to 1)
    """
    a = np.asarray(returns_a, dtype=np.float64)
    b = np.asarray(returns_b, dtype=np.float64)
    min_len = min(len(a), len(b))
    return float(np.corrcoef(a[:min_len], b[:min_len])[0, 1])


def calculate_all_metrics(
    prices: list[float] | np.ndarray,
    risk_free_rate: float = 0.035,
) -> PortfolioMetrics:
    """Calculate all core metrics from a price time series at once.

    Args:
        prices: Daily closing price array
        risk_free_rate: Annual risk-free rate

    Returns:
        PortfolioMetrics object
    """
    p = np.asarray(prices, dtype=np.float64)
    returns = calculate_returns(p)
    years = len(returns) / TRADING_DAYS

    total_return = (p[-1] / p[0]) - 1
    cagr = calculate_cagr(p[0], p[-1], years) if years > 0 else 0.0
    vol = calculate_volatility(returns)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate)
    sortino = calculate_sortino_ratio(returns, risk_free_rate)
    mdd = calculate_mdd(p)
    mdd_dur = calculate_mdd_duration(p)
    calmar = calculate_calmar_ratio(cagr, mdd)

    return PortfolioMetrics(
        cagr=round(cagr, 6),
        volatility=round(vol, 6),
        sharpe_ratio=round(sharpe, 4),
        sortino_ratio=round(sortino, 4),
        mdd=round(mdd, 6),
        mdd_duration_days=mdd_dur,
        calmar_ratio=round(calmar, 4),
        total_return=round(total_return, 6),
        best_day=round(float(np.max(returns)), 6),
        worst_day=round(float(np.min(returns)), 6),
        positive_days_pct=round(float(np.mean(returns > 0)), 4),
    )


# -- Rolling metrics --

def rolling_sharpe(
    returns: list[float] | np.ndarray,
    window: int = 60,
    risk_free_rate: float = 0.035,
) -> np.ndarray:
    """Calculate Rolling Sharpe Ratio.

    Args:
        returns: Daily returns
        window: Window size (default 60 trading days ~ 3 months)
        risk_free_rate: Annual risk-free rate

    Returns:
        Rolling Sharpe array (first window-1 elements are NaN)
    """
    r = np.asarray(returns, dtype=np.float64)
    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = r - daily_rf
    result = np.full(len(r), np.nan)
    for i in range(window - 1, len(r)):
        w = excess[i - window + 1 : i + 1]
        std = np.std(w, ddof=1)
        result[i] = float(np.mean(w) / std * math.sqrt(TRADING_DAYS)) if std > 0 else 0.0
    return result


def rolling_volatility(
    returns: list[float] | np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """Calculate Rolling annualized volatility.

    Args:
        returns: Daily returns
        window: Window size (default 20 trading days ~ 1 month)

    Returns:
        Rolling volatility array
    """
    r = np.asarray(returns, dtype=np.float64)
    result = np.full(len(r), np.nan)
    for i in range(window - 1, len(r)):
        w = r[i - window + 1 : i + 1]
        result[i] = float(np.std(w, ddof=1) * math.sqrt(TRADING_DAYS))
    return result


def rolling_beta(
    returns: list[float] | np.ndarray,
    benchmark_returns: list[float] | np.ndarray,
    window: int = 60,
) -> np.ndarray:
    """Calculate Rolling Beta."""
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)
    min_len = min(len(r), len(b))
    r, b = r[:min_len], b[:min_len]
    result = np.full(min_len, np.nan)
    for i in range(window - 1, min_len):
        wr = r[i - window + 1 : i + 1]
        wb = b[i - window + 1 : i + 1]
        cov = np.cov(wr, wb)
        var_b = cov[1, 1]
        result[i] = float(cov[0, 1] / var_b) if var_b > 0 else 0.0
    return result


# -- Drawdown series --

def drawdown_series(prices: list[float] | np.ndarray) -> np.ndarray:
    """Return the full drawdown time series.

    Returns:
        Drawdown ratio array at each point (0 or negative, e.g., -0.15)
    """
    p = np.asarray(prices, dtype=np.float64)
    peak = np.maximum.accumulate(p)
    safe_peak = np.where(peak == 0, 1, peak)
    return (p - peak) / safe_peak


# -- VaR / CVaR --

def calculate_var(
    returns: list[float] | np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Calculate Value at Risk (historical method).

    Args:
        returns: Daily returns
        confidence: Confidence level (default 95%)

    Returns:
        VaR (negative, e.g., -0.025 = potential 2.5% daily loss)
    """
    r = np.asarray(returns, dtype=np.float64)
    return float(np.percentile(r, (1 - confidence) * 100))


def calculate_cvar(
    returns: list[float] | np.ndarray,
    confidence: float = 0.95,
) -> float:
    """Calculate Conditional VaR (Expected Shortfall).

    Average of losses exceeding VaR.

    Returns:
        CVaR (negative)
    """
    r = np.asarray(returns, dtype=np.float64)
    var = calculate_var(r, confidence)
    tail = r[r <= var]
    return float(np.mean(tail)) if len(tail) > 0 else var


# -- Distribution metrics --

def calculate_skewness(returns: list[float] | np.ndarray) -> float:
    """Calculate skewness of the return distribution.

    Positive: right tail (favorable), Negative: left tail (unfavorable)
    """
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    if n < 3:
        return 0.0
    mean = np.mean(r)
    std = np.std(r, ddof=1)
    if std == 0:
        return 0.0
    return float((n / ((n - 1) * (n - 2))) * np.sum(((r - mean) / std) ** 3))


def calculate_kurtosis(returns: list[float] | np.ndarray) -> float:
    """Calculate excess kurtosis of the return distribution.

    Greater than 0 indicates heavier tails than the normal distribution (fat tails).
    Fisher sample correction applied (consistent with skewness).
    """
    r = np.asarray(returns, dtype=np.float64)
    n = len(r)
    if n < 4:
        return 0.0
    mean = np.mean(r)
    std = np.std(r, ddof=1)
    if std == 0:
        return 0.0
    m4 = float(np.sum(((r - mean) / std) ** 4))
    # Fisher sample correction
    kurt = (n * (n + 1) * m4) / ((n - 1) * (n - 2) * (n - 3))
    adjust = (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return kurt - adjust


# -- Alpha / Information Ratio --

def calculate_alpha(
    returns: list[float] | np.ndarray,
    benchmark_returns: list[float] | np.ndarray,
    risk_free_rate: float = 0.035,
) -> float:
    """Calculate Jensen's Alpha.

    Alpha = Portfolio excess return - Beta * Benchmark excess return
    """
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)
    min_len = min(len(r), len(b))
    r, b = r[:min_len], b[:min_len]

    daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    port_excess = np.mean(r) - daily_rf
    bench_excess = np.mean(b) - daily_rf
    beta = calculate_beta(r, b)

    return float((port_excess - beta * bench_excess) * TRADING_DAYS)


def calculate_information_ratio(
    returns: list[float] | np.ndarray,
    benchmark_returns: list[float] | np.ndarray,
) -> float:
    """Calculate Information Ratio.

    Excess return / Tracking error
    """
    r = np.asarray(returns, dtype=np.float64)
    b = np.asarray(benchmark_returns, dtype=np.float64)
    min_len = min(len(r), len(b))
    active = r[:min_len] - b[:min_len]
    te = np.std(active, ddof=1)
    if te == 0:
        return 0.0
    return float(np.mean(active) / te * math.sqrt(TRADING_DAYS))


# -- Other ratios --

def calculate_omega_ratio(
    returns: list[float] | np.ndarray,
    threshold: float = 0.0,
) -> float:
    """Calculate Omega Ratio.

    Sum of gains above threshold / Sum of losses below threshold
    """
    r = np.asarray(returns, dtype=np.float64)
    gains = np.sum(np.maximum(r - threshold, 0))
    losses = np.sum(np.maximum(threshold - r, 0))
    if losses == 0:
        return float("inf")
    return float(gains / losses)


def calculate_tail_ratio(
    returns: list[float] | np.ndarray,
    percentile: float = 5.0,
) -> float:
    """Tail Ratio -- upper tail / lower tail ratio.

    Greater than 1 means the upper tail is thicker (favorable)
    """
    r = np.asarray(returns, dtype=np.float64)
    upper = np.abs(np.percentile(r, 100 - percentile))
    lower = np.abs(np.percentile(r, percentile))
    if lower == 0:
        return float("inf")
    return float(upper / lower)


# -- Monthly / Annual returns --

def monthly_returns(
    prices: list[float] | np.ndarray,
    dates: list[str] | None = None,
) -> list[dict]:
    """Calculate monthly returns.

    Args:
        prices: Daily prices
        dates: Date list in YYYYMMDD format (None for index-based)

    Returns:
        [{"year": 2025, "month": 1, "return": 0.05}, ...]
    """
    p = np.asarray(prices, dtype=np.float64)
    if dates is None or len(dates) != len(p):
        return []

    result = []
    month_start_price = p[0]
    current_ym = dates[0][:6]

    for i in range(1, len(p)):
        ym = dates[i][:6]
        if ym != current_ym:
            ret = (p[i - 1] / month_start_price) - 1
            result.append({
                "year": int(current_ym[:4]),
                "month": int(current_ym[4:6]),
                "return": round(ret, 6),
            })
            month_start_price = p[i]  # New month start price = first day of new month
            current_ym = ym

    # Last month
    ret = (p[-1] / month_start_price) - 1 if month_start_price > 0 else 0.0
    result.append({
        "year": int(current_ym[:4]),
        "month": int(current_ym[4:6]),
        "return": round(ret, 6),
    })
    return result


def annual_returns(
    prices: list[float] | np.ndarray,
    dates: list[str] | None = None,
) -> list[dict]:
    """Calculate annual returns.

    Returns:
        [{"year": 2025, "return": 0.12}, ...]
    """
    p = np.asarray(prices, dtype=np.float64)
    if dates is None or len(dates) != len(p):
        return []

    result = []
    year_start_price = p[0]
    current_year = dates[0][:4]

    for i in range(1, len(p)):
        yr = dates[i][:4]
        if yr != current_year:
            ret = (p[i - 1] / year_start_price) - 1
            result.append({"year": int(current_year), "return": round(ret, 6)})
            year_start_price = p[i]  # New year start price
            current_year = yr

    ret = (p[-1] / year_start_price) - 1 if year_start_price > 0 else 0.0
    result.append({"year": int(current_year), "return": round(ret, 6)})
    return result


# -- Win / Loss streaks --

def longest_streak(returns: list[float] | np.ndarray) -> dict:
    """Calculate the longest consecutive win/loss streaks in days.

    Returns:
        {"win_streak": int, "loss_streak": int}
    """
    r = np.asarray(returns, dtype=np.float64)
    max_win = max_loss = 0
    cur_win = cur_loss = 0
    for ret in r:
        if ret > 0:
            cur_win += 1
            cur_loss = 0
        elif ret < 0:
            cur_loss += 1
            cur_win = 0
        else:
            cur_win = cur_loss = 0
        max_win = max(max_win, cur_win)
        max_loss = max(max_loss, cur_loss)
    return {"win_streak": max_win, "loss_streak": max_loss}


# -- Correlation matrix (N x N) --

def correlation_matrix(
    returns_dict: dict[str, list[float] | np.ndarray],
) -> dict:
    """Calculate the correlation matrix for multiple assets.

    Args:
        returns_dict: {"asset_name": [returns, ...], ...}

    Returns:
        {"labels": [...], "matrix": [[...], ...]}
    """
    if not returns_dict:
        return {"labels": [], "matrix": []}
    labels = list(returns_dict.keys())
    n = len(labels)
    if n == 1:
        return {"labels": labels, "matrix": [[1.0]]}
    arrays = []
    min_len = min(len(v) for v in returns_dict.values())
    for label in labels:
        arr = np.asarray(returns_dict[label], dtype=np.float64)[:min_len]
        arrays.append(arr)

    mat = np.corrcoef(arrays)
    return {
        "labels": labels,
        "matrix": [[round(float(mat[i][j]), 4) for j in range(n)] for i in range(n)],
    }
