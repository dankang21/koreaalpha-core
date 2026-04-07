"""Benchmark portfolio comparison module.

Provides pure comparison logic and grade calculation only.
Benchmark portfolio definitions (stock codes, weights, ETF mappings) are managed at the service level.
"""

from dataclasses import dataclass

from .metrics import PortfolioMetrics, calculate_all_metrics


@dataclass
class ComparisonResult:
    """Portfolio vs benchmark comparison result."""

    benchmark_name: str
    user_metrics: PortfolioMetrics
    benchmark_metrics: PortfolioMetrics
    sharpe_diff: float  # User - Benchmark
    volatility_ratio: float  # User volatility / Benchmark volatility
    mdd_diff: float  # User MDD - Benchmark MDD
    cagr_diff: float  # User CAGR - Benchmark CAGR
    grade: str  # A+, A, B+, B, C, D, F


def grade_portfolio(user: PortfolioMetrics, bench: PortfolioMetrics) -> str:
    """Assign a grade relative to the benchmark.

    Weighted evaluation: Sharpe 40% + CAGR 30% + MDD 30%.

    Returns:
        A+, A, B+, B, C, D, F
    """
    score = 0

    # Sharpe Ratio comparison (40% weight)
    if bench.sharpe_ratio != 0:
        sharpe_ratio = user.sharpe_ratio / bench.sharpe_ratio
        if sharpe_ratio >= 1.3:
            score += 40
        elif sharpe_ratio >= 1.1:
            score += 32
        elif sharpe_ratio >= 0.9:
            score += 24
        elif sharpe_ratio >= 0.7:
            score += 16
        else:
            score += 8
    else:
        if user.sharpe_ratio >= 1.0:
            score += 36
        elif user.sharpe_ratio >= 0.5:
            score += 28
        elif user.sharpe_ratio >= 0:
            score += 20
        else:
            score += 8

    # CAGR comparison (30% weight)
    cagr_diff = user.cagr - bench.cagr
    if cagr_diff >= 0.03:
        score += 30
    elif cagr_diff >= 0.01:
        score += 24
    elif cagr_diff >= -0.01:
        score += 18
    elif cagr_diff >= -0.03:
        score += 12
    else:
        score += 6

    # MDD comparison (30% weight)
    mdd_diff = user.mdd - bench.mdd
    if mdd_diff >= 0.05:
        score += 30
    elif mdd_diff >= 0.02:
        score += 24
    elif mdd_diff >= -0.02:
        score += 18
    elif mdd_diff >= -0.05:
        score += 12
    else:
        score += 6

    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B+"
    elif score >= 60:
        return "B"
    elif score >= 50:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"


def compare_with_benchmark(
    user_prices: list[float],
    benchmark_prices: list[float],
    benchmark_name: str = "Benchmark",
    risk_free_rate: float = 0.035,
) -> ComparisonResult:
    """Compare a user portfolio against a benchmark.

    Args:
        user_prices: User portfolio daily value array
        benchmark_prices: Benchmark daily value array
        benchmark_name: Benchmark display name
        risk_free_rate: Risk-free rate

    Returns:
        ComparisonResult
    """
    user_m = calculate_all_metrics(user_prices, risk_free_rate)
    bench_m = calculate_all_metrics(benchmark_prices, risk_free_rate)

    vol_ratio = (
        user_m.volatility / bench_m.volatility
        if bench_m.volatility != 0
        else 0.0
    )

    return ComparisonResult(
        benchmark_name=benchmark_name,
        user_metrics=user_m,
        benchmark_metrics=bench_m,
        sharpe_diff=round(user_m.sharpe_ratio - bench_m.sharpe_ratio, 4),
        volatility_ratio=round(vol_ratio, 4),
        mdd_diff=round(user_m.mdd - bench_m.mdd, 6),
        cagr_diff=round(user_m.cagr - bench_m.cagr, 6),
        grade=grade_portfolio(user_m, bench_m),
    )


def compare_with_multiple(
    user_prices: list[float],
    benchmarks: dict[str, list[float]],
    risk_free_rate: float = 0.035,
) -> list[ComparisonResult]:
    """Compare against multiple benchmarks.

    Args:
        user_prices: User portfolio daily values
        benchmarks: {benchmark_name: daily_values} dictionary
        risk_free_rate: Risk-free rate

    Returns:
        List of ComparisonResult
    """
    return [
        compare_with_benchmark(user_prices, prices, name, risk_free_rate)
        for name, prices in benchmarks.items()
    ]
