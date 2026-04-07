"""koreaalpha-core: Korean stock market portfolio analysis engine."""

__version__ = "0.4.0"

from .portfolio.metrics import (
    TRADING_DAYS, TRADING_DAYS_KR, TRADING_DAYS_US,
    PortfolioMetrics,
    calculate_all_metrics,
    calculate_returns,
    calculate_cagr,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_mdd,
    calculate_mdd_duration,
    calculate_calmar_ratio,
    calculate_beta,
    calculate_correlation,
    # Rolling
    rolling_sharpe,
    rolling_volatility,
    rolling_beta,
    # Drawdown
    drawdown_series,
    # VaR
    calculate_var,
    calculate_cvar,
    # Distribution
    calculate_skewness,
    calculate_kurtosis,
    # Alpha
    calculate_alpha,
    calculate_information_ratio,
    # Ratios
    calculate_omega_ratio,
    calculate_tail_ratio,
    # Returns matrix
    monthly_returns,
    annual_returns,
    # Streak
    longest_streak,
    # Correlation matrix
    correlation_matrix,
)
from .portfolio.backtest import run_backtest, BacktestResult, BacktestConfig
from .portfolio.benchmark import (
    ComparisonResult,
    compare_with_benchmark,
    compare_with_multiple,
    grade_portfolio,
)
from .stock.fundamental import calculate_all_fundamentals, FundamentalMetrics
from .kr_market import (
    is_kr_trading_day,
    count_trading_days,
    get_trading_days,
    calc_transaction_cost,
    calc_round_trip_cost,
    calc_dividend_tax,
    calc_overseas_cgt,
    calc_after_tax_return,
    DEFAULT_SECURITIES_TAX_RATE,
    DEFAULT_DIVIDEND_TAX_RATE,
    DEFAULT_OVERSEAS_CGT_RATE,
    DEFAULT_FINANCIAL_INCOME_THRESHOLD,
)
