# koreaalpha-core

**Korean stock market portfolio analysis engine**

A portfolio analysis engine specialized for the Korean stock market. A pure math/statistics calculation library.

[![PyPI](https://img.shields.io/pypi/v/koreaalpha-core)](https://pypi.org/project/koreaalpha-core/)
[![Python](https://img.shields.io/pypi/pyversions/koreaalpha-core)](https://pypi.org/project/koreaalpha-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-108%20passed-brightgreen)]()

## Features

- **Portfolio Metrics (28)** — CAGR, Sharpe, Sortino, MDD, Calmar, Beta, VaR, CVaR, Alpha, Omega, Skewness, Kurtosis, etc.
- **Efficient Frontier** — Monte Carlo simulation, min-variance / max-Sharpe portfolio optimization
- **Portfolio Scoring** — Rule-based scoring (0-100) and grading (A+ to F), deterministic and AI-independent
- **Factor Analysis** — Momentum, value, quality, growth factor scoring engine
- **Rolling Indicators** — Rolling Sharpe, Volatility, Beta (for chart visualization)
- **Backtesting** — Rebalancing, transaction costs, slippage, Korean trading day-based, dividend reinvestment
- **Benchmark Comparison** — Pure comparison logic + grade calculation (A+ to F)
- **Fundamental** — PER, PBR, ROE, ROA, FCF yield (supports negative PER for loss-making companies)
- **Korean Market** — Automatic Korean trading day calculation, securities transaction tax, dividend tax, overseas capital gains tax, after-tax returns
- **Zero pandas dependency** — Uses only numpy, lightweight

## Installation

```bash
pip install koreaalpha-core
```

## Quick Start

### Portfolio Metrics

```python
from koreaalpha_core import calculate_all_metrics

prices = [1000, 1020, 1015, 1050, 1080, 1070, 1100, ...]
metrics = calculate_all_metrics(prices)

metrics.cagr           # Compound Annual Growth Rate
metrics.sharpe_ratio   # Sharpe Ratio
metrics.sortino_ratio  # Sortino Ratio
metrics.mdd            # Maximum Drawdown
metrics.volatility     # Annualized Volatility
metrics.calmar_ratio   # CAGR / |MDD|
```

### Efficient Frontier

```python
import numpy as np
from koreaalpha_core import calculate_efficient_frontier

# Daily mean returns and covariance matrix (from your data pipeline)
mean_returns = np.array([0.0005, 0.0003, 0.0001])
cov_matrix = np.array([
    [0.0004, 0.0001, 0.00005],
    [0.0001, 0.0002, 0.00008],
    [0.00005, 0.00008, 0.0001],
])
current_weights = np.array([0.5, 0.3, 0.2])

result = calculate_efficient_frontier(mean_returns, cov_matrix, current_weights)

print(f"Max Sharpe: {result.max_sharpe.sharpe:.4f}")
print(f"Min Variance Vol: {result.min_variance.volatility:.4f}")
print(f"Frontier points: {len(result.frontier_points)}")
```

### Portfolio Scoring

```python
from koreaalpha_core import calculate_portfolio_score

metrics = {
    "sharpe_ratio": 1.2,
    "mdd": -0.12,
    "cagr": 0.10,
    "sortino_ratio": 1.5,
    "calmar_ratio": 0.8,
}
score, grade = calculate_portfolio_score(metrics)
print(f"Score: {score}/100, Grade: {grade}")  # e.g. Score: 78/100, Grade: B+
```

### Factor Scoring

```python
import numpy as np
from koreaalpha_core import calculate_factor_scores

prices = np.array([...])  # 1 year of daily closing prices
scores = calculate_factor_scores(
    prices, per=12.5, pbr=1.2, roe=0.18,
    revenue_growth=0.15, earnings_growth=0.20,
)
print(f"Momentum: {scores.momentum}")
print(f"Value: {scores.value}")
print(f"Quality: {scores.quality}")
print(f"Growth: {scores.growth}")
print(f"Composite: {scores.composite}")

# Custom factor weights
weighted = scores.weighted_composite(w_momentum=0.4, w_value=0.3, w_quality=0.2, w_growth=0.1)
```

### Rolling Indicators

```python
from koreaalpha_core import calculate_returns, rolling_sharpe, rolling_volatility

returns = calculate_returns(prices)
rs = rolling_sharpe(returns, window=60)     # 60-day rolling Sharpe
rv = rolling_volatility(returns, window=20) # 20-day rolling volatility
```

### Risk Metrics

```python
from koreaalpha_core import calculate_var, calculate_cvar, calculate_skewness, drawdown_series

var = calculate_var(returns, 0.95)    # 95% VaR
cvar = calculate_cvar(returns, 0.95)  # 95% CVaR (Expected Shortfall)
sk = calculate_skewness(returns)      # Skewness (negative = crash risk)
dd = drawdown_series(prices)          # Full drawdown time series
```

### Benchmark Comparison

```python
from koreaalpha_core import compare_with_benchmark

result = compare_with_benchmark(
    user_prices=[...],
    benchmark_prices=[...],
    benchmark_name="Balanced Portfolio",
)
print(f"Grade: {result.grade}")       # A+, A, B+, B, C, D, F
print(f"Sharpe diff: {result.sharpe_diff:+.4f}")
print(f"CAGR diff: {result.cagr_diff:+.2%}")
```

### Backtesting

```python
from koreaalpha_core import run_backtest, BacktestConfig

result = run_backtest(
    asset_prices={"AAPL": [...], "MSFT": [...]},
    allocations={"AAPL": 0.6, "MSFT": 0.4},
    config=BacktestConfig(
        initial_capital=10_000_000,
        rebalance_period="quarterly",      # monthly/quarterly/yearly/none
        transaction_cost_pct=0.0015,
        use_kr_trading_days=True,          # Rebalance based on Korean trading days
        dividend_reinvest=True,
        dividend_yields={"AAPL": 0.005},
    ),
    dates=["20240102", "20240103", ...],
)
print(f"Final value: {result.portfolio_values[-1]:,.0f} KRW")
print(f"Sharpe: {result.metrics.sharpe_ratio:.2f}")
```

### Korean Market

```python
from datetime import date
from koreaalpha_core import (
    is_kr_trading_day, count_trading_days,
    calc_transaction_cost, calc_dividend_tax, calc_after_tax_return,
)

# Trading days (delegated to korean-holidays package — auto-calculated for any year)
is_kr_trading_day(date(2030, 1, 1))  # False (New Year's Day)
count_trading_days(date(2026, 1, 1), date(2026, 12, 31))  # ~248 days

# Transaction cost (tax rates can be overridden via parameters)
cost = calc_transaction_cost(10_000_000, is_sell=True)  # Default 0.18%

# Dividend income tax
tax = calc_dividend_tax(25_000_000)
# {"gross": 25000000, "tax": 3850000, "net": 21150000, "is_over_threshold": True}

# After-tax return
result = calc_after_tax_return(0.10, 100_000_000, dividend_income=5_000_000)
# {"gross_return": 0.1, "after_tax_return": 0.09923, "total_tax": 770000}
```

### Fundamental Analysis

```python
from koreaalpha_core import calculate_all_fundamentals

metrics = calculate_all_fundamentals(
    price=55000, eps=5000, bps=40000,
    net_income=30e9, equity=200e9,
    total_assets=400e9, total_liabilities=200e9,
    revenue=300e9, operating_income=45e9,
    fcf=25e9, market_cap=330e12,
)
print(f"PER: {metrics.per}")           # 11.0 (returns negative for losses)
print(f"ROE: {metrics.roe:.2%}")       # 15.00%
print(f"Debt ratio: {metrics.debt_ratio:.2%}")  # 100.00%
```

### More

```python
from koreaalpha_core import (
    calculate_alpha,              # Jensen's Alpha
    calculate_information_ratio,  # Information Ratio
    calculate_omega_ratio,        # Omega Ratio
    calculate_tail_ratio,         # Tail Ratio
    monthly_returns,              # Monthly return matrix
    annual_returns,               # Annual returns
    longest_streak,               # Longest win/loss streak
    correlation_matrix,           # N x N correlation matrix
    grade_portfolio,              # Grade vs benchmark (A+ to F)
    compare_with_multiple,        # Compare against multiple benchmarks
    portfolio_stats,              # Single portfolio return/vol/Sharpe
    FrontierResult,               # Efficient frontier result dataclass
    FactorScores,                 # Factor scoring result dataclass
)
```

## Architecture

```
korean-holidays (PyPI, MIT)
  └── Lunar calendar conversion + automatic substitute holiday calculation
       |
koreaalpha-core (PyPI, MIT)
  ├── portfolio/metrics.py    — 28 portfolio analysis functions
  ├── portfolio/backtest.py   — Backtesting engine with KR trading days
  ├── portfolio/benchmark.py  — Pure comparison logic (no data)
  ├── portfolio/frontier.py   — Efficient frontier (Monte Carlo)
  ├── portfolio/score.py      — Rule-based scoring (0-100, A+ to F)
  ├── factor/scoring.py       — Momentum, value, quality, growth factors
  ├── stock/fundamental.py    — Fundamental indicators
  ├── kr_market.py            — Transaction costs / taxes (parameterized)
  ├── utils/                  — Formatting / validation
  └── 108 tests
```

## Design Principles

- **Pure calculation library** — No API calls, no DB access, no authentication
- **Data-logic separation** — Benchmark definitions, stock lists, and presets are managed at the service level
- **Tax rates: defaults + override** — Callers can pass parameters when policy changes
- **No pandas dependency** — Uses only numpy, lightweight
- **Korean market defaults** — TRADING_DAYS=248, risk-free rate=3.5%
- **Deterministic scoring** — Same input always produces same output, no AI dependency

## Comparison with Alternatives

| Feature | koreaalpha-core | quantstats | empyrical |
|---------|:---:|:---:|:---:|
| Korean trading calendar | O | X | X |
| Transaction tax (parameterized) | O | X | X |
| Dividend/CGT tax calculator | O | X | X |
| Efficient frontier | O | X | X |
| Portfolio scoring (0-100) | O | X | X |
| Factor analysis engine | O | X | X |
| Backtesting with KR holidays | O | X | X |
| VaR/CVaR/Skewness/Kurtosis | O | O | O |
| Rolling indicators | O | O | X |
| Fundamental analysis | O | X | X |
| pandas-free | O | X | X |

## Disclaimer

This library is a technical tool for investment analysis and does not provide investment advice or financial services. All investment decisions are the sole responsibility of the user.

## License

MIT License. See [LICENSE](LICENSE).
