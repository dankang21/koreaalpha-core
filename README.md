# koreaalpha-core

**Korean stock market portfolio analysis engine**

A portfolio analysis engine specialized for the Korean stock market. A pure math/statistics calculation library.

[![Python](https://img.shields.io/pypi/pyversions/koreaalpha-core)](https://pypi.org/project/koreaalpha-core/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-83%20passed-brightgreen)]()

## Features

- **Portfolio Metrics (28)** — CAGR, Sharpe, Sortino, MDD, Calmar, Beta, VaR, CVaR, Alpha, Omega, Skewness, Kurtosis, etc.
- **Rolling Indicators** — Rolling Sharpe, Volatility, Beta (for chart visualization)
- **Backtesting** — Rebalancing, transaction costs, slippage, Korean trading day-based, dividend reinvestment
- **Benchmark Comparison** — Pure comparison logic + grade calculation (A+ to F) (benchmark data defined at service level)
- **Fundamental** — PER, PBR, ROE, ROA, FCF yield (supports negative PER for loss-making companies)
- **Korean Market** — Automatic Korean trading day calculation, securities transaction tax, dividend tax, overseas capital gains tax, after-tax returns
- **Zero pandas dependency** — Uses only numpy, lightweight

## Installation

```bash
pip install -e .  # Local install (private package)
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

# Benchmark name and price data are passed from the service layer
result = compare_with_benchmark(
    user_prices=[...],
    benchmark_prices=[...],
    benchmark_name="올웨더 포트폴리오",
)
print(f"Grade: {result.grade}")       # A+, A, B+, B, C, D, F
print(f"Sharpe diff: {result.sharpe_diff:+.4f}")
print(f"CAGR diff: {result.cagr_diff:+.2%}")
```

### Backtesting

```python
from koreaalpha_core import run_backtest, BacktestConfig

result = run_backtest(
    asset_prices={"삼성전자": [...], "SK하이닉스": [...]},
    allocations={"삼성전자": 0.6, "SK하이닉스": 0.4},
    config=BacktestConfig(
        initial_capital=10_000_000,
        rebalance_period="quarterly",      # monthly/quarterly/yearly/none
        transaction_cost_pct=0.0015,
        use_kr_trading_days=True,          # Rebalance based on Korean trading days
        dividend_reinvest=True,            # Reinvest dividends
        dividend_yields={"삼성전자": 0.02},
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
cost = calc_transaction_cost(10_000_000, is_sell=True, securities_tax_rate=0.0015)  # Custom tax rate

# Dividend income tax
tax = calc_dividend_tax(25_000_000)
# {"gross": 25000000, "tax": 3850000, "net": 21150000, "is_over_threshold": True}
tax = calc_dividend_tax(25_000_000, tax_rate=0.14)  # Tax rate override

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
    calculate_alpha,           # Jensen's Alpha
    calculate_information_ratio,  # Information Ratio
    calculate_omega_ratio,     # Omega Ratio
    calculate_tail_ratio,      # Tail Ratio
    monthly_returns,           # Monthly return matrix
    annual_returns,            # Annual returns
    longest_streak,            # Longest win/loss streak
    correlation_matrix,        # N×N correlation matrix
    grade_portfolio,           # Grade vs benchmark (A+ to F)
    compare_with_multiple,     # Compare against multiple benchmarks
)
```

## Architecture

```
korean-holidays (PyPI, MIT)
  └── Lunar calendar conversion + automatic substitute holiday calculation
       ↑
koreaalpha-core (private, Proprietary)
  ├── portfolio/metrics.py   — 28 analysis functions
  ├── portfolio/backtest.py  — Backtesting engine
  ├── portfolio/benchmark.py — Pure comparison logic (no data)
  ├── stock/fundamental.py   — Fundamental indicators
  ├── kr_market.py           — Transaction costs / taxes (parameterized)
  ├── utils/                 — Formatting / validation
  └── 83 tests
```

## Design Principles

- **Pure calculation library** — No API calls, no DB access, no authentication
- **Data-logic separation** — Benchmark definitions, holidays, and colors are managed at the service level
- **Tax rates: defaults + override** — Callers can pass parameters when policy changes
- **No pandas dependency** — Uses only numpy, lightweight
- **Korean market defaults** — TRADING_DAYS=248, risk-free rate=3.5%

## Comparison with Alternatives

| Feature | koreaalpha-core | quantstats | empyrical |
|---------|:---:|:---:|:---:|
| Korean trading calendar | O | X | X |
| Transaction tax (parameterized) | O | X | X |
| Dividend/CGT tax calculator | O | X | X |
| Backtesting with KR holidays | O | X | X |
| VaR/CVaR/Skewness/Kurtosis | O | O | O |
| Rolling indicators | O | O | X |
| Fundamental analysis | O | X | X |
| pandas-free | O | X | X |
| Data-logic separation | O | X | X |

## Disclaimer

This library is a technical tool for investment analysis and does not provide investment advice or financial services. All investment decisions are the sole responsibility of the user.

## License

MIT License. See [LICENSE](LICENSE).
