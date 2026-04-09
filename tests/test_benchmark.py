"""Benchmark comparison tests."""

import numpy as np

from koreaalpha_core.portfolio.benchmark import (
    compare_with_benchmark,
    compare_with_multiple,
    grade_portfolio,
)
from koreaalpha_core.portfolio.metrics import calculate_all_metrics


class TestCompareWithBenchmark:
    def test_basic_comparison(self):
        np.random.seed(42)
        user = [1000]
        bench = [1000]
        for _ in range(251):
            user.append(user[-1] * (1 + np.random.normal(0.0005, 0.012)))
            bench.append(bench[-1] * (1 + np.random.normal(0.0003, 0.008)))

        result = compare_with_benchmark(user, bench, "Benchmark A")
        assert result.benchmark_name == "Benchmark A"
        assert result.grade in ["A+", "A", "B+", "B", "C", "D", "F"]
        assert isinstance(result.volatility_ratio, float)

    def test_custom_name(self):
        prices = [100, 110, 105, 115, 120]
        result = compare_with_benchmark(prices, prices, "My Benchmark")
        assert result.benchmark_name == "My Benchmark"
        assert result.sharpe_diff == 0.0

    def test_default_name(self):
        prices = [100, 110, 105, 115, 120]
        result = compare_with_benchmark(prices, prices)
        assert result.benchmark_name == "Benchmark"


class TestCompareWithMultiple:
    def test_multiple(self):
        np.random.seed(42)
        user = [1000]
        for _ in range(100):
            user.append(user[-1] * (1 + np.random.normal(0.0005, 0.01)))
        bench1 = [1000]
        bench2 = [1000]
        for _ in range(100):
            bench1.append(bench1[-1] * (1 + np.random.normal(0.0003, 0.008)))
            bench2.append(bench2[-1] * (1 + np.random.normal(0.0001, 0.015)))

        results = compare_with_multiple(user, {"Benchmark A": bench1, "Benchmark B": bench2})
        assert len(results) == 2
        assert results[0].benchmark_name == "Benchmark A"
        assert results[1].benchmark_name == "Benchmark B"


class TestGradePortfolio:
    def test_same_metrics(self):
        np.random.seed(42)
        prices = [1000]
        for _ in range(251):
            prices.append(prices[-1] * (1 + np.random.normal(0.0003, 0.01)))
        m = calculate_all_metrics(prices)
        grade = grade_portfolio(m, m)
        assert grade in ["A+", "A", "B+", "B", "C", "D", "F"]

    def test_deterministic(self):
        np.random.seed(42)
        prices = [1000]
        for _ in range(251):
            prices.append(prices[-1] * (1 + np.random.normal(0.0003, 0.01)))
        m = calculate_all_metrics(prices)
        g1 = grade_portfolio(m, m)
        g2 = grade_portfolio(m, m)
        assert g1 == g2  # same input, same result
