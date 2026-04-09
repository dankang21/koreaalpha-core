"""Tests for efficient frontier calculation."""

import numpy as np
import pytest

from koreaalpha_core.portfolio.frontier import (
    portfolio_stats,
    calculate_efficient_frontier,
    FrontierResult,
)


@pytest.fixture
def two_asset_data():
    """Two-asset test data with known properties."""
    mean_returns = np.array([0.0005, 0.0003])  # daily
    cov_matrix = np.array([
        [0.0004, 0.0001],
        [0.0001, 0.0002],
    ])
    return mean_returns, cov_matrix


class TestPortfolioStats:
    def test_basic(self, two_asset_data):
        mean_ret, cov = two_asset_data
        weights = np.array([0.6, 0.4])
        ret, vol, sharpe = portfolio_stats(weights, mean_ret, cov)
        assert ret > 0
        assert vol > 0
        assert isinstance(sharpe, float)

    def test_equal_weights(self, two_asset_data):
        mean_ret, cov = two_asset_data
        weights = np.array([0.5, 0.5])
        ret, vol, sharpe = portfolio_stats(weights, mean_ret, cov)
        assert 0 < ret < 1
        assert 0 < vol < 1

    def test_single_asset(self, two_asset_data):
        mean_ret, cov = two_asset_data
        weights = np.array([1.0, 0.0])
        ret, vol, _ = portfolio_stats(weights, mean_ret, cov)
        expected_ret = mean_ret[0] * 248
        assert abs(ret - expected_ret) < 1e-6


class TestEfficientFrontier:
    def test_basic_result(self, two_asset_data):
        mean_ret, cov = two_asset_data
        weights = np.array([0.6, 0.4])
        result = calculate_efficient_frontier(mean_ret, cov, weights, n_points=20)

        assert isinstance(result, FrontierResult)
        assert len(result.frontier_points) > 0
        assert result.min_variance.volatility <= result.max_sharpe.volatility + 0.01
        assert result.max_sharpe.sharpe >= result.min_variance.sharpe - 0.01

    def test_individual_assets(self, two_asset_data):
        mean_ret, cov = two_asset_data
        weights = np.array([0.5, 0.5])
        result = calculate_efficient_frontier(mean_ret, cov, weights, n_points=10)
        assert len(result.individual_assets) == 2

    def test_reproducibility(self, two_asset_data):
        mean_ret, cov = two_asset_data
        weights = np.array([0.5, 0.5])
        r1 = calculate_efficient_frontier(mean_ret, cov, weights, seed=42)
        r2 = calculate_efficient_frontier(mean_ret, cov, weights, seed=42)
        assert r1.max_sharpe.sharpe == r2.max_sharpe.sharpe

    def test_three_assets(self):
        mean_ret = np.array([0.0006, 0.0003, 0.0001])
        cov = np.array([
            [0.0004, 0.0001, 0.00005],
            [0.0001, 0.0002, 0.00008],
            [0.00005, 0.00008, 0.0001],
        ])
        weights = np.array([0.4, 0.3, 0.3])
        result = calculate_efficient_frontier(mean_ret, cov, weights, n_points=15)
        assert len(result.individual_assets) == 3
        assert result.n_simulations >= 3000
