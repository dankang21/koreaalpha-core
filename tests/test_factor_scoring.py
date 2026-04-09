"""Tests for factor scoring."""

import numpy as np
import pytest

from koreaalpha_core.factor.scoring import (
    calculate_momentum,
    calculate_value_score,
    calculate_quality_score,
    calculate_growth_score,
    calculate_factor_scores,
    FactorScores,
)


class TestMomentum:
    def test_uptrend(self):
        prices = np.linspace(100, 150, 250)  # steady uptrend
        score = calculate_momentum(prices)
        assert score > 0

    def test_downtrend(self):
        prices = np.linspace(150, 100, 250)  # steady downtrend
        score = calculate_momentum(prices)
        assert score < 0

    def test_short_data(self):
        prices = np.array([100, 101, 102])
        score = calculate_momentum(prices)
        assert score == 0.0


class TestValueScore:
    def test_low_per_pbr(self):
        score = calculate_value_score(per=5.0, pbr=0.5)
        assert score > 0

    def test_high_per_pbr(self):
        score = calculate_value_score(per=50.0, pbr=5.0)
        low_score = score
        high_score = calculate_value_score(per=5.0, pbr=0.5)
        assert high_score > low_score

    def test_none_values(self):
        assert calculate_value_score() == 0.0
        assert calculate_value_score(per=None, pbr=None) == 0.0


class TestQualityScore:
    def test_high_roe_low_vol(self):
        score = calculate_quality_score(roe=0.25, volatility=0.10)
        assert score > 0

    def test_low_roe_high_vol(self):
        score = calculate_quality_score(roe=0.02, volatility=0.50)
        assert score < calculate_quality_score(roe=0.25, volatility=0.10)


class TestGrowthScore:
    def test_high_growth(self):
        score = calculate_growth_score(revenue_growth=0.30, earnings_growth=0.40)
        assert score > 0

    def test_no_growth(self):
        assert calculate_growth_score() == 0.0


class TestFactorScores:
    def test_composite(self):
        prices = np.linspace(100, 130, 250)
        scores = calculate_factor_scores(
            prices, per=10.0, pbr=1.5, roe=0.15,
            revenue_growth=0.10, earnings_growth=0.15,
        )
        assert isinstance(scores, FactorScores)
        assert scores.composite != 0

    def test_weighted_composite(self):
        prices = np.linspace(100, 130, 250)
        scores = calculate_factor_scores(prices, per=10.0, pbr=1.5, roe=0.15)
        c1 = scores.weighted_composite(w_momentum=1, w_value=0, w_quality=0, w_growth=0)
        c2 = scores.weighted_composite(w_momentum=0, w_value=1, w_quality=0, w_growth=0)
        assert c1 != c2  # different weights → different composites
