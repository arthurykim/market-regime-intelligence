"""Tests for feature engineering functions."""

import numpy as np
import pandas as pd
import pytest

from app.services.feature_engineering import (
    build_feature_table,
    compute_log_returns,
    compute_return_zscore,
    compute_rolling_drawdown,
    compute_rolling_volatility,
    compute_vix_rv_spread,
)


@pytest.fixture
def sample_prices() -> pd.Series:
    """Simple ascending price series."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    prices = pd.Series(np.linspace(100, 120, 100), index=dates)
    return prices


@pytest.fixture
def sample_combined() -> pd.DataFrame:
    """Combined SPY + VIX dataframe for integration test."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    return pd.DataFrame(
        {
            "spy_close": np.linspace(100, 120, 100),
            "vix_close": np.linspace(15, 25, 100),
        },
        index=dates,
    )


class TestLogReturns:
    def test_first_value_is_nan(self, sample_prices):
        returns = compute_log_returns(sample_prices)
        assert pd.isna(returns.iloc[0])

    def test_positive_returns_for_ascending_prices(self, sample_prices):
        returns = compute_log_returns(sample_prices)
        assert (returns.iloc[1:] > 0).all()

    def test_length_matches_input(self, sample_prices):
        returns = compute_log_returns(sample_prices)
        assert len(returns) == len(sample_prices)

    def test_known_value(self):
        prices = pd.Series([100.0, 110.0])
        returns = compute_log_returns(prices)
        expected = np.log(110 / 100)
        assert abs(returns.iloc[1] - expected) < 1e-10


class TestRollingVolatility:
    def test_output_is_non_negative(self, sample_prices):
        returns = compute_log_returns(sample_prices)
        vol = compute_rolling_volatility(returns, window=10)
        valid = vol.dropna()
        assert (valid >= 0).all()

    def test_leading_nans(self, sample_prices):
        returns = compute_log_returns(sample_prices)
        vol = compute_rolling_volatility(returns, window=20)
        # First 20 values should be NaN (19 from window + 1 from returns)
        assert pd.isna(vol.iloc[:20]).all()

    def test_annualization(self, sample_prices):
        returns = compute_log_returns(sample_prices)
        vol_annualized = compute_rolling_volatility(returns, window=10)
        vol_raw = returns.rolling(window=10).std()
        ratio = (vol_annualized / vol_raw).dropna()
        expected_ratio = np.sqrt(252)
        assert np.allclose(ratio.values, expected_ratio, atol=0.01)


class TestRollingDrawdown:
    def test_ascending_prices_zero_drawdown(self, sample_prices):
        dd = compute_rolling_drawdown(sample_prices, window=60)
        # Strictly ascending prices should have ~0 drawdown
        assert (dd >= -1e-10).all()

    def test_drawdown_after_drop(self):
        prices = pd.Series([100, 110, 120, 100, 90])
        dd = compute_rolling_drawdown(prices, window=5)
        # Last value: 90 vs max of 120 = -25%
        assert abs(dd.iloc[-1] - (-0.25)) < 1e-10

    def test_output_range(self, sample_prices):
        dd = compute_rolling_drawdown(sample_prices)
        assert (dd <= 0).all()


class TestReturnZscore:
    def test_output_length(self, sample_prices):
        returns = compute_log_returns(sample_prices)
        zscore = compute_return_zscore(returns, window=10)
        assert len(zscore) == len(returns)

    def test_extreme_value_detected(self):
        returns = pd.Series([0.01] * 30 + [0.10])
        zscore = compute_return_zscore(returns, window=20)
        # The spike at the end should have a high z-score
        assert zscore.iloc[-1] > 2.0


class TestVixRvSpread:
    def test_positive_spread_when_vix_exceeds_rv(self):
        vix = pd.Series([25.0, 30.0])
        rv = pd.Series([0.15, 0.20])
        spread = compute_vix_rv_spread(vix, rv)
        assert (spread > 0).all()


class TestBuildFeatureTable:
    def test_output_columns(self, sample_combined):
        features = build_feature_table(sample_combined)
        expected_cols = {
            "spy_close", "vix_close", "log_return",
            "realized_vol_20d", "drawdown_60d",
            "return_zscore_20d", "vix_rv_spread",
        }
        assert expected_cols == set(features.columns)

    def test_no_nans_in_output(self, sample_combined):
        features = build_feature_table(sample_combined)
        assert not features.isna().any().any()

    def test_fewer_rows_than_input(self, sample_combined):
        features = build_feature_table(sample_combined)
        # Rows are dropped due to rolling windows
        assert len(features) < len(sample_combined)
