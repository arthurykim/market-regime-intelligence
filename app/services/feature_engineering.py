"""Compute risk and volatility features from price data."""

import numpy as np
import pandas as pd

from app.config import (
    ANNUALIZATION_FACTOR,
    ROLLING_DRAWDOWN_WINDOW,
    ROLLING_VOL_WINDOW,
    ZSCORE_WINDOW,
)


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Daily log returns."""
    return np.log(prices / prices.shift(1))


def compute_rolling_volatility(
    returns: pd.Series,
    window: int = ROLLING_VOL_WINDOW,
) -> pd.Series:
    """Annualized rolling standard deviation of returns."""
    return returns.rolling(window=window).std() * np.sqrt(ANNUALIZATION_FACTOR)


def compute_rolling_drawdown(
    prices: pd.Series,
    window: int = ROLLING_DRAWDOWN_WINDOW,
) -> pd.Series:
    """Rolling max drawdown over a trailing window.

    Returns a non-positive Series: 0 means at rolling high, -0.10 means 10% below.
    """
    rolling_max = prices.rolling(window=window, min_periods=1).max()
    drawdown = (prices - rolling_max) / rolling_max
    return drawdown


def compute_return_zscore(
    returns: pd.Series,
    window: int = ZSCORE_WINDOW,
) -> pd.Series:
    """Z-score of daily return relative to a rolling window."""
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    return (returns - rolling_mean) / rolling_std


def compute_vix_rv_spread(
    vix: pd.Series,
    realized_vol: pd.Series,
) -> pd.Series:
    """Spread between implied volatility (VIX) and realized volatility.

    A large positive spread suggests the market is pricing in more fear
    than recent price action warrants.
    """
    return vix - (realized_vol * 100)


def build_feature_table(combined: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature table from combined SPY + VIX data.

    Expects columns: spy_close, vix_close
    """
    features = pd.DataFrame(index=combined.index)
    features.index.name = "date"

    features["spy_close"] = combined["spy_close"]
    features["vix_close"] = combined["vix_close"]

    features["log_return"] = compute_log_returns(combined["spy_close"])
    features["realized_vol_20d"] = compute_rolling_volatility(features["log_return"])
    features["drawdown_60d"] = compute_rolling_drawdown(combined["spy_close"])
    features["return_zscore_20d"] = compute_return_zscore(features["log_return"])
    features["vix_rv_spread"] = compute_vix_rv_spread(
        combined["vix_close"],
        features["realized_vol_20d"],
    )

    features = features.dropna()
    return features
