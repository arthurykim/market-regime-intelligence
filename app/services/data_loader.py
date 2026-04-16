"""Data ingestion: download and cache SPY + VIX daily data via yfinance."""

import pandas as pd
import yfinance as yf

from app.config import (
    DATA_DIR,
    DEFAULT_START_DATE,
    SPY_TICKER,
    VIX_TICKER,
)


def download_ticker(ticker: str, start: str = DEFAULT_START_DATE) -> pd.DataFrame:
    """Download daily OHLCV data for a single ticker."""
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    # yfinance sometimes returns MultiIndex columns; flatten if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "date"
    return df


def load_spy(start: str = DEFAULT_START_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """Load SPY data from cache or download fresh."""
    path = DATA_DIR / "spy_daily.parquet"
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)
    df = download_ticker(SPY_TICKER, start=start)
    df.to_parquet(path)
    return df


def load_vix(start: str = DEFAULT_START_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """Load VIX data from cache or download fresh."""
    path = DATA_DIR / "vix_daily.parquet"
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)
    df = download_ticker(VIX_TICKER, start=start)
    df.to_parquet(path)
    return df


def load_combined(start: str = DEFAULT_START_DATE, force_refresh: bool = False) -> pd.DataFrame:
    """Load SPY close + VIX close aligned on the same date index."""
    spy = load_spy(start=start, force_refresh=force_refresh)
    vix = load_vix(start=start, force_refresh=force_refresh)

    combined = pd.DataFrame({
        "spy_close": spy["Close"],
        "vix_close": vix["Close"],
    })
    combined.index.name = "date"
    combined = combined.dropna()
    return combined
