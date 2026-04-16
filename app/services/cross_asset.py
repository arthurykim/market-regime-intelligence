"""Cross-asset data loading and correlation analysis.

Downloads and aligns daily data for a basket of ETFs representing
major asset classes, then computes correlation metrics conditioned
on market regime.
"""

import pandas as pd
import numpy as np
import yfinance as yf

from app.config import DATA_DIR, DEFAULT_START_DATE


ASSET_UNIVERSE = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100 (Tech)",
    "XLF": "Financials",
    "XLE": "Energy",
    "GLD": "Gold",
    "TLT": "Long-Term Treasuries",
}


def load_asset_prices(
    start: str = DEFAULT_START_DATE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download and cache daily close prices for the asset universe."""
    path = DATA_DIR / "cross_asset_prices.parquet"
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    tickers = list(ASSET_UNIVERSE.keys())
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns: (Price, Ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]

    prices.index.name = "date"
    prices = prices.dropna()
    prices.to_parquet(path)
    return prices


def compute_asset_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns for all assets."""
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def compute_correlation_by_regime(
    returns: pd.DataFrame,
    regimes: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Compute correlation matrix for each regime.

    Returns a dict mapping regime name to its correlation matrix.
    """
    aligned_returns = returns.loc[returns.index.isin(regimes.index)]
    aligned_regimes = regimes.loc[regimes.index.isin(returns.index)]

    correlations = {}
    for regime in ["calm", "elevated_risk", "crisis"]:
        mask = aligned_regimes == regime
        regime_returns = aligned_returns[mask]
        if len(regime_returns) > 10:
            correlations[regime] = regime_returns.corr().round(4)

    return correlations


def compute_rolling_correlation(
    returns: pd.DataFrame,
    target: str = "SPY",
    window: int = 60,
) -> pd.DataFrame:
    """Compute rolling correlation of each asset vs a target (default SPY)."""
    rolling_corrs = pd.DataFrame(index=returns.index)

    for col in returns.columns:
        if col == target:
            continue
        rolling_corrs[col] = returns[col].rolling(window=window).corr(returns[target])

    return rolling_corrs.dropna()


def compute_regime_mean_returns(
    returns: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Annualized mean return for each asset in each regime."""
    aligned_returns = returns.loc[returns.index.isin(regimes.index)]
    aligned_regimes = regimes.loc[regimes.index.isin(returns.index)]

    results = {}
    for regime in ["calm", "elevated_risk", "crisis"]:
        mask = aligned_regimes == regime
        regime_returns = aligned_returns[mask]
        results[regime] = (regime_returns.mean() * 252).round(4)

    return pd.DataFrame(results)


def format_correlation_report(
    correlations: dict[str, pd.DataFrame],
    regime_returns: pd.DataFrame,
) -> str:
    """Format cross-asset analysis as a human-readable report."""
    lines = []
    lines.append("=" * 80)
    lines.append("CROSS-ASSET CORRELATION ANALYSIS BY REGIME")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Assets: {', '.join(ASSET_UNIVERSE.values())}")
    lines.append("")

    for regime, corr_matrix in correlations.items():
        lines.append(f"--- CORRELATION MATRIX: {regime.upper()} ---\n")
        lines.append(corr_matrix.to_string())
        lines.append("")

    lines.append("\n--- ANNUALIZED MEAN RETURN BY REGIME ---\n")
    lines.append(regime_returns.to_string())

    lines.append("\n\n--- KEY THINGS TO LOOK FOR ---\n")
    lines.append("  1. SPY-TLT correlation: Does it flip negative in crisis? (flight to safety)")
    lines.append("  2. SPY-GLD correlation: Does gold decouple from equities during stress?")
    lines.append("  3. SPY-QQQ correlation: Tech and S&P move together in calm — does it")
    lines.append("     tighten further in crisis? (correlation convergence)")
    lines.append("  4. XLF during crisis: Financials often lead into stress and get hit hardest")
    lines.append("  5. XLE behavior: Energy is commodity-driven — does it follow equities or")
    lines.append("     diverge based on its own supply/demand dynamics?")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)
