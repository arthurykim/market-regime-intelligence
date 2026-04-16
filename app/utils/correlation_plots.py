"""Visualizations for cross-asset correlation analysis."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from app.config import OUTPUTS_DIR


REGIME_COLORS = {
    "calm": "#2ecc71",
    "elevated_risk": "#f39c12",
    "crisis": "#e74c3c",
}


def plot_correlation_heatmaps(
    correlations: dict[str, pd.DataFrame],
    filename: str = "correlation_heatmaps.png",
) -> str:
    """Plot side-by-side correlation heatmaps for each regime."""
    regimes_to_plot = [r for r in ["calm", "elevated_risk", "crisis"] if r in correlations]
    n = len(regimes_to_plot)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, regime in zip(axes, regimes_to_plot):
        corr = correlations[regime]
        im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(corr.index, fontsize=9)

        # Annotate cells
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                val = corr.iloc[i, j]
                color = "white" if abs(val) > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

        title_color = REGIME_COLORS.get(regime, "black")
        ax.set_title(regime.replace("_", " ").title(), fontsize=12,
                      fontweight="bold", color=title_color)

    fig.suptitle("Cross-Asset Correlation by Market Regime", fontsize=14, y=1.02)
    fig.colorbar(im, ax=axes, shrink=0.8, label="Correlation")
    plt.tight_layout()

    path = OUTPUTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_rolling_correlations(
    rolling_corrs: pd.DataFrame,
    regimes: pd.Series,
    filename: str = "rolling_correlations.png",
) -> str:
    """Plot rolling correlations of each asset vs SPY with regime shading."""
    assets = rolling_corrs.columns.tolist()
    n = len(assets)

    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = ["#2980b9", "#8e44ad", "#e67e22", "#27ae60", "#e74c3c"]

    for idx, (ax, asset) in enumerate(zip(axes, assets)):
        # Regime background shading
        aligned_regimes = regimes.loc[regimes.index.isin(rolling_corrs.index)]
        prev_regime = aligned_regimes.iloc[0]
        start_idx = aligned_regimes.index[0]

        for i in range(1, len(aligned_regimes)):
            current_regime = aligned_regimes.iloc[i]
            if current_regime != prev_regime or i == len(aligned_regimes) - 1:
                end_idx = aligned_regimes.index[i]
                ax.axvspan(start_idx, end_idx, alpha=0.15,
                          color=REGIME_COLORS[prev_regime])
                start_idx = end_idx
                prev_regime = current_regime

        ax.plot(rolling_corrs.index, rolling_corrs[asset],
                color=colors[idx % len(colors)], linewidth=0.8)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)
        ax.set_ylabel(f"{asset}\nvs SPY", fontsize=9)
        ax.set_ylim(-1, 1)

    axes[0].set_title("60-Day Rolling Correlation vs SPY", fontsize=13)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = OUTPUTS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_regime_return_comparison(
    regime_returns: pd.DataFrame,
    filename: str = "regime_return_comparison.png",
) -> str:
    """Bar chart comparing annualized returns by asset and regime."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(regime_returns.index))
    width = 0.25

    for i, regime in enumerate(["calm", "elevated_risk", "crisis"]):
        if regime in regime_returns.columns:
            values = regime_returns[regime].values * 100
            bars = ax.bar(x + i * width, values, width,
                         label=regime.replace("_", " ").title(),
                         color=REGIME_COLORS[regime], alpha=0.8)

    ax.set_xlabel("Asset")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("Annualized Return by Asset and Market Regime", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels(regime_returns.index, rotation=45, ha="right")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend()
    plt.tight_layout()

    path = OUTPUTS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
