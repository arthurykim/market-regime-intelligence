"""Generate and save regime visualizations."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from app.config import OUTPUTS_DIR


REGIME_COLORS = {
    "calm": "#2ecc71",
    "elevated_risk": "#f39c12",
    "crisis": "#e74c3c",
}


def plot_regime_timeline(
    features: pd.DataFrame,
    regimes: pd.Series,
    filename: str = "regime_timeline.png",
) -> str:
    """Plot SPY price with regime-colored background shading."""
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(features.index, features["spy_close"], color="black", linewidth=0.8, label="SPY Close")

    # Shade background by regime
    prev_regime = regimes.iloc[0]
    start_idx = regimes.index[0]

    for i in range(1, len(regimes)):
        current_regime = regimes.iloc[i]
        if current_regime != prev_regime or i == len(regimes) - 1:
            end_idx = regimes.index[i]
            ax.axvspan(start_idx, end_idx, alpha=0.25, color=REGIME_COLORS[prev_regime])
            start_idx = end_idx
            prev_regime = current_regime

    # Legend patches
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=c, alpha=0.4, label=r) for r, c in REGIME_COLORS.items()]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=9)

    ax.set_title("SPY Price with Market Regime Classification", fontsize=13)
    ax.set_ylabel("Price ($)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = OUTPUTS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def plot_feature_dashboard(
    features: pd.DataFrame,
    regimes: pd.Series,
    filename: str = "feature_dashboard.png",
) -> str:
    """Multi-panel chart: VIX, realized vol, drawdown, regime labels."""
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Panel 1: VIX
    axes[0].plot(features.index, features["vix_close"], color="#8e44ad", linewidth=0.8)
    axes[0].axhline(y=20, color="orange", linestyle="--", linewidth=0.7, alpha=0.7, label="VIX=20")
    axes[0].axhline(y=30, color="red", linestyle="--", linewidth=0.7, alpha=0.7, label="VIX=30")
    axes[0].set_ylabel("VIX")
    axes[0].set_title("Feature Dashboard", fontsize=13)
    axes[0].legend(fontsize=8)

    # Panel 2: Realized Volatility
    axes[1].plot(features.index, features["realized_vol_20d"], color="#2980b9", linewidth=0.8)
    axes[1].set_ylabel("Realized Vol (20d)")

    # Panel 3: Drawdown
    axes[2].fill_between(features.index, features["drawdown_60d"], 0, color="#e74c3c", alpha=0.4)
    axes[2].axhline(y=-0.05, color="orange", linestyle="--", linewidth=0.7, alpha=0.7)
    axes[2].axhline(y=-0.10, color="red", linestyle="--", linewidth=0.7, alpha=0.7)
    axes[2].set_ylabel("Drawdown (60d)")

    # Panel 4: Regime labels as colored scatter
    regime_num = regimes.map({"calm": 0, "elevated_risk": 1, "crisis": 2})
    colors = regimes.map(REGIME_COLORS)
    axes[3].scatter(features.index, regime_num, c=colors, s=1, alpha=0.6)
    axes[3].set_yticks([0, 1, 2])
    axes[3].set_yticklabels(["calm", "elevated_risk", "crisis"])
    axes[3].set_ylabel("Regime")

    axes[3].xaxis.set_major_locator(mdates.YearLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    path = OUTPUTS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)
