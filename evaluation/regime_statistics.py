"""Compute descriptive statistics about regime behavior.

Analyzes regime durations, transition frequencies, and return
distributions conditioned on regime labels.
"""

import pandas as pd
import numpy as np


def compute_regime_durations(regimes: pd.Series) -> pd.DataFrame:
    """Compute the duration of each consecutive regime period.

    Returns a DataFrame with columns: regime, start, end, duration_days.
    """
    periods = []
    current_regime = regimes.iloc[0]
    start_date = regimes.index[0]

    for i in range(1, len(regimes)):
        if regimes.iloc[i] != current_regime:
            periods.append({
                "regime": current_regime,
                "start": start_date,
                "end": regimes.index[i - 1],
                "duration_days": (regimes.index[i - 1] - start_date).days + 1,
            })
            current_regime = regimes.iloc[i]
            start_date = regimes.index[i]

    # Final period
    periods.append({
        "regime": current_regime,
        "start": start_date,
        "end": regimes.index[-1],
        "duration_days": (regimes.index[-1] - start_date).days + 1,
    })

    return pd.DataFrame(periods)


def compute_duration_stats(durations: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics for regime durations by regime type."""
    stats = durations.groupby("regime")["duration_days"].agg(
        ["count", "mean", "median", "min", "max", "std"]
    ).round(1)
    stats.columns = ["num_periods", "mean_days", "median_days", "min_days", "max_days", "std_days"]
    return stats


def compute_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """Compute regime-to-regime transition probability matrix.

    Each cell (i, j) = P(regime tomorrow = j | regime today = i).
    """
    transitions = pd.DataFrame({
        "from": regimes.iloc[:-1].values,
        "to": regimes.iloc[1:].values,
    })
    counts = pd.crosstab(transitions["from"], transitions["to"])

    # Normalize rows to probabilities
    probs = counts.div(counts.sum(axis=1), axis=0).round(4)
    return probs


def compute_return_by_regime(
    features: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Compute return distribution statistics conditioned on regime."""
    df = features[["log_return"]].copy()
    df["regime"] = regimes

    stats = df.groupby("regime")["log_return"].agg(
        ["count", "mean", "std", "min", "max"]
    )
    stats["mean_annualized"] = stats["mean"] * 252
    stats["std_annualized"] = stats["std"] * np.sqrt(252)
    stats["sharpe_approx"] = stats["mean_annualized"] / stats["std_annualized"]
    stats = stats.round(6)
    return stats


def compute_vix_by_regime(
    features: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    """VIX summary statistics by regime."""
    df = features[["vix_close"]].copy()
    df["regime"] = regimes

    stats = df.groupby("regime")["vix_close"].agg(
        ["mean", "median", "min", "max", "std"]
    ).round(2)
    return stats


def format_statistics_report(
    duration_stats: pd.DataFrame,
    transition_matrix: pd.DataFrame,
    return_stats: pd.DataFrame,
    vix_stats: pd.DataFrame,
) -> str:
    """Format all regime statistics into a human-readable report."""
    lines = []
    lines.append("=" * 80)
    lines.append("REGIME STATISTICS REPORT")
    lines.append("=" * 80)

    lines.append("\n--- REGIME DURATION STATISTICS ---\n")
    lines.append(duration_stats.to_string())

    lines.append("\n\n--- REGIME TRANSITION PROBABILITIES ---")
    lines.append("(Row = current regime, Column = next day's regime)\n")
    lines.append(transition_matrix.to_string())

    lines.append("\n\n--- DAILY RETURN STATISTICS BY REGIME ---\n")
    lines.append(return_stats.to_string())

    lines.append("\n\n--- VIX STATISTICS BY REGIME ---\n")
    lines.append(vix_stats.to_string())

    lines.append("\n\n--- KEY OBSERVATIONS ---\n")
    lines.append("Look for:")
    lines.append("  1. How sticky are regimes? (high self-transition probability = sticky)")
    lines.append("  2. Does crisis come from elevated_risk or directly from calm?")
    lines.append("  3. Is the return difference between regimes economically meaningful?")
    lines.append("  4. How long does the average crisis last vs. calm period?")
    lines.append("")
    lines.append("=" * 80)

    return "\n".join(lines)


if __name__ == "__main__":
    from app.services.data_loader import load_combined
    from app.services.feature_engineering import build_feature_table
    from app.services.regime_classifier import classify_regimes

    print("Loading data...")
    combined = load_combined()
    features = build_feature_table(combined)
    regimes = classify_regimes(features)

    durations = compute_regime_durations(regimes)
    duration_stats = compute_duration_stats(durations)
    transition_matrix = compute_transition_matrix(regimes)
    return_stats = compute_return_by_regime(features, regimes)
    vix_stats = compute_vix_by_regime(features, regimes)

    report = format_statistics_report(duration_stats, transition_matrix, return_stats, vix_stats)
    print(report)
