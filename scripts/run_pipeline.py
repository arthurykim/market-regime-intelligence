"""Full pipeline runner: download, compute, classify, evaluate, plot, report.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --refresh    # force re-download data
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.data_loader import load_combined
from app.services.feature_engineering import build_feature_table
from app.services.regime_classifier import classify_regimes
from app.utils.plotting import plot_regime_timeline, plot_feature_dashboard
from evaluation.event_validation import run_event_validation, format_validation_report
from evaluation.regime_statistics import (
    compute_regime_durations,
    compute_duration_stats,
    compute_transition_matrix,
    compute_return_by_regime,
    compute_vix_by_regime,
    format_statistics_report,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def main():
    force_refresh = "--refresh" in sys.argv

    # Step 1: Data ingestion
    print("[1/6] Loading market data...")
    combined = load_combined(force_refresh=force_refresh)
    print(f"       {len(combined)} trading days: {combined.index[0].date()} to {combined.index[-1].date()}")

    # Step 2: Feature engineering
    print("[2/6] Computing features...")
    features = build_feature_table(combined)
    print(f"       {len(features)} rows, {len(features.columns)} features")

    # Step 3: Regime classification
    print("[3/6] Classifying regimes...")
    regimes = classify_regimes(features)
    counts = regimes.value_counts()
    for regime, count in counts.items():
        pct = count / len(regimes) * 100
        print(f"       {regime}: {count} days ({pct:.1f}%)")

    # Step 4: Event validation
    print("[4/6] Running event validation...")
    validation_results = run_event_validation(features, regimes)
    validation_report = format_validation_report(validation_results)
    passed = sum(1 for r in validation_results if r["status"] == "PASS")
    print(f"       {passed}/{len(validation_results)} events correctly identified")

    # Step 5: Regime statistics
    print("[5/6] Computing regime statistics...")
    durations = compute_regime_durations(regimes)
    duration_stats = compute_duration_stats(durations)
    transition_matrix = compute_transition_matrix(regimes)
    return_stats = compute_return_by_regime(features, regimes)
    vix_stats = compute_vix_by_regime(features, regimes)
    stats_report = format_statistics_report(
        duration_stats, transition_matrix, return_stats, vix_stats
    )

    # Step 6: Generate visualizations
    print("[6/6] Generating visualizations...")
    p1 = plot_regime_timeline(features, regimes)
    p2 = plot_feature_dashboard(features, regimes)
    print(f"       Saved: {p1}")
    print(f"       Saved: {p2}")

    # Save reports
    eval_path = RESULTS_DIR / "event_validation_report.txt"
    eval_path.write_text(validation_report)
    print(f"\n       Event validation report: {eval_path}")

    stats_path = RESULTS_DIR / "regime_statistics_report.txt"
    stats_path.write_text(stats_report)
    print(f"       Regime statistics report: {stats_path}")

    # Save feature table snapshot
    feature_path = RESULTS_DIR / "feature_table_latest.csv"
    output = features.copy()
    output["regime"] = regimes
    output.to_csv(feature_path)
    print(f"       Feature table CSV: {feature_path}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
