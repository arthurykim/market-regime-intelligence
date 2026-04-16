"""Validate regime classifier against known historical market events.

Tests whether the classifier correctly identifies crisis and elevated_risk
periods during well-documented market stress events.
"""

import pandas as pd

from app.services.data_loader import load_combined
from app.services.feature_engineering import build_feature_table
from app.services.regime_classifier import classify_regimes


# Known market stress events with expected regime during peak stress
KNOWN_EVENTS = [
    {
        "name": "2008 Financial Crisis (Lehman Collapse)",
        "peak_start": "2008-09-15",
        "peak_end": "2008-11-30",
        "expected_regime": "crisis",
        "description": "Lehman Brothers bankruptcy triggered global financial meltdown. VIX hit 80+.",
    },
    {
        "name": "2010 Flash Crash",
        "peak_start": "2010-05-06",
        "peak_end": "2010-05-20",
        "expected_regime": "elevated_risk",
        "description": "Dow dropped ~1000 points intraday. Brief but severe.",
    },
    {
        "name": "2011 US Debt Downgrade",
        "peak_start": "2011-08-05",
        "peak_end": "2011-10-15",
        "expected_regime": "crisis",
        "description": "S&P downgraded US debt from AAA. VIX spiked above 40.",
    },
    {
        "name": "2015 China Devaluation / Vol Spike",
        "peak_start": "2015-08-20",
        "peak_end": "2015-09-30",
        "expected_regime": "crisis",
        "description": "China devalued yuan, global markets sold off. VIX hit 40.",
    },
    {
        "name": "2018 Q4 Selloff",
        "peak_start": "2018-12-01",
        "peak_end": "2018-12-31",
        "expected_regime": "elevated_risk",
        "description": "Fed tightening + trade war fears. SPY fell ~20% from highs.",
    },
    {
        "name": "COVID-19 Crash",
        "peak_start": "2020-03-01",
        "peak_end": "2020-04-15",
        "expected_regime": "crisis",
        "description": "Pandemic lockdowns triggered fastest bear market in history. VIX hit 82.",
    },
    {
        "name": "2022 Bear Market (Fed Tightening)",
        "peak_start": "2022-06-01",
        "peak_end": "2022-06-30",
        "expected_regime": "elevated_risk",
        "description": "Aggressive rate hikes. SPY dropped >20% from January highs.",
    },
    {
        "name": "2023 SVB / Banking Crisis",
        "peak_start": "2023-03-10",
        "peak_end": "2023-03-31",
        "expected_regime": "elevated_risk",
        "description": "Silicon Valley Bank collapsed. Regional banking fears spread.",
    },
    {
        "name": "2025 Tariff Shock",
        "peak_start": "2025-04-03",
        "peak_end": "2025-04-15",
        "expected_regime": "crisis",
        "description": "Broad tariff announcement triggered VIX spike to 52.",
    },
    {
        "name": "2013 Bull Market (Control - Calm Period)",
        "peak_start": "2013-06-01",
        "peak_end": "2013-08-31",
        "expected_regime": "calm",
        "description": "Low volatility bull run. VIX mostly below 15. Should NOT be flagged.",
    },
    {
        "name": "2017 Low Vol Environment (Control - Calm Period)",
        "peak_start": "2017-06-01",
        "peak_end": "2017-08-31",
        "expected_regime": "calm",
        "description": "Historically low VIX. Extended calm regime.",
    },
]


def validate_event(
    features: pd.DataFrame,
    regimes: pd.Series,
    event: dict,
) -> dict:
    """Check if the classifier labels an event period with the expected regime."""
    mask = (regimes.index >= event["peak_start"]) & (regimes.index <= event["peak_end"])
    period_regimes = regimes[mask]

    if len(period_regimes) == 0:
        return {
            "event": event["name"],
            "status": "NO_DATA",
            "detail": f"No data found for {event['peak_start']} to {event['peak_end']}",
        }

    regime_counts = period_regimes.value_counts()
    dominant_regime = regime_counts.idxmax()
    expected = event["expected_regime"]

    # For crisis events: check if crisis appeared at all (not just dominant)
    # For calm events: check if calm is dominant
    if expected == "crisis":
        hit = "crisis" in regime_counts.index
    elif expected == "elevated_risk":
        hit = ("elevated_risk" in regime_counts.index) or ("crisis" in regime_counts.index)
    else:
        hit = dominant_regime == expected

    return {
        "event": event["name"],
        "period": f"{event['peak_start']} to {event['peak_end']}",
        "expected": expected,
        "dominant_regime": dominant_regime,
        "regime_breakdown": regime_counts.to_dict(),
        "trading_days": len(period_regimes),
        "status": "PASS" if hit else "FAIL",
        "description": event["description"],
    }


def run_event_validation(features: pd.DataFrame, regimes: pd.Series) -> list[dict]:
    """Run validation across all known events."""
    results = []
    for event in KNOWN_EVENTS:
        result = validate_event(features, regimes, event)
        results.append(result)
    return results


def format_validation_report(results: list[dict]) -> str:
    """Format validation results as a human-readable report."""
    lines = []
    lines.append("=" * 80)
    lines.append("REGIME CLASSIFIER — EVENT VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append("")

    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    lines.append(f"Results: {passed}/{total} events correctly identified")
    lines.append("")

    for r in results:
        status_marker = "PASS" if r["status"] == "PASS" else "FAIL"
        lines.append(f"[{status_marker}] {r['event']}")
        lines.append(f"  Period: {r.get('period', 'N/A')}")
        lines.append(f"  Expected: {r['expected']}")
        lines.append(f"  Dominant regime: {r.get('dominant_regime', 'N/A')}")

        if "regime_breakdown" in r:
            breakdown = ", ".join(
                f"{regime}: {count}d" for regime, count in r["regime_breakdown"].items()
            )
            lines.append(f"  Breakdown: {breakdown}")

        lines.append(f"  Context: {r['description']}")
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)


if __name__ == "__main__":
    print("Loading data and computing features...")
    combined = load_combined()
    features = build_feature_table(combined)
    regimes = classify_regimes(features)

    print("Running event validation...\n")
    results = run_event_validation(features, regimes)
    report = format_validation_report(results)
    print(report)
