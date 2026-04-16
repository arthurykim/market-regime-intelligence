"""Classify market regimes from feature data."""

from enum import Enum

import pandas as pd

from app.config import (
    CRISIS_DRAWDOWN_THRESHOLD,
    CRISIS_VIX_THRESHOLD,
    ELEVATED_DRAWDOWN_THRESHOLD,
    ELEVATED_VIX_THRESHOLD,
)


class Regime(str, Enum):
    CALM = "calm"
    ELEVATED_RISK = "elevated_risk"
    CRISIS = "crisis"


def classify_regime_row(vix: float, drawdown: float) -> str:
    """Classify a single observation into a regime.

    Priority: crisis > elevated_risk > calm.
    """
    if vix >= CRISIS_VIX_THRESHOLD or drawdown <= CRISIS_DRAWDOWN_THRESHOLD:
        return Regime.CRISIS.value
    if vix >= ELEVATED_VIX_THRESHOLD or drawdown <= ELEVATED_DRAWDOWN_THRESHOLD:
        return Regime.ELEVATED_RISK.value
    return Regime.CALM.value


def classify_regimes(features: pd.DataFrame) -> pd.Series:
    """Apply rule-based regime classification across the full feature table."""
    regimes = features.apply(
        lambda row: classify_regime_row(row["vix_close"], row["drawdown_60d"]),
        axis=1,
    )
    regimes.name = "regime"
    return regimes
