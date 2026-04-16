"""Tests for regime classification logic."""

import numpy as np
import pandas as pd
import pytest

from app.services.regime_classifier import Regime, classify_regime_row, classify_regimes


class TestClassifyRegimeRow:
    def test_calm(self):
        assert classify_regime_row(vix=15.0, drawdown=-0.02) == "calm"

    def test_elevated_vix(self):
        assert classify_regime_row(vix=22.0, drawdown=-0.02) == "elevated_risk"

    def test_elevated_drawdown(self):
        assert classify_regime_row(vix=15.0, drawdown=-0.06) == "elevated_risk"

    def test_crisis_vix(self):
        assert classify_regime_row(vix=35.0, drawdown=-0.02) == "crisis"

    def test_crisis_drawdown(self):
        assert classify_regime_row(vix=15.0, drawdown=-0.12) == "crisis"

    def test_crisis_both(self):
        assert classify_regime_row(vix=40.0, drawdown=-0.20) == "crisis"

    def test_boundary_vix_20_is_elevated(self):
        assert classify_regime_row(vix=20.0, drawdown=0.0) == "elevated_risk"

    def test_boundary_vix_30_is_crisis(self):
        assert classify_regime_row(vix=30.0, drawdown=0.0) == "crisis"

    def test_boundary_drawdown_minus_5_is_elevated(self):
        assert classify_regime_row(vix=10.0, drawdown=-0.05) == "elevated_risk"

    def test_boundary_drawdown_minus_10_is_crisis(self):
        assert classify_regime_row(vix=10.0, drawdown=-0.10) == "crisis"

    def test_crisis_takes_priority(self):
        # VIX says crisis, drawdown says elevated — crisis wins
        assert classify_regime_row(vix=32.0, drawdown=-0.06) == "crisis"


class TestClassifyRegimes:
    def test_output_length_matches_input(self):
        features = pd.DataFrame({
            "vix_close": [12.0, 22.0, 35.0],
            "drawdown_60d": [-0.01, -0.03, -0.15],
        })
        regimes = classify_regimes(features)
        assert len(regimes) == 3

    def test_correct_labels(self):
        features = pd.DataFrame({
            "vix_close": [12.0, 22.0, 35.0],
            "drawdown_60d": [-0.01, -0.03, -0.15],
        })
        regimes = classify_regimes(features)
        assert list(regimes) == ["calm", "elevated_risk", "crisis"]

    def test_all_calm(self):
        features = pd.DataFrame({
            "vix_close": [10.0, 12.0, 14.0, 11.0],
            "drawdown_60d": [0.0, -0.01, -0.02, -0.03],
        })
        regimes = classify_regimes(features)
        assert (regimes == "calm").all()

    def test_regime_enum_values(self):
        assert Regime.CALM.value == "calm"
        assert Regime.ELEVATED_RISK.value == "elevated_risk"
        assert Regime.CRISIS.value == "crisis"
