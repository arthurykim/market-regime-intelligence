"""Orchestration layer: loads data, computes features, classifies regimes."""

from datetime import date
from typing import Optional

import pandas as pd

from app.services.data_loader import load_combined
from app.services.feature_engineering import build_feature_table
from app.services.regime_classifier import classify_regimes


class MarketDataService:
    """Singleton-style service that holds the computed feature + regime data."""

    def __init__(self) -> None:
        self._features: Optional[pd.DataFrame] = None
        self._regimes: Optional[pd.Series] = None

    def refresh(self, force_download: bool = False) -> None:
        """Reload data, recompute features and regimes."""
        combined = load_combined(force_refresh=force_download)
        self._features = build_feature_table(combined)
        self._regimes = classify_regimes(self._features)

    @property
    def features(self) -> pd.DataFrame:
        if self._features is None:
            self.refresh()
        return self._features  # type: ignore[return-value]

    @property
    def regimes(self) -> pd.Series:
        if self._regimes is None:
            self.refresh()
        return self._regimes  # type: ignore[return-value]

    def get_current_regime(self) -> dict:
        """Most recent regime label and associated features."""
        latest = self.features.iloc[-1]
        return {
            "date": str(self.features.index[-1].date()),
            "regime": self.regimes.iloc[-1],
            "vix_close": round(float(latest["vix_close"]), 2),
            "realized_vol_20d": round(float(latest["realized_vol_20d"]), 4),
            "drawdown_60d": round(float(latest["drawdown_60d"]), 4),
        }

    def get_regime_history(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> list[dict]:
        """Regime labels over a date range."""
        regimes = self.regimes.copy()
        features = self.features.copy()

        if start:
            regimes = regimes[regimes.index >= start]
            features = features[features.index >= start]
        if end:
            regimes = regimes[regimes.index <= end]
            features = features[features.index <= end]

        results = []
        for dt in regimes.index:
            results.append({
                "date": str(dt.date()),
                "regime": regimes[dt],
                "vix_close": round(float(features.loc[dt, "vix_close"]), 2),
            })
        return results

    def get_latest_features(self) -> dict:
        """Most recent row of the feature table."""
        latest = self.features.iloc[-1]
        return {
            "date": str(self.features.index[-1].date()),
            "spy_close": round(float(latest["spy_close"]), 2),
            "vix_close": round(float(latest["vix_close"]), 2),
            "log_return": round(float(latest["log_return"]), 6),
            "realized_vol_20d": round(float(latest["realized_vol_20d"]), 4),
            "drawdown_60d": round(float(latest["drawdown_60d"]), 4),
            "return_zscore_20d": round(float(latest["return_zscore_20d"]), 4),
            "vix_rv_spread": round(float(latest["vix_rv_spread"]), 2),
        }

    def get_summary(self) -> dict:
        """High-level summary of the dataset and regime distribution."""
        total_days = len(self.regimes)
        regime_counts = self.regimes.value_counts().to_dict()
        regime_pct = {k: round(v / total_days * 100, 1) for k, v in regime_counts.items()}

        return {
            "data_start": str(self.features.index[0].date()),
            "data_end": str(self.features.index[-1].date()),
            "total_trading_days": total_days,
            "current_regime": self.regimes.iloc[-1],
            "regime_distribution": regime_pct,
            "latest_vix": round(float(self.features.iloc[-1]["vix_close"]), 2),
            "latest_realized_vol": round(float(self.features.iloc[-1]["realized_vol_20d"]), 4),
        }


# Module-level instance used by the API
market_service = MarketDataService()
