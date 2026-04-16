"""Pydantic response models for the API."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    message: str


class CurrentRegimeResponse(BaseModel):
    date: str
    regime: str
    vix_close: float
    realized_vol_20d: float
    drawdown_60d: float


class RegimeHistoryEntry(BaseModel):
    date: str
    regime: str
    vix_close: float


class LatestFeaturesResponse(BaseModel):
    date: str
    spy_close: float
    vix_close: float
    log_return: float
    realized_vol_20d: float
    drawdown_60d: float
    return_zscore_20d: float
    vix_rv_spread: float


class RegimeDistribution(BaseModel):
    calm: float = 0.0
    elevated_risk: float = 0.0
    crisis: float = 0.0


class SummaryResponse(BaseModel):
    data_start: str
    data_end: str
    total_trading_days: int
    current_regime: str
    regime_distribution: dict[str, float]
    latest_vix: float
    latest_realized_vol: float
