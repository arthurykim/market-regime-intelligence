"""FastAPI route definitions."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import (
    CurrentRegimeResponse,
    HealthResponse,
    LatestFeaturesResponse,
    RegimeHistoryEntry,
    SummaryResponse,
)
from app.services.summary_service import market_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", message="Market Regime Intelligence API is running")


@router.get("/regime/current", response_model=CurrentRegimeResponse)
def get_current_regime():
    try:
        data = market_service.get_current_regime()
        return CurrentRegimeResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regime/history", response_model=list[RegimeHistoryEntry])
def get_regime_history(
    start: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
):
    try:
        data = market_service.get_regime_history(start=start, end=end)
        return [RegimeHistoryEntry(**entry) for entry in data]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/latest", response_model=LatestFeaturesResponse)
def get_latest_features():
    try:
        data = market_service.get_latest_features()
        return LatestFeaturesResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", response_model=SummaryResponse)
def get_summary():
    try:
        data = market_service.get_summary()
        return SummaryResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
