"""FastAPI application entry point."""

from fastapi import FastAPI

from app.api.routes import router
from app.services.summary_service import market_service
from app.utils.logging_utils import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Market Regime Intelligence",
    description="Detects and classifies US equity market volatility regimes using SPY and VIX data.",
    version="1.0.0",
)

app.include_router(router)


@app.on_event("startup")
def startup_event():
    logger.info("Loading market data and computing features...")
    market_service.refresh()
    logger.info("Data loaded. API ready.")
