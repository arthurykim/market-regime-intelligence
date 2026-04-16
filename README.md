# Market Regime and Volatility Intelligence System

A Python-based system that detects and classifies US equity market volatility regimes using SPY and VIX data. Computes risk features from historical price data, applies rule-based regime classification, and exposes results through a FastAPI service.

This is a **descriptive analysis tool**, not a trading bot or price prediction system.

## What It Does

- Downloads 20 years of daily SPY and VIX data via yfinance
- Computes risk features: rolling realized volatility, drawdown depth, VIX-realized vol spread, return z-scores
- Classifies each trading day into one of three regimes: **calm**, **elevated_risk**, or **crisis**
- Generates regime timeline and feature dashboard visualizations
- Serves regime data and features through a REST API

## Regime Classification

Regimes are assigned using a rule-based classifier with the following logic (priority order):

| Regime | Condition |
|--------|-----------|
| **crisis** | VIX >= 30 OR 60-day drawdown <= -10% |
| **elevated_risk** | VIX >= 20 OR 60-day drawdown <= -5% |
| **calm** | Everything else |

Over the full dataset (2006-2026), the distribution is approximately:
- **calm**: 62.5% of trading days
- **elevated_risk**: 25.5%
- **crisis**: 12.0%

## Features Computed

| Feature | Description |
|---------|-------------|
| `log_return` | Daily log return of SPY |
| `realized_vol_20d` | 20-day rolling annualized volatility |
| `drawdown_60d` | 60-day rolling maximum drawdown |
| `return_zscore_20d` | Z-score of daily return over 20-day window |
| `vix_rv_spread` | VIX minus annualized realized vol (implied vs. realized gap) |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /regime/current` | Current regime with supporting features |
| `GET /regime/history?start=YYYY-MM-DD&end=YYYY-MM-DD` | Regime labels over a date range |
| `GET /features/latest` | Most recent row of the feature table |
| `GET /summary` | Dataset overview and regime distribution |

## Quick Start

```bash
# Clone and set up
git clone <repo-url>
cd market-regime-intelligence
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Generate plots (downloads data on first run)
python -c "
from app.services.data_loader import load_combined
from app.services.feature_engineering import build_feature_table
from app.services.regime_classifier import classify_regimes
from app.utils.plotting import plot_regime_timeline, plot_feature_dashboard

combined = load_combined(force_refresh=True)
features = build_feature_table(combined)
regimes = classify_regimes(features)
plot_regime_timeline(features, regimes)
plot_feature_dashboard(features, regimes)
print('Plots saved to outputs/')
"

# Start the API
uvicorn app.main:app --reload

# Test it
curl http://localhost:8000/summary
```

## Project Structure

```
market-regime-intelligence/
  app/
    main.py                        # FastAPI entry point
    config.py                      # Parameters, thresholds, paths
    api/routes.py                  # Endpoint definitions
    services/
      data_loader.py               # yfinance ingestion + parquet caching
      feature_engineering.py       # Risk feature computation
      regime_classifier.py         # Rule-based regime classification
      summary_service.py           # Orchestration layer for API
    models/schemas.py              # Pydantic response models
    utils/
      plotting.py                  # Visualization generation
      logging_utils.py             # Logger config
  tests/
    test_feature_engineering.py    # 16 tests for feature functions
    test_regime_classifier.py      # 15 tests for classification logic
  data/                            # Cached parquet files
  outputs/                         # Generated charts
```

## Tests

31 tests covering:
- Log return correctness and edge cases
- Rolling volatility annualization
- Drawdown computation (ascending prices, drops)
- Z-score spike detection
- VIX-realized vol spread
- Feature table integration (columns, NaN handling, row count)
- Regime boundary conditions (VIX=20, VIX=30, drawdown=-5%, drawdown=-10%)
- Priority logic (crisis overrides elevated_risk)
- Output shape and label correctness

## Tech Stack

Python, FastAPI, pandas, NumPy, matplotlib, scikit-learn, yfinance, pytest

## Limitations

- Regime thresholds are fixed and based on common market conventions, not optimized from data
- VIX is used as a proxy for implied volatility; it reflects S&P 500 options, not individual stocks
- Data depends on yfinance availability and may have gaps on early dates
- This is descriptive analysis — regimes are labeled after the fact, not predicted in advance
