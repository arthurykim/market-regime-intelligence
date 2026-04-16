from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DATA_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

# Tickers
SPY_TICKER = "SPY"
VIX_TICKER = "^VIX"

# Feature parameters
ROLLING_VOL_WINDOW = 20
ROLLING_DRAWDOWN_WINDOW = 60
ZSCORE_WINDOW = 20
ANNUALIZATION_FACTOR = 252

# Regime thresholds (rule-based)
CRISIS_VIX_THRESHOLD = 30.0
CRISIS_DRAWDOWN_THRESHOLD = -0.10
ELEVATED_VIX_THRESHOLD = 20.0
ELEVATED_DRAWDOWN_THRESHOLD = -0.05

# Data defaults
DEFAULT_START_DATE = "2006-01-01"
