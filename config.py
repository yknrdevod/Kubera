"""
particle/config.py
==================
Single source of truth for all configuration.

To update enctoken daily:
    1. Open Kite in browser
    2. Open DevTools → Network → any API call → copy Authorization header value
    3. Paste below (after "enctoken ")
"""

import os
from pathlib import Path

# ── CREDENTIALS ───────────────────────────────────────────────────────────────
# Your Zerodha enctoken. Expires daily. Update each morning.
ENCTOKEN = "rlZ4Ft="

# ── PATHS ─────────────────────────────────────────────────────────────────────
# Root directory for all downloaded data and outputs.
# Change this to your preferred location.
BASE_DIR     = Path(r"D:\Stocks")
DATA_DIR     = BASE_DIR / "data"        # parquet files per symbol
REPORTS_DIR  = BASE_DIR / "reports"    # CSV outputs from analysis engines
LOGS_DIR     = BASE_DIR / "logs"       # run logs

META_FILE       = BASE_DIR / "metadata.json"
INSTRUMENT_FILE = BASE_DIR / "instruments.parquet"

# ── DOWNLOAD SETTINGS ─────────────────────────────────────────────────────────
TIMEFRAME   = "minute"    # candle interval sent to Zerodha API
CHUNK_DAYS  = 60          # days per API request (Zerodha max: 60 for minute)
MAX_YEARS   = 3           # how far back to backfill (Zerodha stores ~3 years of 1-min)
API_SLEEP   = 2.0         # seconds between API calls (rate limit safety)
MAX_RETRIES = 3           # retry attempts on API failure

# ── MARKET HOURS (IST) ────────────────────────────────────────────────────────
MARKET_OPEN  = "09:15"
MARKET_CLOSE = "15:30"

# ── API ENDPOINTS ─────────────────────────────────────────────────────────────
INSTRUMENTS_URL = "https://api.kite.trade/instruments"
HISTORICAL_URL  = "https://kite.zerodha.com/oms/instruments/historical/{token}/{timeframe}"

# ── HEADERS ───────────────────────────────────────────────────────────────────
def get_headers() -> dict:
    """Return auth headers. Always reads current ENCTOKEN."""
    return {"Authorization": f"enctoken {ENCTOKEN}"}

# ── ANALYSIS ENGINE DEFAULTS ──────────────────────────────────────────────────
# These are read by analysis modules. Change here, applies everywhere.
SWING = {
    "bucket_pct"       : 0.005,
    "lambda"           : 0.05,
    "forward_window"   : 1950,
    "outcome_threshold": 0.01,
    "min_candles"      : 30 * 390,
    "ma_short"         : 20 * 390,
    "ma_long"          : 50 * 390,
}

OVERNIGHT = {
    "bucket_pct"            : 0.002,
    "lambda"                : 0.35,
    "forward_window"        : 500,
    "outcome_threshold"     : 0.004,
    "min_days"              : 15,
    "absorption_percentile"  : 80,
    "absorption_multiplier"  : 1.50,
}

CORRECTION = {
    "min_depth"   : 3.0,
    "peak_window" : 390,
}

# ── ENSURE DIRECTORIES EXIST ──────────────────────────────────────────────────
def init_dirs():
    """Create all required directories if they don't exist."""
    for d in [BASE_DIR, DATA_DIR, REPORTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
