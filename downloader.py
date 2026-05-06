"""
particle/downloader.py
======================
Zerodha historical data fetcher.

Core principle:
    Download ONLY what is missing. Never re-download what exists.

Smart update logic (via storage.what_needs_downloading):
    Forward gap  → fetch from last_date+1 to today
    Backward gap → fetch from limit_back to first_date-1
    Both gaps    → fetch both in one run
    No gap       → already up to date, skip

All data is saved incrementally — safe to interrupt and resume.
"""

import logging
import time
from datetime import date, timedelta
from typing import Iterator, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config as cfg
from . import instruments, storage

log = logging.getLogger(__name__)


# ── HTTP SESSION ──────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    """Create a session with retry logic baked in."""
    session = requests.Session()
    retry   = Retry(
        total           = cfg.MAX_RETRIES,
        backoff_factor  = 0.5,
        status_forcelist= [429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session

_session = _make_session()


# ── CHUNK GENERATOR ───────────────────────────────────────────────────────────

def _chunks(
    start: date,
    end: date,
    chunk_days: int = cfg.CHUNK_DAYS,
) -> Iterator[Tuple[date, date]]:
    """
    Split a date range into chunks of chunk_days.
    Yields (chunk_start, chunk_end) pairs in chronological order.
    """
    current = start
    while current <= end:
        chunk_end   = min(current + timedelta(days=chunk_days - 1), end)
        yield current, chunk_end
        current     = chunk_end + timedelta(days=1)


# ── API FETCH ─────────────────────────────────────────────────────────────────

def _fetch_chunk(token: int, start: date, end: date) -> Optional[pd.DataFrame]:
    """
    Fetch one chunk from Zerodha historical API.

    Returns DataFrame with columns:
        timestamp, open, high, low, close, volume, oi

    Returns None on failure (logged, not raised — caller continues).
    """
    url    = cfg.HISTORICAL_URL.format(token=token, timeframe=cfg.TIMEFRAME)
    params = {
        "oi"  : 1,
        "from": start.strftime("%Y-%m-%d"),
        "to"  : end.strftime("%Y-%m-%d"),
    }

    try:
        resp = _session.get(url, headers=cfg.get_headers(), params=params, timeout=30)
    except requests.RequestException as e:
        log.error(f"Request failed: {e}")
        return None

    if resp.status_code != 200:
        log.error(f"API error {resp.status_code}: {resp.text[:200]}")
        return None

    candles = resp.json().get("data", {}).get("candles", [])

    if not candles:
        log.debug(f"No candles returned for {start} → {end}")
        return None

    df = pd.DataFrame(
        candles,
        columns=["timestamp", "open", "high", "low", "close", "volume", "oi"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ── CORE DOWNLOAD ─────────────────────────────────────────────────────────────

def _download_range(
    symbol : str,
    token  : int,
    start  : date,
    end    : date,
    label  : str = "",
) -> int:
    """
    Download all chunks in [start, end] and save incrementally.
    Returns total candles fetched.
    """
    total      = 0
    chunk_list = list(_chunks(start, end))
    n          = len(chunk_list)

    log.info(f"[{symbol}] {label}  {start} → {end}  ({n} chunks)")

    for i, (cs, ce) in enumerate(chunk_list, 1):
        log.info(f"[{symbol}] Chunk {i}/{n}: {cs} → {ce}")

        df = _fetch_chunk(token, cs, ce)

        if df is None or df.empty:
            log.warning(f"[{symbol}] Empty chunk {cs} → {ce} — skipping")
            time.sleep(cfg.API_SLEEP)
            continue

        storage.save(symbol, df)
        total += len(df)

        log.info(f"[{symbol}] +{len(df):,} candles  (total so far: {total:,})")
        time.sleep(cfg.API_SLEEP)

    return total


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def update(symbol: str) -> dict:
    """
    Smart incremental update for one symbol.

    1. Checks what date ranges are missing (forward + backward gaps).
    2. Downloads only those ranges.
    3. Saves incrementally — safe to interrupt.
    4. Returns a result dict for CLI display and agentic consumption.

    This is the primary function for daily use.
    """
    symbol = symbol.upper()
    result = {
        "symbol"          : symbol,
        "status"          : "ok",
        "forward_fetched" : 0,
        "backward_fetched": 0,
        "total_fetched"   : 0,
        "message"         : "",
    }

    # What do we need?
    gaps = storage.what_needs_downloading(symbol)
    forward  = gaps["forward"]
    backward = gaps["backward"]

    if not forward and not backward:
        result["message"] = "Already up to date. Nothing to download."
        log.info(f"[{symbol}] Already up to date.")
        return result

    # Get instrument token
    try:
        token = instruments.get_token(symbol)
    except ValueError as e:
        result["status"]  = "error"
        result["message"] = str(e)
        log.error(f"[{symbol}] {e}")
        return result

    # Forward gap → update to today
    if forward:
        start, end = forward
        n = _download_range(symbol, token, start, end, label="Forward update")
        result["forward_fetched"] = n

    # Backward gap → extend history
    if backward:
        start, end = backward
        n = _download_range(symbol, token, start, end, label="Backward backfill")
        result["backward_fetched"] = n

    result["total_fetched"] = result["forward_fetched"] + result["backward_fetched"]

    # Final metadata summary
    meta = storage.get_symbol_meta(symbol)
    result["message"] = (
        f"Done. Range: {meta.get('start')} → {meta.get('end')}  "
        f"Rows: {meta.get('rows', 0):,}"
    )
    log.info(f"[{symbol}] {result['message']}")
    return result


def update_many(symbols: list) -> list:
    """
    Update multiple symbols in sequence.
    Returns list of result dicts (one per symbol).

    Used for batch mode — run on a watchlist.
    """
    results = []
    for i, sym in enumerate(symbols, 1):
        log.info(f"\n{'='*50}")
        log.info(f"Symbol {i}/{len(symbols)}: {sym}")
        log.info(f"{'='*50}")
        results.append(update(sym))
    return results


def backfill(symbol: str, years: int = cfg.MAX_YEARS) -> dict:
    """
    Force a full backfill regardless of what exists.
    Use this to rebuild data from scratch for one symbol.
    Existing data is preserved — new data is merged.
    """
    symbol = symbol.upper()
    today  = date.today()
    start  = date(today.year - years, today.month, today.day)

    try:
        token = instruments.get_token(symbol)
    except ValueError as e:
        return {"symbol": symbol, "status": "error", "message": str(e)}

    n = _download_range(symbol, token, start, today, label="Full backfill")

    meta = storage.get_symbol_meta(symbol)
    return {
        "symbol"       : symbol,
        "status"       : "ok",
        "total_fetched": n,
        "message"      : f"Backfill complete. {meta.get('start')} → {meta.get('end')}  Rows: {meta.get('rows',0):,}",
    }


def fetch_custom(symbol: str, start: date, end: date) -> dict:
    """
    Download a specific date range for one symbol.
    Useful for targeted gap filling.
    """
    symbol = symbol.upper()

    try:
        token = instruments.get_token(symbol)
    except ValueError as e:
        return {"symbol": symbol, "status": "error", "message": str(e)}

    n = _download_range(symbol, token, start, end, label="Custom range")

    return {
        "symbol"       : symbol,
        "status"       : "ok",
        "total_fetched": n,
        "message"      : f"Custom fetch complete. {start} → {end}  Fetched: {n:,} candles",
    }
