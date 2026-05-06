"""
particle/instruments.py
=======================
NSE instrument list management.

Fetches from Zerodha API once per day.
Caches as parquet. Returns instrument_token for any symbol.

No candle data here. Only symbol → token mapping.
"""

import logging
from datetime import datetime
from io import StringIO

import pandas as pd
import requests

from . import config as cfg

log = logging.getLogger(__name__)


def load() -> pd.DataFrame:
    """
    Load NSE instruments.
    Uses cached file if downloaded today. Re-downloads otherwise.

    Returns DataFrame with columns: tradingsymbol, instrument_token
    """
    path  = cfg.INSTRUMENT_FILE
    today = datetime.today().date()

    # Use cache if fresh
    if path.exists():
        modified = datetime.fromtimestamp(path.stat().st_mtime).date()
        if modified == today:
            log.debug("Using cached instruments (downloaded today)")
            return pd.read_parquet(path)
        log.info("Instrument cache outdated. Re-downloading ...")
    else:
        log.info("No instrument cache found. Downloading ...")

    return _download_and_cache()


def _download_and_cache() -> pd.DataFrame:
    """Download instruments from Zerodha API and save to parquet."""
    resp = requests.get(cfg.INSTRUMENTS_URL, headers=cfg.get_headers(), timeout=30)

    if resp.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch instruments. HTTP {resp.status_code}: {resp.text[:200]}"
        )

    df = pd.read_csv(StringIO(resp.text))

    # Keep only NSE equity
    df = df[
        (df["exchange"] == "NSE") &
        (df["segment"]  == "NSE")
    ][["tradingsymbol", "instrument_token"]].drop_duplicates().reset_index(drop=True)

    df.to_parquet(cfg.INSTRUMENT_FILE, index=False)
    log.info(f"Downloaded {len(df):,} NSE instruments → {cfg.INSTRUMENT_FILE}")
    return df


def get_token(symbol: str) -> int:
    """
    Return instrument_token for a symbol.
    Raises ValueError if symbol not found.
    """
    symbol = symbol.upper().strip()
    df     = load()
    row    = df[df["tradingsymbol"] == symbol]

    if row.empty:
        raise ValueError(
            f"Symbol '{symbol}' not found in NSE instruments. "
            f"Check spelling or run instruments.load() to refresh."
        )

    return int(row["instrument_token"].iloc[0])


def search(query: str) -> pd.DataFrame:
    """
    Search instruments by partial symbol name.
    Returns matching rows sorted alphabetically.
    """
    df = load()
    mask = df["tradingsymbol"].str.contains(query.upper(), na=False)
    return df[mask].sort_values("tradingsymbol").reset_index(drop=True)


def exists(symbol: str) -> bool:
    """Check whether a symbol exists in NSE instruments."""
    try:
        get_token(symbol)
        return True
    except ValueError:
        return False
