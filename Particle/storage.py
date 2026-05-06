"""
particle/storage.py
===================
All parquet read/write operations and metadata management.

Core guarantees:
    - NEVER overwrites existing data — always reads, merges, deduplicates
    - Handles pre-existing parquet files with any column schema (datetime or timestamp)
    - Rebuilds metadata from parquet file contents if metadata is missing or stale
    - Metadata is always derived from actual file contents — never assumed

Metadata schema (metadata.json):
    {
        "SBIN": {
            "start"        : "2023-05-04",   first candle date in file
            "end"          : "2026-05-04",   last candle date in file
            "candles"      : 456123,         total candle rows
            "trading_days" : 730,            unique trading days
            "last_updated" : "2026-05-04",   last time this entry was written
            "source"       : "zerodha",      data origin
            "timeframe"    : "1min"          candle interval
        }
    }
"""

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from . import config as cfg

log = logging.getLogger(__name__)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

MARKET_OPEN  = pd.Timestamp(cfg.MARKET_OPEN).time()
MARKET_CLOSE = pd.Timestamp(cfg.MARKET_CLOSE).time()

# Columns the analysis engines need — authoritative schema
REQUIRED_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


# ── SCHEMA NORMALISATION ──────────────────────────────────────────────────────

def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise any parquet to the authoritative schema.

    Handles:
        - Old Zerodha notebook files with 'datetime' column
        - Mixed schema files (partial timestamp / datetime)
        - Timezone-aware timestamps (strips tz)
        - Ensures _date helper column is present

    Never drops data. Only renames and type-coerces.
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Rename datetime → timestamp if needed
    if "timestamp" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
        log.debug("Renamed 'datetime' → 'timestamp'")

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")

    # Strip timezone if present
    if hasattr(df["timestamp"].dtype, "tz") and df["timestamp"].dtype.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    # Drop rows where timestamp could not be parsed
    bad = df["timestamp"].isna().sum()
    if bad:
        log.warning(f"Dropping {bad} rows with unparseable timestamps")
        df = df.dropna(subset=["timestamp"])

    # Ensure numeric OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add _date helper
    df["_date"] = df["timestamp"].dt.date

    return df


def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only candles within official NSE trading hours."""
    if df.empty:
        return df
    t = df["timestamp"].dt.time
    return df[(t >= MARKET_OPEN) & (t <= MARKET_CLOSE)].copy()


# ── METADATA ──────────────────────────────────────────────────────────────────

def load_metadata() -> dict:
    """Load full metadata.json. Returns empty dict if file missing."""
    if cfg.META_FILE.exists():
        try:
            with open(cfg.META_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"metadata.json unreadable ({e}) — starting fresh")
    return {}


def save_metadata(meta: dict) -> None:
    """Write metadata.json atomically."""
    tmp = cfg.META_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(meta, f, indent=4)
    tmp.replace(cfg.META_FILE)


def get_symbol_meta(symbol: str) -> dict:
    """Return metadata for one symbol. Empty dict if not found."""
    return load_metadata().get(symbol.upper(), {})


def rebuild_meta_from_file(symbol: str) -> dict:
    """
    Read actual parquet file and rebuild metadata from its contents.

    Called when:
        - metadata.json has no entry for this symbol
        - Parquet file exists (possibly from old notebook or manual copy)

    Guarantees metadata always reflects reality on disk.
    """
    symbol = symbol.upper()
    path   = data_path(symbol)

    if not path.exists():
        return {}

    log.info(f"[{symbol}] No metadata found — rebuilding from parquet file")
    try:
        df = pd.read_parquet(path)
        df = _normalise(df)
        df = _filter_market_hours(df)
    except Exception as e:
        log.error(f"[{symbol}] Cannot read parquet for metadata rebuild: {e}")
        return {}

    if df.empty:
        return {}

    entry = _build_meta_entry(df)
    meta  = load_metadata()
    meta[symbol] = entry
    save_metadata(meta)
    log.info(f"[{symbol}] Metadata rebuilt: {entry['start']} → {entry['end']}  "
             f"({entry['candles']:,} candles  {entry['trading_days']} days)")
    return entry


def _build_meta_entry(df: pd.DataFrame) -> dict:
    """Build a complete metadata entry from a normalised dataframe."""
    return {
        "start"        : str(df["timestamp"].min().date()),
        "end"          : str(df["timestamp"].max().date()),
        "candles"      : int(len(df)),
        "trading_days" : int(df["_date"].nunique()),
        "last_updated" : str(date.today()),
        "source"       : "zerodha",
        "timeframe"    : "1min",
    }


def update_symbol_meta(symbol: str, df: pd.DataFrame) -> dict:
    """
    Update metadata for a symbol from the full merged dataframe.
    Always derived from actual file contents — never assumed.
    Returns the written entry.
    """
    symbol = symbol.upper()
    meta   = load_metadata()
    entry  = _build_meta_entry(df)
    meta[symbol] = entry
    save_metadata(meta)
    log.debug(f"[{symbol}] Metadata → {entry['start']} to {entry['end']}  "
              f"{entry['candles']:,} candles  {entry['trading_days']} days")
    return entry


# ── DATA FILE ─────────────────────────────────────────────────────────────────

def data_path(symbol: str) -> Path:
    return cfg.DATA_DIR / f"{symbol.upper()}.parquet"


def exists(symbol: str) -> bool:
    return data_path(symbol).exists()


def load(symbol: str) -> Optional[pd.DataFrame]:
    """
    Load and normalise parquet for a symbol.
    Returns None if file does not exist.
    Always returns data with 'timestamp' column regardless of source schema.
    """
    path = data_path(symbol)
    if not path.exists():
        log.warning(f"[{symbol}] No data file at {path}")
        return None

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        log.error(f"[{symbol}] Cannot read parquet: {e}")
        return None

    df = _normalise(df)
    log.info(f"[{symbol}] Loaded {len(df):,} candles  "
             f"{df['_date'].min()} → {df['_date'].max()}")
    return df


def load_range(
    symbol : str,
    start  : Optional[date] = None,
    end    : Optional[date] = None,
) -> Optional[pd.DataFrame]:
    """Load parquet filtered to a date range. Useful for analysis engines."""
    df = load(symbol)
    if df is None:
        return None
    if start:
        df = df[df["_date"] >= start]
    if end:
        df = df[df["_date"] <= end]
    return df.reset_index(drop=True)


def save(symbol: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Safely merge new candles into existing parquet.

    Steps:
        1. Normalise new data (rename columns, parse timestamps)
        2. Filter to market hours
        3. Read existing parquet if it exists — normalise it too
        4. Concatenate existing + new
        5. Deduplicate on timestamp (exact match)
        6. Sort ascending by timestamp
        7. Write back to parquet
        8. Update metadata from full merged result

    NEVER loses existing data. Safe to call repeatedly with overlapping ranges.
    Returns the full merged dataframe.
    """
    symbol = symbol.upper()
    path   = data_path(symbol)

    # ── Normalise incoming data
    new_df = _normalise(new_df)
    new_df = _filter_market_hours(new_df)

    if new_df.empty:
        log.debug(f"[{symbol}] No candles after market hours filter — skip")
        if path.exists():
            return load(symbol)
        return new_df

    # ── Read and normalise existing data
    if path.exists():
        try:
            existing = pd.read_parquet(path)
            existing = _normalise(existing)
            existing = _filter_market_hours(existing)
            log.debug(f"[{symbol}] Existing: {len(existing):,} candles")
        except Exception as e:
            log.error(f"[{symbol}] Cannot read existing parquet ({e}) — "
                      f"proceeding with new data only. Original file preserved.")
            # Do NOT overwrite if we can't read the existing file
            raise

        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()

    # ── Dedup + sort
    before = len(combined)
    combined = (combined
                .drop_duplicates(subset=["timestamp"])
                .sort_values("timestamp")
                .reset_index(drop=True))
    dupes = before - len(combined)
    if dupes:
        log.debug(f"[{symbol}] Removed {dupes:,} duplicate candles")

    # ── Keep only columns analysis engines need + oi if present
    keep = [c for c in ["timestamp","open","high","low","close","volume","oi","_date"]
            if c in combined.columns]
    combined = combined[keep]

    # ── Write
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path, index=False)

    # ── Update metadata
    update_symbol_meta(symbol, combined)

    new_rows = len(new_df)
    total    = len(combined)
    log.info(f"[{symbol}] Saved  +{new_rows:,} new  →  {total:,} total  "
             f"({combined['_date'].min()} to {combined['_date'].max()})")

    return combined


# ── SMART GAP DETECTION ───────────────────────────────────────────────────────

def what_needs_downloading(symbol: str) -> dict:
    """
    Determine exactly which date ranges are missing for a symbol.

    Logic:
        1. Try metadata first (fastest)
        2. If no metadata but parquet exists — rebuild metadata from file
        3. Compute forward gap (last date → today)
        4. Compute backward gap (max_years_back → first date)

    Returns:
        {
            "forward":  (start_date, end_date) or None,
            "backward": (start_date, end_date) or None,
        }

    Both can be None if data is already complete.
    """
    today      = date.today()
    limit_back = date(today.year - cfg.MAX_YEARS, today.month, today.day)
    result     = {"forward": None, "backward": None}

    # ── Get metadata — rebuild from file if missing
    meta = get_symbol_meta(symbol)

    if not meta and exists(symbol):
        # Parquet exists but no metadata — rebuild from file contents
        meta = rebuild_meta_from_file(symbol)

    if not meta:
        # Truly nothing downloaded — need everything
        log.info(f"[{symbol}] No existing data — will download {limit_back} → {today}")
        result["forward"] = (limit_back, today)
        return result

    existing_start = datetime.strptime(meta["start"], "%Y-%m-%d").date()
    existing_end   = datetime.strptime(meta["end"],   "%Y-%m-%d").date()

    log.info(f"[{symbol}] Existing data: {existing_start} → {existing_end}  "
             f"({meta.get('candles',0):,} candles)")

    # ── Forward gap
    if existing_end < today:
        fwd_start         = existing_end + timedelta(days=1)
        result["forward"] = (fwd_start, today)
        log.info(f"[{symbol}] Forward gap: {fwd_start} → {today}")
    else:
        log.info(f"[{symbol}] Forward: up to date")

    # ── Backward gap
    if existing_start > limit_back:
        bwd_end            = existing_start - timedelta(days=1)
        result["backward"] = (limit_back, bwd_end)
        log.info(f"[{symbol}] Backward gap: {limit_back} → {bwd_end}")
    else:
        log.info(f"[{symbol}] Backward: complete to limit")

    return result


# ── VALIDATION ────────────────────────────────────────────────────────────────

def validate(symbol: str) -> dict:
    """Integrity check on stored parquet. Returns summary dict."""
    df = load(symbol)
    if df is None:
        return {"ok": False, "symbol": symbol.upper(), "error": "No data file"}

    issues = []

    # Required columns
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        issues.append(f"Missing columns: {missing}")

    # Nulls
    check_cols = [c for c in REQUIRED_COLS if c in df.columns]
    nulls = df[check_cols].isnull().sum()
    if nulls.any():
        issues.append(f"Nulls: {nulls[nulls > 0].to_dict()}")

    # OHLC sanity
    if {"high","low","open"}.issubset(df.columns):
        bad_hl = (df["high"] < df["low"]).sum()
        if bad_hl:
            issues.append(f"{bad_hl} candles: high < low")
        bad_o = ((df["open"] > df["high"]) | (df["open"] < df["low"])).sum()
        if bad_o:
            issues.append(f"{bad_o} candles: open outside high/low range")

    # Duplicates
    dupes = df["timestamp"].duplicated().sum()
    if dupes:
        issues.append(f"{dupes} duplicate timestamps")

    return {
        "ok"           : len(issues) == 0,
        "symbol"       : symbol.upper(),
        "rows"         : len(df),
        "trading_days" : int(df["_date"].nunique()) if "_date" in df.columns else None,
        "start"        : str(df["timestamp"].min().date()),
        "end"          : str(df["timestamp"].max().date()),
        "issues"       : issues,
    }


# ── STATUS SUMMARY ────────────────────────────────────────────────────────────

def summary() -> pd.DataFrame:
    """Return dataframe summarising all downloaded symbols from metadata."""
    meta = load_metadata()
    if not meta:
        return pd.DataFrame()

    rows = []
    for sym, m in sorted(meta.items()):
        rows.append({
            "symbol"       : sym,
            "start"        : m.get("start", ""),
            "end"          : m.get("end", ""),
            "candles"      : m.get("candles", ""),
            "trading_days" : m.get("trading_days", ""),
            "last_updated" : m.get("last_updated", ""),
        })
    return pd.DataFrame(rows)
