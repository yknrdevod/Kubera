"""
particle/engines/swing.py
=========================
Swing pressure engine — 1-min or daily OHLCV.

Layers 1-5: bucket, decay, outcome, distance, freshness  (via base)
Layer 6: trend context (MA)
Layer 7: gap risk
"""

import logging
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config as cfg
from .base import (
    Bucket, build_buckets, calc_pressure, compute_gap_risk,
    compute_outcomes, compute_trend, interpret_score,
    levels_to_rows, trend_alignment, prepare_df,
)

log = logging.getLogger(__name__)

PROFILES = {
    "1min": dict(
        label="1-Minute", unit="1-min candles",
        forward_window=cfg.SWING["forward_window"],
        outcome_threshold=cfg.SWING["outcome_threshold"],
        lam=cfg.SWING["lambda"], bucket_pct=cfg.SWING["bucket_pct"],
        min_candles=cfg.SWING["min_candles"], days_from="timestamp",
        ma_short=cfg.SWING["ma_short"], ma_long=cfg.SWING["ma_long"],
    ),
    "daily": dict(
        label="Daily", unit="daily candles",
        forward_window=5, outcome_threshold=cfg.SWING["outcome_threshold"],
        lam=cfg.SWING["lambda"], bucket_pct=cfg.SWING["bucket_pct"],
        min_candles=60, days_from="index", ma_short=20, ma_long=50,
    ),
}


def _to_daily(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("_date")
              .agg(timestamp=("timestamp","first"), open=("open","first"),
                   high=("high","max"), low=("low","min"),
                   close=("close","last"), volume=("volume","sum"))
              .reset_index(drop=True)
              .assign(_date=lambda d: pd.to_datetime(d["timestamp"]).dt.date))


def run(
    df        : pd.DataFrame,
    ticker    : str  = "UNKNOWN",
    timeframe : str  = "1min",
    outdir    : Path = None,
    silent    : bool = False,
) -> dict:
    """
    Run swing pressure analysis.

    Returns result dict with all values — suitable for agentic pipelines.
    Saves summary + levels CSV to outdir.
    """
    ticker    = ticker.upper()
    tf        = timeframe.lower()
    p         = PROFILES.get(tf, PROFILES["1min"])
    outdir    = Path(outdir) if outdir else cfg.REPORTS_DIR
    df        = prepare_df(df)

    if tf == "daily" and df["timestamp"].dt.hour.nunique() > 1:
        log.info(f"[{ticker}] Aggregating 1-min → daily")
        df = _to_daily(df)

    if len(df) < p["min_candles"]:
        msg = f"Need >= {p['min_candles']} {p['unit']}. Got {len(df)}."
        log.warning(f"[{ticker}] {msg}")
        return {"ticker": ticker, "status": "error", "message": msg}

    log.info(f"[{ticker}] Swing ({p['label']})  {len(df):,} candles")

    closes   = df["close"].values
    outcomes = compute_outcomes(closes, p["forward_window"], p["outcome_threshold"])
    buckets  = build_buckets(df, outcomes, p["bucket_pct"], p["lam"], p["days_from"])
    current  = float(closes[-1])
    pressure = calc_pressure(buckets, current)
    trend    = compute_trend(df, p["ma_short"], p["ma_long"])
    gap      = compute_gap_risk(df)

    score    = pressure["score"]
    sig, meaning, action = interpret_score(score)
    alignment = trend_alignment(score, trend["direction"])

    sb = pressure["below_buckets"]
    rb = pressure["above_buckets"]
    stop   = round(sb[0].zone_price * 0.98, 4) if sb else None
    target = round(rb[0].zone_price, 4)         if rb else None
    rr     = round(rb[0].distance / sb[0].distance, 2) if sb and rb and sb[0].distance > 0 else None

    result = dict(
        ticker=ticker, timeframe=p["label"], system="Swing",
        status="ok", run_date=str(date.today()),
        current_price=round(current,4),
        data_start=str(df["_date"].min()), data_end=str(df["_date"].max()),
        candles=len(df),
        pressure_score=score, signal=sig, meaning=meaning, action=action,
        above_pressure=pressure["above_pressure"],
        below_pressure=pressure["below_pressure"],
        trend_direction=trend["direction"],
        ma_short=trend["ma_short"], ma_long=trend["ma_long"],
        trend_alignment=alignment,
        gap_risk_score=gap.get("gap_risk_score"),
        gap_risk_label=gap.get("risk_label"),
        gap_avg_pct=gap.get("avg_gap_pct"),
        gap_std_pct=gap.get("std_gap_pct"),
        gap_positive_pct=gap.get("positive_pct"),
        gap_negative_pct=gap.get("negative_pct"),
        gap_large_pct=gap.get("large_gap_pct"),
        stop_loss=stop, target=target, risk_reward=rr,
        above_buckets=pressure["above_buckets"],
        below_buckets=pressure["below_buckets"],
    )

    if not silent:
        _print(result, gap)

    _save_csv(result, gap, outdir)
    return result


def _print(r: dict, gap: dict):
    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  [{r['ticker']}]  SWING  ({r['timeframe']})")
    print(sep)
    print(f"  Price          : {r['current_price']:,.4f}")
    print(f"  Pressure Score : {r['pressure_score']:+.4f}")
    print(f"  Signal         : {r['signal']}")
    print(f"  Action         : {r['action']}")
    print(sep)
    print(f"  Trend          : {r['trend_direction']}")
    print(f"  Alignment      : {r['trend_alignment']}")
    if gap.get("available"):
        print(f"  Gap Risk       : {gap['risk_label']}  score={gap['gap_risk_score']:.3f}  "
              f"avg={gap['avg_gap_pct']:+.3f}%  large={gap['large_gap_pct']:.1f}%")
    print(sep)
    hdr = f"  {'Zone':>10}  {'Outcome':>8}  {'Visits':>7}  {'Fresh':>7}  {'Dist%':>6}  {'State':<10}  {'Force':>10}"
    print(f"\n  RESISTANCE (above):"); print(hdr)
    for b in r["above_buckets"][:5]:
        print(f"  {b.zone_price:>10,.4f}  {b.avg_outcome:>+8.3f}  {b.visit_count:>7}  "
              f"{b.freshness:>7.3f}  {b.distance*100:>5.2f}%  {b.strength_label:<10}  {b.pressure_contribution:>+10.2f}")
    print(f"\n  SUPPORT (below):"); print(hdr)
    for b in r["below_buckets"][:5]:
        print(f"  {b.zone_price:>10,.4f}  {b.avg_outcome:>+8.3f}  {b.visit_count:>7}  "
              f"{b.freshness:>7.3f}  {b.distance*100:>5.2f}%  {b.strength_label:<10}  {b.pressure_contribution:>+10.2f}")
    if r["stop_loss"] and r["target"]:
        print(f"\n  Stop: {r['stop_loss']:,.4f}  Target: {r['target']:,.4f}  R:R 1:{r['risk_reward']:.1f}")
    print(f"{sep}\n")


def _save_csv(r: dict, gap: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    sym = r["ticker"]; dt = r["run_date"].replace("-","")
    meta = {k: r.get(k,"") for k in [
        "run_date","ticker","timeframe","system","current_price",
        "pressure_score","signal","meaning","action",
        "above_pressure","below_pressure","trend_direction",
        "ma_short","ma_long","trend_alignment","stop_loss","target","risk_reward"]}
    meta.update({f"gap_{k}": gap.get(k,"") for k in
                 ["risk_score","risk_label","avg_gap_pct","std_gap_pct",
                  "positive_pct","negative_pct","large_gap_pct"]})
    sp = outdir / f"{sym}_swing_summary_{dt}.csv"
    pd.DataFrame([meta]).to_csv(sp, index=False)
    level_meta = {"run_date": r["run_date"], "ticker": sym,
                  "timeframe": r["timeframe"], "system": "Swing"}
    rows = levels_to_rows(r["above_buckets"], r["below_buckets"],
                          r["current_price"], r["pressure_score"], level_meta)
    lp = outdir / f"{sym}_swing_levels_{dt}.csv"
    pd.DataFrame(rows).to_csv(lp, index=False)
    log.info(f"[{sym}] Swing CSVs → {outdir}")
