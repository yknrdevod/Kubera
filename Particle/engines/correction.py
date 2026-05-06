"""
particle/engines/correction.py
================================
Correction profiler and phase detector.

Part 1: Historical correction events (depth, duration, recovery, type)
Part 2: Current phase (Trending/Topping/Correcting/Bottoming/Recovering)
Part 3: Forecast based on historical profile
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from .. import config as cfg
from .base import prepare_df

log = logging.getLogger(__name__)

MIN_DEPTH   = cfg.CORRECTION["min_depth"]
PEAK_WINDOW = cfg.CORRECTION["peak_window"]   # candles each side for 1-min


@dataclass
class CorrectionEvent:
    peak_date      : object
    trough_date    : object
    peak_price     : float
    trough_price   : float
    depth_pct      : float
    duration_days  : int
    recovery_date  : Optional[object] = None
    recovery_days  : Optional[int]    = None
    correction_type: str              = "price"

    @property
    def recovered(self): return self.recovery_date is not None


def _find_peaks_troughs(closes: np.ndarray, window: int):
    n = len(closes)
    peaks = np.zeros(n, dtype=bool)
    troughs = np.zeros(n, dtype=bool)
    for i in range(window, n - window):
        seg = closes[i - window : i + window + 1]
        if closes[i] == seg.max() and closes[i] > closes[i-1]: peaks[i]   = True
        if closes[i] == seg.min() and closes[i] < closes[i-1]: troughs[i] = True
    return peaks, troughs


def _detect_events(df: pd.DataFrame, peaks: np.ndarray,
                   troughs: np.ndarray, min_depth: float) -> list:
    closes = df["close"].values
    dates  = df["_date"].values
    n      = len(closes)
    events = []
    peak_pos   = np.where(peaks)[0]
    trough_pos = np.where(troughs)[0]

    for pi in peak_pos:
        next_t = trough_pos[trough_pos > pi]
        if not len(next_t): continue
        ti         = next_t[0]
        pp, tp     = closes[pi], closes[ti]
        depth_pct  = (tp - pp) / pp * 100
        if depth_pct > -min_depth: continue

        pd_  = dates[pi]; td_ = dates[ti]
        dur  = (pd.to_datetime(td_) - pd.to_datetime(pd_)).days

        # Recovery
        rec_date = rec_days = None
        future   = np.where((np.arange(n) > ti) & (closes >= pp))[0]
        if len(future):
            ri = future[0]; rd_ = dates[ri]
            rec_date = rd_
            rec_days = (pd.to_datetime(rd_) - pd.to_datetime(td_)).days

        abs_d = abs(depth_pct)
        ctype = ("price" if abs_d > 5 and dur < 20
                 else "time" if abs_d < 5 and dur > 15
                 else "hybrid")

        events.append(CorrectionEvent(
            peak_date=pd_, trough_date=td_,
            peak_price=round(float(pp),4), trough_price=round(float(tp),4),
            depth_pct=round(depth_pct,2), duration_days=dur,
            recovery_date=rec_date, recovery_days=rec_days,
            correction_type=ctype,
        ))
    return events


def _build_profile(events: list, span_days: int) -> dict:
    if not events:
        return {"total_events": 0}
    depths    = np.array([e.depth_pct for e in events])
    durations = np.array([e.duration_days for e in events])
    recs      = np.array([e.recovery_days for e in events if e.recovery_days])
    p33       = float(np.percentile(depths, 67))
    p67       = float(np.percentile(depths, 33))
    return dict(
        total_events=len(events),
        avg_depth_pct=round(float(np.mean(depths)),2),
        median_depth_pct=round(float(np.median(depths)),2),
        std_depth_pct=round(float(np.std(depths)),2),
        max_depth_pct=round(float(np.min(depths)),2),
        min_depth_pct=round(float(np.max(depths)),2),
        avg_duration=round(float(np.mean(durations)),1),
        median_duration=round(float(np.median(durations)),1),
        avg_recovery=round(float(np.mean(recs)),1) if len(recs) else 0,
        frequency_days=round(span_days / max(len(events),1),1),
        shallow_zone=(round(p33,2), round(float(np.max(depths)),2)),
        deep_zone=(round(float(np.min(depths)),2), round(p67,2)),
    )


def _detect_phase(df: pd.DataFrame, profile: dict, pressure_score: float = None) -> dict:
    closes  = df["close"].values; n = len(closes)
    cur     = float(closes[-1])
    dates   = df["_date"].values
    lookback = min(60, n)
    rw      = closes[-lookback:]; rd = dates[-lookback:]
    peak_idx = int(np.argmax(rw))
    peak_p   = float(rw[peak_idx]); peak_d = rd[peak_idx]
    trough_p = float(rw[np.argmin(rw)])
    drawdown = (cur - peak_p) / peak_p * 100 if peak_p > 0 else 0
    days_sp  = (pd.to_datetime(dates[-1]) - pd.to_datetime(peak_d)).days
    slope    = float(closes[-1] - closes[max(0, n-10)])
    trend_up = slope > 0
    avg_dd   = profile.get("avg_depth_pct", -5)
    abs_dd   = abs(drawdown)

    if   abs_dd < 1.5 and trend_up:                              phase, pnum = "Trending",   1
    elif abs_dd < abs(avg_dd)*0.5 and not trend_up:              phase, pnum = "Topping",    2
    elif abs_dd >= abs(avg_dd)*0.5 and not trend_up:             phase, pnum = "Correcting", 3
    elif abs_dd >= abs(avg_dd)*0.5 and trend_up:                 phase, pnum = "Bottoming",  4
    elif abs_dd > 0.5 and trend_up:                              phase, pnum = "Recovering", 5
    else:                                                         phase, pnum = "Trending",   1

    conf = "High" if abs_dd > abs(avg_dd)*0.7 or abs_dd < 1.5 else "Moderate"
    if pressure_score is not None:
        if phase == "Bottoming" and pressure_score < -0.4: conf = "High"
        if phase == "Topping"   and pressure_score >  0.4: conf = "High"

    # Forecast
    forecast = {}
    if profile.get("total_events",0) == 0:
        forecast["note"] = "Insufficient historical data."
    elif phase == "Trending":
        forecast["next_correction"]  = f"Historically every {profile.get('frequency_days',0):.0f} days"
        forecast["expected_depth"]   = f"{avg_dd:.1f}% to {avg_dd - profile.get('std_depth_pct',1):.1f}%"
    elif phase == "Topping":
        forecast["warning"]          = "Correction likely approaching."
        forecast["expected_depth"]   = f"{avg_dd:.1f}% avg"
        forecast["expected_duration"]= f"{profile.get('avg_duration',0):.0f} days typically"
    elif phase == "Correcting":
        exp_total = abs(avg_dd); already = abs_dd
        remaining = max(0, exp_total - already)
        pct_done  = min(100, already / exp_total * 100) if exp_total > 0 else 0
        forecast["already_fallen"]      = f"{drawdown:.1f}%"
        forecast["avg_correction"]      = f"{avg_dd:.1f}%"
        forecast["estimated_remaining"] = f"-{remaining:.1f}% more likely" if remaining > 0 else "May be near bottom"
        forecast["pct_complete"]        = f"{pct_done:.0f}% of avg correction done"
        forecast["days_elapsed"]        = f"{days_sp} days"
        forecast["days_remaining_est"]  = f"~{max(0, profile.get('avg_duration',0)-days_sp):.0f} more days"
    elif phase == "Bottoming":
        forecast["note"]        = "Correction appears near completion."
        forecast["drawdown"]    = f"{drawdown:.1f}%  vs avg {avg_dd:.1f}%"
        forecast["avg_recovery"]= f"{profile.get('avg_recovery',0):.0f} days to recover prior high"
    elif phase == "Recovering":
        forecast["still_below"] = f"{drawdown:.1f}% below recent peak"
        forecast["avg_recovery"]= f"{profile.get('avg_recovery',0):.0f} days historically to full recovery"

    return dict(
        phase=phase, phase_number=pnum,
        current_price=round(cur,4), recent_peak=round(peak_p,4),
        recent_peak_date=str(peak_d),
        current_drawdown=round(drawdown,2),
        days_since_peak=days_sp, confidence=conf,
        forecast=forecast,
    )


def run(
    df             : pd.DataFrame,
    ticker         : str   = "UNKNOWN",
    timeframe      : str   = "daily",
    min_depth      : float = MIN_DEPTH,
    pressure_score : float = None,
    outdir         : Path  = None,
    silent         : bool  = False,
) -> dict:
    """
    Run correction analysis.

    timeframe: "daily" (broad view) or "1min" (intraday fractal view)
    pressure_score: pass from swing/overnight to improve phase confidence
    """
    ticker = ticker.upper()
    outdir = Path(outdir) if outdir else cfg.REPORTS_DIR
    df     = prepare_df(df)

    tf = timeframe.lower()

    # For daily view on 1-min data: aggregate
    if tf == "daily" and df["timestamp"].dt.hour.nunique() > 1:
        log.info(f"[{ticker}] Aggregating 1-min → daily for correction")
        df = (df.groupby("_date")
                .agg(timestamp=("timestamp","first"), open=("open","first"),
                     high=("high","max"), low=("low","min"),
                     close=("close","last"), volume=("volume","sum"))
                .reset_index(drop=True)
                .assign(_date=lambda d: pd.to_datetime(d["timestamp"]).dt.date))

    span_days = (pd.to_datetime(df["_date"].max()) - pd.to_datetime(df["_date"].min())).days
    log.info(f"[{ticker}] Correction ({tf})  {len(df):,} candles  {span_days}d")

    closes  = df["close"].values
    pw      = PEAK_WINDOW if tf == "1min" else 10
    peaks, troughs = _find_peaks_troughs(closes, pw)
    events  = _detect_events(df, peaks, troughs, min_depth)
    profile = _build_profile(events, span_days)
    phase   = _detect_phase(df, profile, pressure_score)

    result = dict(
        ticker=ticker, system="Correction", timeframe=tf,
        status="ok", run_date=str(date.today()),
        data_start=str(df["_date"].min()), data_end=str(df["_date"].max()),
        **profile, **phase, events=events,
    )

    if not silent:
        _print(result)

    _save_csv(result, outdir)
    return result


def _print(r: dict):
    sep = "─" * 68
    emoji = {"Trending":"📈","Topping":"⚠️ ","Correcting":"📉","Bottoming":"🔍","Recovering":"🔄"}
    print(f"\n{sep}")
    print(f"  [{r['ticker']}]  CORRECTION  ({r['timeframe'].upper()})")
    print(sep)
    print(f"  Current Price  : {r['current_price']:,.4f}")
    print(f"  Phase          : {emoji.get(r['phase'],'  ')} {r['phase']}")
    print(f"  Confidence     : {r['confidence']}")
    print(f"  Drawdown       : {r['current_drawdown']:+.2f}%  from peak {r['recent_peak']:,.4f}  ({r['days_since_peak']}d ago)")
    print(sep)
    if r.get("total_events", 0) > 0:
        print(f"  Historical ({r['total_events']} events)")
        print(f"    Avg depth   : {r.get('avg_depth_pct',0):.1f}%   Deepest: {r.get('max_depth_pct',0):.1f}%")
        print(f"    Avg duration: {r.get('avg_duration',0):.0f}d    Recovery: {r.get('avg_recovery',0):.0f}d")
        print(f"    Frequency   : every {r.get('frequency_days',0):.0f}d on average")
    print(sep)
    print(f"  Forecast")
    for k, v in r.get("forecast",{}).items():
        print(f"    {k.replace('_',' ').title():<28}: {v}")
    print(sep)
    if r.get("events"):
        print(f"\n  Recent events:")
        print(f"  {'Peak':<12} {'Trough':<12} {'Depth%':>8}  {'Days':>5}  {'Rec':>6}  {'Type'}")
        for e in r["events"][-6:]:
            rec = f"{e.recovery_days}d" if e.recovery_days else "ongoing"
            print(f"  {str(e.peak_date):<12} {str(e.trough_date):<12} "
                  f"{e.depth_pct:>8.1f}%  {e.duration_days:>5}  {rec:>6}  {e.correction_type}")
    print(f"{sep}\n")


def _save_csv(r: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    sym = r["ticker"]; dt = r["run_date"].replace("-","")

    # Profile + phase summary
    exclude = {"events","above_buckets","below_buckets","forecast"}
    flat    = {k: v for k, v in r.items() if k not in exclude and not isinstance(v, (list, dict, set))}
    for k, v in r.get("forecast",{}).items():
        flat[f"forecast_{k}"] = v
    sp = outdir / f"{sym}_correction_profile_{dt}.csv"
    pd.DataFrame([flat]).to_csv(sp, index=False)

    # Events
    if r.get("events"):
        rows = [dict(
            run_date=r["run_date"], ticker=sym, timeframe=r["timeframe"],
            peak_date=e.peak_date, trough_date=e.trough_date,
            peak_price=e.peak_price, trough_price=e.trough_price,
            depth_pct=e.depth_pct, duration_days=e.duration_days,
            recovery_date=e.recovery_date, recovery_days=e.recovery_days,
            correction_type=e.correction_type, recovered=e.recovered,
        ) for e in r["events"]]
        ep = outdir / f"{sym}_correction_events_{dt}.csv"
        pd.DataFrame(rows).to_csv(ep, index=False)

    log.info(f"[{sym}] Correction CSVs → {outdir}")
