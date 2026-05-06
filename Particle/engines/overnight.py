"""
particle/engines/overnight.py
==============================
Overnight pressure engine — enter during session, exit next morning.

Layers 1-5 via base. Additional layers:
    Layer 6: Session multipliers (open/morning/midday/afternoon/power_hour/squareoff)
    Layer 7: Institutional absorption (high volume, tiny candle body)
    Layer 8: Trend context
    Layer 9: Gap risk
"""

import logging
import math
import numpy as np
import pandas as pd
from collections import Counter
from datetime import date, time
from pathlib import Path

from .. import config as cfg
from .base import (
    OvernightBucket, build_buckets, calc_pressure, compute_gap_risk,
    compute_outcomes, compute_trend, interpret_score,
    levels_to_rows, trend_alignment, prepare_df,
)

log = logging.getLogger(__name__)

# ── SESSION MAP ───────────────────────────────────────────────────────────────

SESSION_MAP = [
    (time( 9,15), time( 9,45), "open",       1.40),
    (time( 9,45), time(11, 0), "morning",    1.00),
    (time(11, 0), time(13, 0), "midday",     0.80),
    (time(13, 0), time(14,45), "afternoon",  0.90),
    (time(14,45), time(15,15), "power_hour", 1.30),
    (time(15,15), time(15,30), "squareoff",  0.05),
]
INST_SESSIONS   = {"open", "power_hour"}
INST_MULT       = cfg.OVERNIGHT["absorption_multiplier"] if "absorption_multiplier" in cfg.OVERNIGHT else 1.50
ABSORB_PCT      = cfg.OVERNIGHT["absorption_percentile"]
MINS_PER_DAY    = 390


def _get_session(t: time):
    for s, e, name, mult in SESSION_MAP:
        if s <= t < e: return name, mult
    return "other", 0.8


def _compute_absorption(df: pd.DataFrame) -> np.ndarray:
    """
    Institutional absorption score per candle.
    High score = large volume, tiny candle body = institution accumulating quietly.
    Normalised to [0, 1].
    """
    body = np.abs(df["close"].values - df["open"].values) * df["close"].values
    body = np.where(body < 0.001, 0.001, body)
    raw  = df["volume"].values / body
    mn, mx = raw.min(), raw.max()
    return (raw - mn) / (mx - mn) if mx > mn else np.zeros(len(raw))


def _session_summary(df: pd.DataFrame) -> dict:
    today = df["_date"].max()
    td    = df[df["_date"] == today].copy()
    if td.empty: return {}
    td["_sn"] = td["_time"].apply(lambda t: _get_session(t)[0])
    tv  = td["volume"].sum()
    sqv = td[td["_sn"] == "squareoff"]["volume"].sum()
    return dict(
        today_open=float(td["open"].iloc[0]), today_close=float(td["close"].iloc[-1]),
        today_high=float(td["high"].max()),   today_low=float(td["low"].min()),
        day_chg_pct=round((td["close"].iloc[-1]-td["open"].iloc[0])/td["open"].iloc[0]*100,2),
        sq_vol_pct=round(sqv/tv*100 if tv > 0 else 0, 1),
    )


def _overnight_verdict(score: float, result: dict, sess: dict):
    sb  = result["below_buckets"]
    ins = sb[0].institutional_flag if sb else False
    sq  = sess.get("sq_vol_pct", 0)

    if score < -0.4 and ins:
        a, c = "STRONG OVERNIGHT HOLD / BUY", "High"
        r = "Institutional accumulation confirmed. Fresh support. Pressure favors upside."
    elif score < -0.4:
        a, c = "HOLD — Monitor at Open", "Moderate"
        r = "Buyer pressure confirmed but no institutional signal. Retail support only."
    elif score < 0.0:
        a, c = "CAUTIOUS HOLD — Tight Stop", "Low"
        r = "Mild buyer pressure. Use tight stop if holding overnight."
    elif score < 0.4:
        a, c = "AVOID OVERNIGHT / EXIT", "Moderate"
        r = "Seller pressure above. Risk/reward does not support overnight hold."
    else:
        a, c = "EXIT — Do Not Hold Overnight", "High"
        r = "Heavy seller pressure. Likely gap-down or morning weakness."

    if sq > 40:
        r += f" WARNING: {sq:.0f}% of today's volume was auto square-off."
    return a, c, r


def run(
    df     : pd.DataFrame,
    ticker : str  = "UNKNOWN",
    outdir : Path = None,
    silent : bool = False,
) -> dict:
    """
    Run overnight pressure analysis on 1-min OHLCV.

    Returns result dict. Saves summary + levels CSV to outdir.
    """
    ticker = ticker.upper()
    outdir = Path(outdir) if outdir else cfg.REPORTS_DIR
    df     = prepare_df(df)

    if "_time" not in df.columns:
        df["_time"] = df["timestamp"].dt.time

    span = (pd.to_datetime(df["_date"].max()) - pd.to_datetime(df["_date"].min())).days
    if span < cfg.OVERNIGHT["min_days"]:
        msg = f"Need >= {cfg.OVERNIGHT['min_days']} days. Got {span}."
        log.warning(f"[{ticker}] {msg}")
        return {"ticker": ticker, "status": "error", "message": msg}

    log.info(f"[{ticker}] Overnight  {len(df):,} candles")

    # Absorption scores
    absorption  = _compute_absorption(df)
    inst_thr    = np.percentile(absorption, ABSORB_PCT)

    # Build per-candle session + absorption weights
    times    = df["_time"].values
    sess_w   = np.array([_get_session(t)[1] for t in times])
    sess_names = [_get_session(t)[0] for t in times]
    is_inst  = np.array([(sess_names[i] in INST_SESSIONS and absorption[i] >= inst_thr)
                          for i in range(len(df))], dtype=bool)
    inst_w   = np.where(is_inst, INST_MULT, 1.0)
    extra_w  = sess_w * inst_w

    # Track extra bucket fields (inst/sq candles)
    def extra_fn(b, i, df):
        if is_inst[i]: b.inst_candles += 1
        if sess_names[i] == "squareoff": b.sq_candles += 1
        b.session_tags.append(sess_names[i])

    closes   = df["close"].values
    outcomes = compute_outcomes(closes,
                                cfg.OVERNIGHT["forward_window"],
                                cfg.OVERNIGHT["outcome_threshold"])
    buckets  = build_buckets(df, outcomes,
                              cfg.OVERNIGHT["bucket_pct"],
                              cfg.OVERNIGHT["lambda"],
                              days_from="timestamp",
                              extra_weights=extra_w,
                              bucket_class=OvernightBucket,
                              extra_fn=extra_fn)
    current  = float(closes[-1])
    pressure = calc_pressure(buckets, current)
    sess     = _session_summary(df)
    trend    = compute_trend(df,
                             MINS_PER_DAY * 20,
                             MINS_PER_DAY * 50)
    gap      = compute_gap_risk(df)

    score    = pressure["score"]
    sig, meaning, _ = interpret_score(score)
    alignment        = trend_alignment(score, trend["direction"])
    action, conf, logic = _overnight_verdict(score, pressure, sess)

    sb = pressure["below_buckets"]
    rb = pressure["above_buckets"]
    stop   = round(sb[0].zone_price * 0.98, 4) if sb else None
    target = round(rb[0].zone_price, 4)         if rb else None
    rr     = round(rb[0].distance / sb[0].distance, 2) if sb and rb and sb[0].distance > 0 else None

    result = dict(
        ticker=ticker, system="Overnight", status="ok",
        run_date=str(date.today()),
        current_price=round(current,4),
        data_start=str(df["_date"].min()), data_end=str(df["_date"].max()),
        candles=len(df),
        pressure_score=score, signal=sig, meaning=meaning,
        overnight_action=action, overnight_confidence=conf, overnight_logic=logic,
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
        today_open=sess.get("today_open"),
        today_close=sess.get("today_close"),
        today_high=sess.get("today_high"),
        today_low=sess.get("today_low"),
        day_chg_pct=sess.get("day_chg_pct"),
        sq_vol_pct=sess.get("sq_vol_pct"),
        stop_loss=stop, target=target, risk_reward=rr,
        above_buckets=pressure["above_buckets"],
        below_buckets=pressure["below_buckets"],
    )

    if not silent:
        _print(result, gap, sess)

    _save_csv(result, gap, sess, outdir)
    return result


def _print(r: dict, gap: dict, sess: dict):
    sep = "─" * 68
    print(f"\n{sep}")
    print(f"  [{r['ticker']}]  OVERNIGHT")
    print(sep)
    print(f"  Price          : {r['current_price']:,.4f}")
    print(f"  Pressure Score : {r['pressure_score']:+.4f}")
    print(f"  Signal         : {r['signal']}")
    if sess:
        sqw = "  ⚠ HIGH" if sess.get("sq_vol_pct",0) > 40 else ""
        print(f"  Today          : O={sess.get('today_open',0):,.4f}  "
              f"H={sess.get('today_high',0):,.4f}  L={sess.get('today_low',0):,.4f}  "
              f"C={sess.get('today_close',0):,.4f}  ({sess.get('day_chg_pct',0):+.2f}%)")
        print(f"  Square-off vol : {sess.get('sq_vol_pct',0):.1f}%{sqw}")
    print(sep)
    print(f"  Trend          : {r['trend_direction']}")
    print(f"  Alignment      : {r['trend_alignment']}")
    if gap.get("available"):
        print(f"  Gap Risk       : {gap['risk_label']}  score={gap['gap_risk_score']:.3f}")
    print(sep)
    hdr = f"  {'Zone':>10}  {'Outcome':>8}  {'Visits':>7}  {'Fresh':>7}  {'State':<16}  {'Force':>10}"
    print(f"\n  RESISTANCE (above):"); print(hdr)
    for b in r["above_buckets"][:5]:
        star = " *" if b.institutional_flag else ""
        sq   = "[SQ]" if b.squareoff_contaminated else ""
        print(f"  {b.zone_price:>10,.4f}  {b.avg_outcome:>+8.3f}  {b.visit_count:>7}  "
              f"{b.freshness:>7.3f}  {b.strength_label+star+sq:<16}  {b.pressure_contribution:>+10.2f}")
    print(f"\n  SUPPORT (below):"); print(hdr)
    for b in r["below_buckets"][:5]:
        star = " *" if b.institutional_flag else ""
        sq   = "[SQ]" if b.squareoff_contaminated else ""
        print(f"  {b.zone_price:>10,.4f}  {b.avg_outcome:>+8.3f}  {b.visit_count:>7}  "
              f"{b.freshness:>7.3f}  {b.strength_label+star+sq:<16}  {b.pressure_contribution:>+10.2f}")
    print(f"\n{sep}")
    print(f"  OVERNIGHT VERDICT")
    print(f"  Action     : {r['overnight_action']}")
    print(f"  Confidence : {r['overnight_confidence']}")
    print(f"  Logic      : {r['overnight_logic']}")
    if r["stop_loss"] and r["target"]:
        print(f"\n  Stop: {r['stop_loss']:,.4f}  Target: {r['target']:,.4f}  R:R 1:{r['risk_reward']:.1f}")
    print(f"\n  * = Institutional  [SQ] = Squareoff noise")
    print(f"{sep}\n")


def _save_csv(r: dict, gap: dict, sess: dict, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    sym = r["ticker"]; dt = r["run_date"].replace("-","")
    keys = ["run_date","ticker","system","current_price","pressure_score",
            "signal","meaning","overnight_action","overnight_confidence","overnight_logic",
            "above_pressure","below_pressure","trend_direction","ma_short","ma_long",
            "trend_alignment","stop_loss","target","risk_reward",
            "today_open","today_close","today_high","today_low","day_chg_pct","sq_vol_pct"]
    meta = {k: r.get(k,"") for k in keys}
    meta.update({f"gap_{k}": gap.get(k,"") for k in
                 ["risk_score","risk_label","avg_gap_pct","std_gap_pct",
                  "positive_pct","negative_pct","large_gap_pct"]})
    sp = outdir / f"{sym}_overnight_summary_{dt}.csv"
    pd.DataFrame([meta]).to_csv(sp, index=False)

    level_meta = {"run_date": r["run_date"], "ticker": sym, "system": "Overnight"}
    rows = []
    for side, bl in [("RESISTANCE", r["above_buckets"]), ("SUPPORT", r["below_buckets"])]:
        for b in bl:
            row = {**level_meta, "side": side,
                   "zone_price": round(b.zone_price,4),
                   "avg_outcome": round(b.avg_outcome,4),
                   "outcome_coverage": round(b.outcome_coverage,4),
                   "visit_count": b.visit_count, "freshness": round(b.freshness,4),
                   "distance_pct": round(b.distance*100,4),
                   "decayed_vol": round(b.decayed_vol,2),
                   "total_volume": round(b.total_volume,2),
                   "candle_count": b.candle_count,
                   "institutional_flag": b.institutional_flag,
                   "squareoff_contaminated": b.squareoff_contaminated,
                   "strength_label": b.strength_label,
                   "force": round(b.pressure_contribution,4),
                   "current_price": r["current_price"],
                   "pressure_score": r["pressure_score"]}
            rows.append(row)
    lp = outdir / f"{sym}_overnight_levels_{dt}.csv"
    pd.DataFrame(rows).to_csv(lp, index=False)
    log.info(f"[{sym}] Overnight CSVs → {outdir}")
