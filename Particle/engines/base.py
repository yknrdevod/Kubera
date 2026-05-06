"""
particle/engines/base.py
========================
Shared computation used by all analysis engines.

Swing and overnight import from here. No duplication.
"""

import math
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── BUCKET ─────────────────────────────────────────────────────────────────────

@dataclass
class Bucket:
    zone_price   : float
    total_volume : float = 0.0
    decayed_vol  : float = 0.0
    candle_count : int   = 0
    outcomes     : list  = field(default_factory=list)
    visit_days   : set   = field(default_factory=set)
    distance     : float = 0.0

    @property
    def visit_count(self) -> int:
        return len(self.visit_days)

    @property
    def freshness(self) -> float:
        return 1.0 / (1.0 + self.visit_count)

    @property
    def avg_outcome(self) -> float:
        valid = [o for o in self.outcomes if o != 0]
        return float(np.mean(valid)) if valid else 0.0

    @property
    def outcome_coverage(self) -> float:
        if not self.outcomes: return 0.0
        return sum(1 for o in self.outcomes if o != 0) / len(self.outcomes)

    @property
    def pressure_contribution(self) -> float:
        return self.decayed_vol * self.avg_outcome * self.distance * self.freshness

    @property
    def strength_label(self) -> str:
        if   self.visit_count == 1 and self.outcome_coverage > 0.6: return "Pristine"
        elif self.visit_count <= 3 and self.outcome_coverage > 0.4: return "Strong"
        elif self.visit_count <= 6:                                  return "Moderate"
        else:                                                        return "Absorbed"


# ── OVERNIGHT BUCKET (extends base) ───────────────────────────────────────────

@dataclass
class OvernightBucket(Bucket):
    """Extends Bucket with institutional and square-off tracking."""
    session_tags : list = field(default_factory=list)
    inst_candles : int  = 0
    sq_candles   : int  = 0

    @property
    def institutional_flag(self) -> bool:
        return self.inst_candles >= 2

    @property
    def squareoff_contaminated(self) -> bool:
        return self.sq_candles > self.candle_count * 0.5

    @property
    def strength_label(self) -> str:
        if   self.institutional_flag and self.visit_count == 1: return "Inst Pristine"
        elif self.institutional_flag:                            return "Institutional"
        elif self.squareoff_contaminated:                        return "Squareoff Noise"
        elif self.visit_count == 1 and self.outcome_coverage > 0.5: return "Pristine"
        elif self.visit_count <= 3 and self.outcome_coverage > 0.3: return "Strong"
        elif self.visit_count <= 6:                              return "Moderate"
        else:                                                    return "Absorbed"


# ── FORWARD OUTCOMES ──────────────────────────────────────────────────────────

def compute_outcomes(closes: np.ndarray, window: int, threshold: float) -> np.ndarray:
    """
    For every candle i, look ahead window candles.
    +1  price rises > threshold
    -1  price falls > threshold
     0  inconclusive / no future data
    """
    n = len(closes)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        end = min(i + window + 1, n)
        if end <= i + 1: continue
        w = closes[i + 1 : end]; b = closes[i]
        if   w.max() >= b * (1 + threshold): out[i] =  1
        elif w.min() <= b * (1 - threshold): out[i] = -1
    return out


# ── BUCKET BUILDER ────────────────────────────────────────────────────────────

def build_buckets(
    df           : pd.DataFrame,
    outcomes     : np.ndarray,
    bucket_pct   : float,
    lam          : float,
    days_from    : str = "timestamp",
    extra_weights: Optional[np.ndarray] = None,
    bucket_class  = None,
    extra_fn      = None,
) -> dict:
    """
    Assign candles to price zones. Accumulate weighted volume.

    extra_weights : per-candle multiplier (session × absorption for overnight)
    bucket_class  : Bucket or OvernightBucket
    extra_fn      : callable(bucket, i, df) for overnight-specific accumulation
    """
    if bucket_class is None:
        bucket_class = Bucket

    price_max = df["close"].max()
    closes    = df["close"].values
    volumes   = df["volume"].values
    dates     = df["_date"].values
    n         = len(df)
    today     = dates[-1]
    buckets   = {}

    for i in range(n):
        idx  = int(closes[i] / (price_max * bucket_pct))
        zp   = (idx + 0.5) * price_max * bucket_pct
        days = (today - dates[i]).days if days_from == "timestamp" else (n - 1 - i)
        dec  = math.exp(-lam * days)
        ext  = float(extra_weights[i]) if extra_weights is not None else 1.0
        cw   = dec * ext

        if idx not in buckets:
            buckets[idx] = bucket_class(zone_price=round(zp, 6))

        b = buckets[idx]
        b.total_volume += volumes[i]
        b.decayed_vol  += volumes[i] * cw
        b.candle_count += 1
        b.outcomes.append(float(outcomes[i]))
        b.visit_days.add(dates[i])

        if extra_fn is not None:
            extra_fn(b, i, df)

    return buckets


# ── PRESSURE ──────────────────────────────────────────────────────────────────

def calc_pressure(buckets: dict, current: float) -> dict:
    """Net pressure score [-1, +1] from all buckets."""
    for b in buckets.values():
        b.distance = abs(b.zone_price - current) / current if current > 0 else 0.0

    ap = bp = norm = 0.0
    ab, bb = [], []

    for b in buckets.values():
        norm += b.decayed_vol * b.distance * b.freshness
        if b.zone_price > current:
            ap += b.pressure_contribution; ab.append(b)
        else:
            bp += b.pressure_contribution; bb.append(b)

    net   = ap - bp
    score = float(np.clip(net / norm if norm > 0 else 0, -1.0, 1.0))

    return {
        "score"          : round(score, 4),
        "above_pressure" : round(ap, 2),
        "below_pressure" : round(bp, 2),
        "norm"           : round(norm, 2),
        "above_buckets"  : sorted(ab, key=lambda b: b.zone_price),
        "below_buckets"  : sorted(bb, key=lambda b: b.zone_price, reverse=True),
    }


# ── TREND ─────────────────────────────────────────────────────────────────────

def compute_trend(df: pd.DataFrame, short_n: int, long_n: int) -> dict:
    closes = df["close"].values; n = len(closes)
    sn = min(short_n, n); ln = min(long_n, n)
    cur = float(closes[-1])
    ma_s = float(np.mean(closes[-sn:])); ma_l = float(np.mean(closes[-ln:]))
    if   cur > ma_s > ma_l: d = "Uptrend"
    elif cur < ma_s < ma_l: d = "Downtrend"
    elif cur > ma_s:         d = "Mixed (above short MA)"
    else:                    d = "Mixed (below short MA)"
    return {"direction": d, "ma_short": round(ma_s,4), "ma_long": round(ma_l,4)}


def trend_alignment(score: float, direction: str) -> str:
    b = score < -0.2; br = score > 0.2
    if   b  and "Uptrend"   in direction: return "Aligned    (buy + uptrend)"
    elif br and "Downtrend" in direction: return "Aligned    (sell + downtrend)"
    elif b  and "Downtrend" in direction: return "Opposed    (buy vs downtrend)"
    elif br and "Uptrend"   in direction: return "Opposed    (sell vs uptrend)"
    else:                                 return "Neutral    (mixed trend)"


# ── GAP RISK ──────────────────────────────────────────────────────────────────

GAP_LOOKBACK  = 60
GAP_LARGE_THR = 0.005


def compute_gap_risk(df: pd.DataFrame) -> dict:
    daily = (df.groupby("_date")
               .agg(day_open=("open","first"), day_close=("close","last"))
               .reset_index().sort_values("_date")
               .tail(GAP_LOOKBACK + 1).reset_index(drop=True))
    if len(daily) < 5: return {"available": False}
    gaps = []
    for i in range(1, len(daily)):
        pc = daily.loc[i-1,"day_close"]; co = daily.loc[i,"day_open"]
        if pc > 0: gaps.append((co - pc) / pc)
    if not gaps: return {"available": False}
    g = np.array(gaps); std = float(np.std(g)); large = float((np.abs(g)>GAP_LARGE_THR).mean())
    risk = float(np.clip(std*50 + large*0.5, 0, 1))
    return dict(
        available=True, sample_days=len(gaps),
        avg_gap_pct=round(float(np.mean(g))*100,3),
        median_gap_pct=round(float(np.median(g))*100,3),
        std_gap_pct=round(std*100,3),
        positive_pct=round(float((g>0).mean())*100,1),
        negative_pct=round(float((g<0).mean())*100,1),
        large_gap_pct=round(large*100,1),
        gap_risk_score=round(risk,3),
        risk_label="Low" if risk<0.25 else ("Medium" if risk<0.55 else "High"),
    )


# ── INTERPRET ─────────────────────────────────────────────────────────────────

def interpret_score(score: float) -> tuple:
    if   score < -0.4: return "Strong Floor",         "Fresh validated support dominates.",  "LOOK TO BUY on pullbacks"
    elif score <  0.0: return "Mild Support",          "Buyer zones outweigh seller zones.",  "CAUTIOUS LONG"
    elif score <  0.4: return "Mild Pressure",         "Seller zones building above.",        "HOLD — do not add"
    else:              return "Heavy Seller Pressure",  "Large trapped supply above.",         "AVOID / EXIT"


# ── CSV HELPERS ───────────────────────────────────────────────────────────────

def levels_to_rows(above_buckets, below_buckets, current, score, meta) -> list:
    rows = []
    for side, bl in [("RESISTANCE", above_buckets), ("SUPPORT", below_buckets)]:
        for b in bl:
            row = {
                **meta, "side": side,
                "zone_price": round(b.zone_price,4),
                "avg_outcome": round(b.avg_outcome,4),
                "outcome_coverage": round(b.outcome_coverage,4),
                "visit_count": b.visit_count,
                "freshness": round(b.freshness,4),
                "distance_pct": round(b.distance*100,4),
                "decayed_vol": round(b.decayed_vol,2),
                "total_volume": round(b.total_volume,2),
                "candle_count": b.candle_count,
                "strength_label": b.strength_label,
                "force": round(b.pressure_contribution,4),
                "current_price": current,
                "pressure_score": score,
            }
            rows.append(row)
    return rows


# ── DATAFRAME PREP ────────────────────────────────────────────────────────────

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, parse timestamp, add _date."""
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    if "timestamp" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["_date"] = df["timestamp"].dt.date
    return df
