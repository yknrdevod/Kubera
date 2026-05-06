import pandas as pd
from typing import Dict, List
from particle.engines import swing, overnight, correction


def run_backtest(
    df: pd.DataFrame,
    engine: str = "swing",
    timeframe: str = "daily",
    #warmup: int = 200,
) -> pd.DataFrame:

    results: List[Dict] = []

    MIN_CANDLES = 12000

    for i in range(MIN_CANDLES, len(df) - 1):
        train = df.iloc[:i].copy()
        today = df.iloc[i]
        next_day = df.iloc[i + 1]

        if engine == "swing":
            r = swing.run(train, timeframe=timeframe, silent=True)
        elif engine == "overnight":
            r = overnight.run(train, silent=True)
        elif engine == "correction":
            r = correction.run(train, timeframe=timeframe, silent=True)
        else:
            raise ValueError(f"Unknown engine: {engine}")

        score = r.get("pressure_score", None)

        ret = (next_day["close"] - today["close"]) / today["close"]

        results.append({
            "date": today["_date"],
            "price": today["close"],
            "score": score,
            "return": ret,
        })

    return pd.DataFrame(results)