import pandas as pd


def evaluate(df: pd.DataFrame) -> dict:
    df = df.dropna()

    df["pred"] = df["score"].apply(lambda x: 1 if x < 0 else -1)
    df["actual"] = df["return"].apply(lambda x: 1 if x > 0 else -1)

    accuracy = (df["pred"] == df["actual"]).mean()
    avg_return = df["return"].mean()
    win_rate = (df["return"] > 0).mean()

    return {
        "accuracy": round(accuracy, 3),
        "avg_return": round(avg_return, 4),
        "win_rate": round(win_rate, 3),
        "samples": len(df),
    }


def evaluate_signal_quality(trades: pd.DataFrame) -> dict:
    total = len(trades)

    return {
        "total_signals": total,
        "target_hit_rate": (trades["outcome"] == "target_hit").mean(),
        "stop_hit_rate": (trades["outcome"] == "stop_hit").mean(),
        "timeout_rate": (trades["outcome"] == "timeout").mean(),
        "avg_days": trades["days"].mean(),
        "avg_return": trades["pnl"].mean(),
    }