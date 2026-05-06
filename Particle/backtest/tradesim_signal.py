import pandas as pd


def simulate_all_signals(df: pd.DataFrame, hold_days: int = 5):
    trades = []

    for i in range(len(df) - hold_days):
        row = df.iloc[i]

        score = row["score"]
        entry_price = row["price"]

        if score > -0.4:
            continue

        stop = entry_price * 0.98
        target = entry_price * 1.03

        outcome = "timeout"
        days_taken = hold_days
        exit_price = entry_price

        for j in range(1, hold_days + 1):
            future = df.iloc[i + j]

            high = future.get("high", future["price"])
            low = future.get("low", future["price"])

            if low <= stop:
                outcome = "stop_hit"
                exit_price = stop
                days_taken = j
                break

            if high >= target:
                outcome = "target_hit"
                exit_price = target
                days_taken = j
                break

        pnl = (exit_price - entry_price) / entry_price

        trades.append({
            "signal_date": row["date"],
            "entry": entry_price,
            "target": target,
            "stop": stop,
            "outcome": outcome,
            "days": days_taken,
            "pnl": pnl,
            "score": score
        })

    return pd.DataFrame(trades)


def evaluate_signal_quality(trades):
    if trades is None or len(trades) == 0:
        return {
            "total_signals": 0,
            "target_hit_rate": 0,
            "stop_hit_rate": 0,
            "timeout_rate": 0,
            "avg_days": 0,
            "avg_return": 0,
        }

    total = len(trades)

    target_hits = (trades["outcome"] == "target_hit").sum()
    stop_hits   = (trades["outcome"] == "stop_hit").sum()
    timeouts    = (trades["outcome"] == "timeout").sum()

    avg_days   = trades["days"].mean()
    avg_return = trades["pnl"].mean()

    return {
        "total_signals": total,
        "target_hit_rate": round(target_hits / total, 3),
        "stop_hit_rate": round(stop_hits / total, 3),
        "timeout_rate": round(timeouts / total, 3),
        "avg_days": round(avg_days, 2),
        "avg_return": round(avg_return, 4),
    }