import pandas as pd


def generate_report(df: pd.DataFrame) -> dict:
    df = df.dropna().copy()

    corr = df["score"].corr(df["return"])

    df["pred"] = df["score"].apply(lambda x: 1 if x < 0 else -1)
    df["actual"] = df["return"].apply(lambda x: 1 if x > 0 else -1)

    accuracy = (df["pred"] == df["actual"]).mean()

    df["bucket"] = pd.cut(
        df["score"],
        bins=[-1, -0.4, -0.2, 0, 0.2, 0.4, 1],
        labels=[
            "strong_buy", "buy", "weak_buy",
            "weak_sell", "sell", "strong_sell"
        ],
    )

    bucket_stats = (
        df.groupby("bucket")
        .agg(
            count=("return", "count"),
            avg_return=("return", "mean"),
            win_rate=("return", lambda x: (x > 0).mean()),
        )
        .reset_index()
    )

    return {
        "correlation": round(corr, 4),
        "accuracy": round(accuracy, 4),
        "bucket_stats": bucket_stats,
    }


def generate_simple_conclusion(report: dict) -> str:
    corr = report["correlation"]
    acc = report["accuracy"]

    lines = []

    if corr < -0.1:
        lines.append("When score is low → price usually goes up.")
    else:
        lines.append("Score is not clearly matching price movement.")

    lines.append(f"Around {int(acc*100)} out of 100 signals are correct.")

    lines.append("Strong signals work better. Avoid neutral signals.")

    if acc > 0.55 and corr < -0.1:
        lines.append("System has some edge.")
    else:
        lines.append("System is weak. Needs improvement.")

    return "\n".join(lines)


def generate_actionable_summary(latest_row: dict) -> str:
    score = latest_row.get("score", 0)
    price = latest_row.get("price", 0)

    lines = []
    lines.append(f"Price ~ {price:.2f}")

    if score < -0.4:
        lines.append("BUY: Strong support below.")
    elif score < -0.2:
        lines.append("Weak BUY: Only if confirmation.")
    elif score < 0.2:
        lines.append("NO TRADE: Market unclear.")
    elif score < 0.4:
        lines.append("Weak SELL pressure.")
    else:
        lines.append("SELL / AVOID: Strong resistance.")

    return "\n".join(lines)