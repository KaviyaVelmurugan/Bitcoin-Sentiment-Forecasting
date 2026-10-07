"""Historical development experiment; final holdout is never evaluated here."""
import argparse
import csv
import hashlib
import json
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np


def read_prices(path):
    prices, missing = {}, 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            day = date.fromisoformat(row["time"][:10])
            if day in prices:
                raise ValueError(f"Duplicate price date: {day}")
            if not row["PriceUSD"]:
                missing += 1
                continue
            value = float(row["PriceUSD"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid price: {day}")
            prices[day] = value
    return prices, missing


def make_examples(prices, sentiment, window=10):
    examples = []
    for day in sorted(prices):
        target_day = day + timedelta(days=1)
        days = [day - timedelta(days=i) for i in range(window - 1, -1, -1)]
        # Ten price returns require eleven prices. Sentiment is lagged one calendar day.
        needed = [days[0] - timedelta(days=1)] + days
        fng_days = [d - timedelta(days=1) for d in days]
        if target_day not in prices or not all(d in prices for d in needed):
            continue
        if not all(d in sentiment for d in fng_days):
            continue
        price_features = [math.log(prices[d] / prices[d - timedelta(days=1)]) for d in days]
        fng_features = [sentiment[d] / 100 for d in fng_days]
        examples.append({"origin_date": day, "target_date": target_day,
                         "latest_fng_date": fng_days[-1], "current_price": prices[day],
                         "target_price": prices[target_day], "price_features": price_features,
                         "fng_features": fng_features,
                         "target_return": math.log(prices[target_day] / prices[day])})
    return examples


def metrics(actual, predicted):
    errors = predicted - actual
    return {"mae_usd": float(np.mean(np.abs(errors))),
            "rmse_usd": float(np.sqrt(np.mean(errors ** 2)))}


def ridge_forecast(train, validation, combined):
    def features(rows):
        return np.array([r["price_features"] + (r["fng_features"] if combined else []) for r in rows])
    train_x, val_x = features(train), features(validation)
    mean, std = train_x.mean(axis=0), train_x.std(axis=0)
    std[std == 0] = 1
    train_x = np.column_stack([np.ones(len(train)), (train_x - mean) / std])
    val_x = np.column_stack([np.ones(len(validation)), (val_x - mean) / std])
    penalty = np.eye(train_x.shape[1]) * 10.0
    penalty[0, 0] = 0
    target = np.array([r["target_return"] for r in train])
    weights = np.linalg.solve(train_x.T @ train_x + penalty, train_x.T @ target)
    return np.array([r["current_price"] for r in validation]) * np.exp(val_x @ weights)


def run(price_path, sentiment_path, output):
    prices, missing = read_prices(price_path)
    with sentiment_path.open(encoding="utf-8", newline="") as handle:
        sentiment = {date.fromisoformat(r["date_utc"]): int(r["fng_value"]) for r in csv.DictReader(handle)}
    examples = make_examples(prices, sentiment)
    if len(examples) < 100:
        raise ValueError("Insufficient complete examples for the proposed split")
    first, second = int(len(examples) * .7), int(len(examples) * .85)
    train, validation, holdout = examples[:first], examples[first:second], examples[second:]
    # Purge the boundary sample so training's outcome predates the next period's origin.
    train = [r for r in train if r["target_date"] < validation[0]["origin_date"]]
    validation = [r for r in validation if r["target_date"] < holdout[0]["origin_date"]]
    actual = np.array([r["target_price"] for r in validation])
    predictions = {"persistence": np.array([r["current_price"] for r in validation]),
                   "price_ridge": ridge_forecast(train, validation, False),
                   "price_sentiment_ridge": ridge_forecast(train, validation, True)}
    scores = {name: metrics(actual, pred) for name, pred in predictions.items()}
    for value in scores.values():
        value["mae_improvement_vs_persistence_pct"] = 100 * (1 - value["mae_usd"] / scores["persistence"]["mae_usd"])
    def span(rows):
        return {"examples": len(rows), "first_target": str(rows[0]["target_date"]), "last_target": str(rows[-1]["target_date"])}
    report = {"price_source": "Coin Metrics community archive, PriceUSD (UTC end-of-day reference price)",
              "price_url": "https://github.com/coinmetrics/data", "license": "CC BY-NC 4.0",
              "first_price": str(min(prices)), "last_price": str(max(prices)),
              "price_rows_with_missing_values": missing, "complete_examples": len(examples),
              "split": {"train": span(train), "validation": span(validation), "reserved_final_test": span(holdout)},
              "validation_only_scores": scores, "final_test_evaluated": False,
              "settings": {"window_days": 10, "ridge_alpha": 10, "sentiment_lag_days": 1},
              "price_sha256": hashlib.sha256(price_path.read_bytes()).hexdigest(),
              "sentiment_sha256": hashlib.sha256(sentiment_path.read_bytes()).hexdigest(),
              "limitations": ["Archive ends before project start: no current forecast",
                              "Provider timestamps and historical processing latency are not verified real-time availability",
                              "One-day sentiment lag is an assumption, not proof against revised historical values",
                              "Aggregated reference price is not a tradable single-venue closing price",
                              "No LSTM or real CryptoPulse news features trained yet",
                              "Validation results are development evidence, not final performance"]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "development_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["origin_date", "target_date", "latest_fng_date", "actual_price"] + list(predictions)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(validation):
            writer.writerow({"origin_date": row["origin_date"], "target_date": row["target_date"],
                             "latest_fng_date": row["latest_fng_date"], "actual_price": actual[i],
                             **{name: values[i] for name, values in predictions.items()}})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--sentiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.prices, args.sentiment, args.output)
