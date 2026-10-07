"""Checks for time alignment, gap handling, and independently reconciled errors."""
import csv
import json
import math
import tempfile
from datetime import date, timedelta
from pathlib import Path
import numpy as np
from build_experiment import make_examples, metrics, read_prices

root = Path(__file__).resolve().parents[1]

start = date(2020, 1, 1)
prices = {start + timedelta(days=i): 100 + i for i in range(30)}
sentiment = {start + timedelta(days=i): 30 for i in range(30)}
examples = make_examples(prices, sentiment)
assert len(examples) == 19
assert examples[0]["origin_date"] == start + timedelta(days=10)
assert examples[0]["target_date"] == start + timedelta(days=11)
assert examples[0]["latest_fng_date"] < examples[0]["origin_date"]
assert examples[-1]["target_date"] == start + timedelta(days=29)
del prices[start + timedelta(days=15)]
assert all(not (r["origin_date"] - timedelta(days=10) <= start + timedelta(days=15) <= r["target_date"]) for r in make_examples(prices, sentiment))
assert metrics(np.array([10, 20]), np.array([12, 16])) == {"mae_usd": 3.0, "rmse_usd": math.sqrt(10)}
with tempfile.TemporaryDirectory(dir=root / "work") as directory:
    path = Path(directory) / "invalid.csv"
    for value in ["nan", "inf", "0", "-1"]:
        path.write_text(f"time,PriceUSD\n2020-01-01,{value}\n")
        try:
            read_prices(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid price accepted")

report = json.loads((root / "work/experiment/development_results.json").read_text())
rows = list(csv.DictReader((root / "work/experiment/validation_predictions.csv").open()))
errors = [float(r["persistence"]) - float(r["actual_price"]) for r in rows]
assert math.isclose(sum(abs(e) for e in errors) / len(errors), report["validation_only_scores"]["persistence"]["mae_usd"])
assert math.isclose(math.sqrt(sum(e * e for e in errors) / len(errors)), report["validation_only_scores"]["persistence"]["rmse_usd"])
assert report["final_test_evaluated"] is False
assert report["split"]["train"]["last_target"] < rows[0]["origin_date"]
assert report["split"]["validation"]["last_target"] < str(date.fromisoformat(report["split"]["reserved_final_test"]["first_target"]) - timedelta(days=1))
print("Passed: window boundaries, lagged sentiment, gap exclusion, invalid prices, independent baseline reconciliation, and holdout separation.")
