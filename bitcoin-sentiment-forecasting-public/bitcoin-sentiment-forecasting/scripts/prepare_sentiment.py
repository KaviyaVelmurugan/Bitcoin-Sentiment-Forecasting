"""Validate Alternative.me archive without third-party dependencies."""
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def normalize(payload):
    if payload.get("metadata", {}).get("error"):
        raise ValueError("Provider returned an API error")
    if not isinstance(payload.get("data"), list) or not payload["data"]:
        raise ValueError("Archive must contain observations")
    result, seen_dates = [], set()
    for item in payload["data"]:
        timestamp = int(item["timestamp"])
        value = int(item["value"])
        if not 0 <= value <= 100:
            raise ValueError("Fear and Greed must be between 0 and 100")
        instant = datetime.fromtimestamp(timestamp, timezone.utc)
        date = instant.date().isoformat()
        if date in seen_dates:
            raise ValueError(f"Duplicate observation date: {date}")
        seen_dates.add(date)
        result.append({"date_utc": date, "provider_timestamp_utc": instant.isoformat(),
                       "fng_value": value, "classification": item["value_classification"],
                       "source": "Alternative.me"})
    return sorted(result, key=lambda row: row["date_utc"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    rows = normalize(json.loads(raw))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dates = [datetime.fromisoformat(row["date_utc"]).date() for row in rows]
    report = {"records": len(rows), "first_date": str(dates[0]), "last_date": str(dates[-1]),
              "missing_calendar_days": (dates[-1] - dates[0]).days + 1 - len(dates),
              "source_url": "https://api.alternative.me/fng/?limit=0",
              "raw_sha256": hashlib.sha256(raw).hexdigest(),
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "availability_note": "Provider timestamp is not independently verified publication time",
              "attribution": "Fear and Greed data provided by Alternative.me"}
    args.output.with_suffix(".quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
