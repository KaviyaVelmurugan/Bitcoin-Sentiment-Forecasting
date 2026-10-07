"""Leakage-safe chronological market-impact research."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

from .contracts import AssetId, MarketCandle
from .indexing import AGGREGATION_VERSION, IndexEvidence, aggregate_window, build_index_evidence
from .validation import load_csv, parse_annotation_row, parse_market_row, parse_news_row


RESEARCH_VERSION = "1.0.0"
MIN_INFERENCE_SAMPLE = 30
MIN_DIRECTION_SAMPLE = 10
RANDOM_SEED = 20260105


@dataclass(frozen=True, slots=True)
class ResearchObservation:
    observation_id: str
    asset_id: AssetId
    signal_window_start: datetime
    signal_window_end: datetime
    signal_available_at: datetime
    sentiment_index: float
    evidence_count: int
    evidence_coverage: float
    base_candle_id: str
    future_candle_id: str
    base_close: float
    future_close: float
    forward_return: float
    future_volume_change: float
    future_range_volatility: float
    prior_candle_return: float
    outcome_available_at: datetime
    chronological_split: str


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else None


def _ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for index in range(start, end):
            result[ordered[index][0]] = average_rank
        start = end
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return pearson(_ranks(left), _ranks(right)) if len(left) == len(right) else None


def benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    if any(not 0 <= value <= 1 for value in p_values.values()):
        raise ValueError("p-values must be between 0 and 1")
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 1.0
    total = len(ordered)
    for reverse_index in range(total - 1, -1, -1):
        name, value = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, value * total / rank)
        adjusted[name] = round(min(1.0, running), 6)
    return adjusted


def _permutation_p_value(left: Sequence[float], right: Sequence[float]) -> float | None:
    observed = pearson(left, right)
    if observed is None:
        return None
    rng = random.Random(RANDOM_SEED)
    exceedances = 0
    iterations = 2000
    shuffled = list(right)
    for _ in range(iterations):
        rng.shuffle(shuffled)
        value = pearson(left, shuffled)
        exceedances += value is not None and abs(value) >= abs(observed)
    return (exceedances + 1) / (iterations + 1)


def _bootstrap_correlation_interval(
    left: Sequence[float], right: Sequence[float], *, iterations: int = 2000
) -> tuple[float, float] | None:
    if len(left) < MIN_INFERENCE_SAMPLE:
        return None
    rng = random.Random(RANDOM_SEED)
    estimates = []
    for _ in range(iterations):
        indices = [rng.randrange(len(left)) for _ in left]
        value = pearson([left[i] for i in indices], [right[i] for i in indices])
        if value is not None:
            estimates.append(value)
    if len(estimates) < iterations * 0.8:
        return None
    estimates.sort()
    return (round(estimates[int(0.025 * len(estimates))], 6),
            round(estimates[int(0.975 * len(estimates)) - 1], 6))


def build_observations(
    evidence: Sequence[IndexEvidence],
    candles: Sequence[MarketCandle],
    *,
    calculated_at: datetime,
) -> list[ResearchObservation]:
    candles_by_key = {(item.asset_id, item.close_time): item for item in candles}
    candidate_ends = sorted({item.close_time for item in candles})
    raw: list[dict[str, object]] = []
    for signal_end in candidate_ends:
        signal_start = signal_end - timedelta(hours=1)
        for asset_id in AssetId:
            base = candles_by_key.get((asset_id, signal_end))
            future = candles_by_key.get((asset_id, signal_end + timedelta(hours=1)))
            if base is None or future is None:
                continue
            index = aggregate_window(
                evidence, asset_id, signal_start, signal_end, calculated_at=calculated_at,
                half_life=timedelta(hours=1), coverage_target=3,
            ).aggregate
            if index.evidence_count == 0:
                continue
            contributing = [
                item for item in evidence
                if item.asset_id == asset_id
                and signal_start <= item.published_at < signal_end
                and item.available_at <= signal_end
            ]
            signal_available_at = max(
                [signal_end, *(item.available_at for item in contributing)]
            )
            if signal_available_at > future.open_time:
                raise ValueError("signal became available after the forward outcome window opened")
            if future.close_time <= signal_available_at or future.retrieved_at <= signal_available_at:
                raise ValueError("future outcome is not strictly after signal availability")
            identity = f"{asset_id.value}|{signal_end.isoformat()}|1h|{RESEARCH_VERSION}"
            raw.append({
                "observation_id": f"obs_{hashlib.sha256(identity.encode()).hexdigest()[:20]}",
                "asset_id": asset_id,
                "signal_window_start": signal_start,
                "signal_window_end": signal_end,
                "signal_available_at": signal_available_at,
                "sentiment_index": index.sentiment_index,
                "evidence_count": index.evidence_count,
                "evidence_coverage": index.evidence_coverage,
                "base_candle_id": base.candle_id,
                "future_candle_id": future.candle_id,
                "base_close": base.close_price,
                "future_close": future.close_price,
                "forward_return": future.close_price / base.close_price - 1,
                "future_volume_change": future.volume / base.volume - 1 if base.volume else 0.0,
                "future_range_volatility": (future.high_price - future.low_price) / future.open_price,
                "prior_candle_return": base.close_price / base.open_price - 1,
                "outcome_available_at": future.retrieved_at,
            })
    raw.sort(key=lambda item: (item["signal_window_end"], item["asset_id"].value))
    split_index = min(len(raw) - 1, max(1, math.floor(len(raw) * 0.7))) if len(raw) > 1 else len(raw)
    return [
        ResearchObservation(
            **item,
            chronological_split="development" if index < split_index else "test",
        )
        for index, item in enumerate(raw)
    ]


def _direction_accuracy(observations: Sequence[ResearchObservation], predictor: str) -> float | None:
    eligible = [item for item in observations if _sign(item.forward_return) != 0]
    if len(eligible) < MIN_DIRECTION_SAMPLE:
        return None
    if predictor == "sentiment":
        predictions = [_sign(item.sentiment_index) for item in eligible]
    elif predictor == "persistence":
        predictions = [_sign(item.prior_candle_return) for item in eligible]
    else:
        raise ValueError("unknown direction predictor")
    return round(sum(prediction == _sign(item.forward_return) for prediction, item in zip(
        predictions, eligible, strict=True
    )) / len(eligible), 4)


def summarize(observations: Sequence[ResearchObservation]) -> dict[str, object]:
    signals = [item.sentiment_index for item in observations]
    outcomes = {
        "forward_return": [item.forward_return for item in observations],
        "future_volume_change": [item.future_volume_change for item in observations],
        "future_range_volatility": [item.future_range_volatility for item in observations],
    }
    correlations = {
        name: {
            "pearson": None if (value := pearson(signals, values)) is None else round(value, 6),
            "spearman": None if (value := spearman(signals, values)) is None else round(value, 6),
            "bootstrap_95_percent_ci": _bootstrap_correlation_interval(signals, values),
        }
        for name, values in outcomes.items()
    }
    if len(observations) >= MIN_INFERENCE_SAMPLE:
        p_values = {
            name: value for name, values in outcomes.items()
            if (value := _permutation_p_value(signals, values)) is not None
        }
        adjusted = benjamini_hochberg(p_values)
        inference_status = "permutation_tests_run_with_bh_correction"
    else:
        p_values, adjusted = {}, {}
        inference_status = f"not_run_requires_at_least_{MIN_INFERENCE_SAMPLE}_observations"
    test = [item for item in observations if item.chronological_split == "test"]
    return {
        "warning": "Association is not causation, and these results are not trading advice.",
        "sample_count": len(observations),
        "development_count": sum(item.chronological_split == "development" for item in observations),
        "test_count": len(test),
        "inference_status": inference_status,
        "correlations_are_descriptive_only": len(observations) < MIN_INFERENCE_SAMPLE,
        "correlations": correlations,
        "raw_permutation_p_values": p_values,
        "benjamini_hochberg_q_values": adjusted,
        "test_direction_accuracy": {
            "status": (
                "reported" if len(test) >= MIN_DIRECTION_SAMPLE
                else f"not_reported_requires_at_least_{MIN_DIRECTION_SAMPLE}_test_observations"
            ),
            "sentiment_sign": _direction_accuracy(test, "sentiment"),
            "prior_return_persistence": _direction_accuracy(test, "persistence"),
        },
    }


def generate_research_reports(
    news_path: Path,
    annotations_path: Path,
    market_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    articles = load_csv(news_path, parse_news_row, "news_articles")
    annotations = load_csv(annotations_path, parse_annotation_row, "annotations")
    candles = load_csv(market_path, parse_market_row, "market_candles")
    calculated_at = max(item.created_at for item in annotations)
    evidence = build_index_evidence(articles, predicted_at=calculated_at)
    observations = build_observations(evidence, candles, calculated_at=calculated_at)
    summary = summarize(observations)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "aligned_observations.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = tuple(ResearchObservation.__dataclass_fields__)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in observations:
            row = asdict(item)
            row["asset_id"] = item.asset_id.value
            for field in (
                "signal_window_start", "signal_window_end", "signal_available_at", "outcome_available_at"
            ):
                row[field] = row[field].isoformat().replace("+00:00", "Z")
            writer.writerow(row)
    with (output_dir / "research_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with (output_dir / "methodology.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "research_version": RESEARCH_VERSION,
            "aggregation_version": AGGREGATION_VERSION,
            "signal": "hourly sentiment index with evidence_count > 0",
            "entry_reference": "close of candle ending at signal_window_end; descriptive, not executable",
            "forward_return": "next hourly close / base hourly close - 1",
            "volume_change": "next hourly volume / base hourly volume - 1",
            "range_volatility": "(next high - next low) / next open",
            "split": "first 70% development, remaining 30% test, ordered by signal time",
            "minimum_inference_sample": MIN_INFERENCE_SAMPLE,
            "minimum_direction_test_sample": MIN_DIRECTION_SAMPLE,
            "multiple_testing": "permutation p-values with Benjamini-Hochberg correction when eligible",
        }, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chronological market-impact research")
    parser.add_argument("--news", type=Path, default=Path("data/sample/news_articles.csv"))
    parser.add_argument("--annotations", type=Path, default=Path("data/sample/annotations.csv"))
    parser.add_argument("--market", type=Path, default=Path("data/sample/market_candles.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/research"))
    args = parser.parse_args()
    summary = generate_research_reports(args.news, args.annotations, args.market, args.output)
    print(
        f"Market research: observations={summary['sample_count']}, "
        f"inference_status={summary['inference_status']}"
    )


if __name__ == "__main__":
    main()
