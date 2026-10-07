"""Reproducible hourly and daily sentiment-index aggregation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Sequence

from .contracts import AssetId, EventLabel, NewsArticle, SentimentAggregate, VaderBaselineScore
from .events import CLASSIFIER_VERSION, ExplainableEventClassifier
from .preprocessing import clean_article, deduplicate_articles
from .sentiment import VADER_MODEL_VERSION, VaderBaseline
from .validation import load_csv, parse_annotation_row, parse_news_row, validate_aggregate


AGGREGATION_VERSION = "1.1.0"
MINIMUM_CONFIDENCE_WEIGHT = 0.10


@dataclass(frozen=True, slots=True)
class IndexEvidence:
    article_id: str
    asset_id: AssetId
    published_at: datetime
    available_at: datetime
    source_name: str
    duplicate_group_id: str
    sentiment_score: float
    confidence_weight: float
    primary_event: EventLabel


@dataclass(frozen=True, slots=True)
class IndexContribution:
    aggregate_id: str
    article_id: str
    asset_id: AssetId
    sentiment_score: float
    confidence_weight: float
    recency_weight: float
    duplicate_weight: float
    source_weight: float
    combined_weight: float
    weighted_contribution: float
    primary_event: EventLabel


@dataclass(frozen=True, slots=True)
class IndexResult:
    aggregate: SentimentAggregate
    contributions: tuple[IndexContribution, ...]
    leading_event: EventLabel | None


def floor_window(value: datetime, resolution: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("window timestamps must use UTC")
    if resolution == "1h":
        return value.replace(minute=0, second=0, microsecond=0)
    if resolution == "1d":
        return value.replace(hour=0, minute=0, second=0, microsecond=0)
    raise ValueError("resolution must be '1h' or '1d'")


def window_delta(resolution: str) -> timedelta:
    if resolution == "1h":
        return timedelta(hours=1)
    if resolution == "1d":
        return timedelta(days=1)
    raise ValueError("resolution must be '1h' or '1d'")


def aggregate_window(
    evidence: Sequence[IndexEvidence],
    asset_id: AssetId,
    window_start: datetime,
    window_end: datetime,
    *,
    calculated_at: datetime,
    half_life: timedelta,
    coverage_target: int,
    use_confidence: bool = True,
    use_recency: bool = True,
) -> IndexResult:
    if not window_start < window_end <= calculated_at:
        raise ValueError("window must end no later than calculated_at")
    if half_life.total_seconds() <= 0 or coverage_target < 1:
        raise ValueError("half_life and coverage_target must be positive")
    rows = [
        item for item in evidence
        if item.asset_id == asset_id
        and window_start <= item.published_at < window_end
        and item.available_at <= window_end
    ]
    duplicate_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for item in rows:
        duplicate_counts[item.duplicate_group_id] = duplicate_counts.get(item.duplicate_group_id, 0) + 1
        source_counts[item.source_name] = source_counts.get(item.source_name, 0) + 1
    identity = (
        f"{asset_id.value}|{window_start.isoformat()}|{window_end.isoformat()}|"
        f"{AGGREGATION_VERSION}|{half_life.total_seconds()}|{coverage_target}|"
        f"{use_confidence}|{use_recency}"
    )
    aggregate_id = f"agg_{hashlib.sha256(identity.encode()).hexdigest()[:20]}"
    contributions: list[IndexContribution] = []
    raw_total_weight = 0.0
    raw_weighted_total = 0.0
    for item in rows:
        age_seconds = max(0.0, (window_end - item.published_at).total_seconds())
        recency_weight = (
            math.exp(-math.log(2) * age_seconds / half_life.total_seconds()) if use_recency else 1.0
        )
        confidence_weight = item.confidence_weight if use_confidence else 1.0
        duplicate_weight = 1 / duplicate_counts[item.duplicate_group_id]
        source_weight = 1 / source_counts[item.source_name]
        combined = confidence_weight * recency_weight * duplicate_weight * source_weight
        raw_total_weight += combined
        raw_weighted_total += item.sentiment_score * combined
        contributions.append(IndexContribution(
            aggregate_id,
            item.article_id,
            item.asset_id,
            item.sentiment_score,
            round(confidence_weight, 6),
            round(recency_weight, 6),
            round(duplicate_weight, 6),
            round(source_weight, 6),
            round(combined, 6),
            round(item.sentiment_score * combined, 6),
            item.primary_event,
        ))
    index_value = raw_weighted_total / raw_total_weight if raw_total_weight else 0.0
    unique_groups = len(duplicate_counts)
    coverage = min(unique_groups / coverage_target, 1.0)
    aggregate = validate_aggregate(SentimentAggregate(
        aggregate_id=aggregate_id,
        asset_id=asset_id,
        window_start=window_start,
        window_end=window_end,
        sentiment_index=round(max(-1.0, min(1.0, index_value)), 6),
        evidence_count=len(rows),
        independent_source_count=len(source_counts),
        duplicate_group_count=unique_groups,
        evidence_coverage=round(coverage, 6),
        aggregation_version=AGGREGATION_VERSION,
        calculated_at=calculated_at,
    ))
    leading = (
        max(contributions, key=lambda item: abs(item.weighted_contribution)).primary_event
        if contributions else None
    )
    return IndexResult(aggregate, tuple(contributions), leading)


def build_index_evidence(
    articles: Sequence[NewsArticle],
    *,
    predicted_at: datetime,
) -> list[IndexEvidence]:
    cleaned = {article.article_id: clean_article(article) for article in articles}
    dedup = deduplicate_articles(list(articles), cleaned)
    vader = VaderBaseline()
    events = ExplainableEventClassifier()
    result: list[IndexEvidence] = []
    for article in articles:
        for asset_id in article.asset_ids:
            sentiment: VaderBaselineScore = vader.score(
                cleaned[article.article_id], asset_id, predicted_at=predicted_at
            )
            event = events.score(cleaned[article.article_id], asset_id, predicted_at=predicted_at)
            result.append(IndexEvidence(
                article.article_id,
                asset_id,
                article.published_at,
                article.processed_at,
                article.source_name,
                dedup.group_by_article_id[article.article_id],
                sentiment.compound_score,
                max(abs(sentiment.compound_score), MINIMUM_CONFIDENCE_WEIGHT),
                event.primary_event_label,
            ))
    return result


def _windows(evidence: Sequence[IndexEvidence], resolution: str) -> list[tuple[datetime, datetime]]:
    start = floor_window(min(item.published_at for item in evidence), resolution)
    end = floor_window(max(item.published_at for item in evidence), resolution) + window_delta(resolution)
    windows = []
    cursor = start
    while cursor < end:
        windows.append((cursor, cursor + window_delta(resolution)))
        cursor += window_delta(resolution)
    return windows


def generate_index_reports(
    news_path: Path,
    annotations_path: Path,
    output_dir: Path,
) -> dict[str, int]:
    articles = load_csv(news_path, parse_news_row, "news_articles")
    annotations = load_csv(annotations_path, parse_annotation_row, "annotations")
    calculated_at = max(item.created_at for item in annotations)
    evidence = build_index_evidence(articles, predicted_at=calculated_at)
    results: list[tuple[str, IndexResult]] = []
    for resolution, half_life, coverage_target in (
        ("1h", timedelta(hours=1), 3),
        ("1d", timedelta(hours=12), 5),
    ):
        for start, end in _windows(evidence, resolution):
            for asset_id in AssetId:
                results.append((resolution, aggregate_window(
                    evidence, asset_id, start, end, calculated_at=calculated_at,
                    half_life=half_life, coverage_target=coverage_target,
                )))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "sentiment_index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "aggregate_id", "resolution", "asset_id", "window_start", "window_end",
            "sentiment_index", "evidence_count", "independent_source_count",
            "duplicate_group_count", "evidence_coverage", "leading_event",
            "aggregation_version", "calculated_at",
        ))
        writer.writeheader()
        for resolution, item in results:
            aggregate = item.aggregate
            writer.writerow({
                "aggregate_id": aggregate.aggregate_id,
                "resolution": resolution,
                "asset_id": aggregate.asset_id.value,
                "window_start": aggregate.window_start.isoformat().replace("+00:00", "Z"),
                "window_end": aggregate.window_end.isoformat().replace("+00:00", "Z"),
                "sentiment_index": f"{aggregate.sentiment_index:.6f}",
                "evidence_count": aggregate.evidence_count,
                "independent_source_count": aggregate.independent_source_count,
                "duplicate_group_count": aggregate.duplicate_group_count,
                "evidence_coverage": f"{aggregate.evidence_coverage:.6f}",
                "leading_event": item.leading_event.value if item.leading_event else "",
                "aggregation_version": aggregate.aggregation_version,
                "calculated_at": aggregate.calculated_at.isoformat().replace("+00:00", "Z"),
            })
    with (output_dir / "index_contributions.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = tuple(IndexContribution.__dataclass_fields__)
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for _, item in results:
            for contribution in item.contributions:
                row = asdict(contribution)
                row["asset_id"] = contribution.asset_id.value
                row["primary_event"] = contribution.primary_event.value
                writer.writerow(row)
    sensitivity_rows = []
    daily_start, daily_end = _windows(evidence, "1d")[0]
    scenarios = (
        ("default", timedelta(hours=12), True, True),
        ("short_half_life", timedelta(hours=3), True, True),
        ("no_confidence", timedelta(hours=12), False, True),
        ("no_recency", timedelta(hours=12), True, False),
    )
    for asset_id in AssetId:
        for name, half_life, use_confidence, use_recency in scenarios:
            item = aggregate_window(
                evidence, asset_id, daily_start, daily_end, calculated_at=calculated_at,
                half_life=half_life, coverage_target=5, use_confidence=use_confidence,
                use_recency=use_recency,
            )
            sensitivity_rows.append({
                "asset_id": asset_id.value,
                "scenario": name,
                "half_life_hours": half_life.total_seconds() / 3600,
                "use_confidence": use_confidence,
                "use_recency": use_recency,
                "sentiment_index": item.aggregate.sentiment_index,
            })
    with (output_dir / "index_sensitivity.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(sensitivity_rows[0]))
        writer.writeheader()
        writer.writerows(sensitivity_rows)
    with (output_dir / "methodology.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "warning": "Synthetic evidence validates aggregation only; this is not a trading signal.",
            "aggregation_version": AGGREGATION_VERSION,
            "sentiment_model": {"name": "vaderSentiment", "version": VADER_MODEL_VERSION},
            "event_classifier_version": CLASSIFIER_VERSION,
            "confidence_proxy": "max(abs(VADER compound), 0.10); not a calibrated probability",
            "coverage": "min(unique duplicate groups / target groups, 1)",
            "hourly": {"half_life_hours": 1, "coverage_target_groups": 3},
            "daily": {"half_life_hours": 12, "coverage_target_groups": 5},
        }, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {
        "index_rows": len(results),
        "contribution_rows": sum(len(item.contributions) for _, item in results),
        "sensitivity_rows": len(sensitivity_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate hourly and daily sentiment indices")
    parser.add_argument("--news", type=Path, default=Path("data/sample/news_articles.csv"))
    parser.add_argument("--annotations", type=Path, default=Path("data/sample/annotations.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/index"))
    args = parser.parse_args()
    counts = generate_index_reports(args.news, args.annotations, args.output)
    print("Sentiment index: " + ", ".join(f"{key}={value}" for key, value in counts.items()))


if __name__ == "__main__":
    main()
