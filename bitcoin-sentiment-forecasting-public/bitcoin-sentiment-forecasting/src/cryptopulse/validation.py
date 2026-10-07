"""CSV loaders and deterministic validation for the Phase 2 data contracts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from .contracts import (
    SCHEMA_VERSION,
    AssetId,
    DatasetSplit,
    EventLabel,
    EventPrediction,
    HumanAnnotation,
    MarketCandle,
    NewsArticle,
    SentimentAggregate,
    SentimentLabel,
    SentimentPrediction,
)


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    dataset: str
    row_number: int
    field: str
    message: str

    def __str__(self) -> str:
        return f"{self.dataset} row {self.row_number}, {self.field}: {self.message}"


class DatasetValidationError(ValueError):
    """Raised when one or more rows violate a data contract."""

    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        super().__init__("\n".join(str(issue) for issue in issues))


def parse_utc(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"must be an ISO 8601 timestamp: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError(f"{field} must use the UTC timezone")
    return parsed.astimezone(UTC)


def parse_bool(value: str) -> bool:
    normalised = value.strip().lower()
    if normalised not in {"true", "false"}:
        raise ValueError("must be true or false")
    return normalised == "true"


def parse_assets(value: str) -> tuple[AssetId, ...]:
    raw_assets = [item.strip() for item in value.split("|") if item.strip()]
    if not raw_assets:
        raise ValueError("must contain at least one asset")
    assets = tuple(AssetId(item) for item in raw_assets)
    if len(assets) != len(set(assets)):
        raise ValueError("must not contain duplicate assets")
    return assets


def _require(row: dict[str, str], fields: tuple[str, ...]) -> None:
    missing = [field for field in fields if not row.get(field, "").strip()]
    if missing:
        raise ValueError(f"missing required values: {', '.join(missing)}")


def _valid_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def parse_news_row(row: dict[str, str]) -> NewsArticle:
    _require(
        row,
        (
            "article_id", "headline", "summary", "source_name", "source_url", "language",
            "published_at", "retrieved_at", "processed_at", "asset_ids",
            "duplicate_group_id", "content_license", "is_synthetic", "schema_version",
        ),
    )
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {row['schema_version']!r}")
    if not row["article_id"].startswith("news_"):
        raise ValueError("article_id must start with 'news_'")
    if not row["duplicate_group_id"].startswith("dup_"):
        raise ValueError("duplicate_group_id must start with 'dup_'")
    if row["language"] != "en":
        raise ValueError("Phase 2 supports language='en' only")
    if not _valid_url(row["source_url"]):
        raise ValueError("source_url must be an absolute HTTP(S) URL")
    published_at = parse_utc(row["published_at"], "published_at")
    retrieved_at = parse_utc(row["retrieved_at"], "retrieved_at")
    processed_at = parse_utc(row["processed_at"], "processed_at")
    if not published_at <= retrieved_at <= processed_at:
        raise ValueError("timestamps must satisfy published_at <= retrieved_at <= processed_at")
    return NewsArticle(
        article_id=row["article_id"],
        headline=row["headline"],
        summary=row["summary"],
        source_name=row["source_name"],
        source_url=row["source_url"],
        language=row["language"],
        published_at=published_at,
        retrieved_at=retrieved_at,
        processed_at=processed_at,
        asset_ids=parse_assets(row["asset_ids"]),
        duplicate_group_id=row["duplicate_group_id"],
        content_license=row["content_license"],
        is_synthetic=parse_bool(row["is_synthetic"]),
    )


def parse_market_row(row: dict[str, str]) -> MarketCandle:
    _require(
        row,
        (
            "candle_id", "asset_id", "symbol", "quote_currency", "interval", "open_time",
            "close_time", "open_price", "high_price", "low_price", "close_price", "volume",
            "provider", "retrieved_at", "is_synthetic", "schema_version",
        ),
    )
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {row['schema_version']!r}")
    if not row["candle_id"].startswith("candle_"):
        raise ValueError("candle_id must start with 'candle_'")
    asset = AssetId(row["asset_id"])
    expected_symbol = {AssetId.BITCOIN: "BTC", AssetId.ETHEREUM: "ETH"}[asset]
    if row["symbol"] != expected_symbol:
        raise ValueError(f"symbol must be {expected_symbol} for {asset.value}")
    if row["quote_currency"] != "USD":
        raise ValueError("Phase 2 supports quote_currency='USD' only")
    if row["interval"] not in {"1h", "1d"}:
        raise ValueError("interval must be '1h' or '1d'")
    open_time = parse_utc(row["open_time"], "open_time")
    close_time = parse_utc(row["close_time"], "close_time")
    retrieved_at = parse_utc(row["retrieved_at"], "retrieved_at")
    if not open_time < close_time <= retrieved_at:
        raise ValueError("timestamps must satisfy open_time < close_time <= retrieved_at")
    prices = {name: float(row[name]) for name in ("open_price", "high_price", "low_price", "close_price")}
    if any(value <= 0 for value in prices.values()):
        raise ValueError("all prices must be greater than zero")
    if prices["low_price"] > min(prices["open_price"], prices["close_price"]):
        raise ValueError("low_price cannot exceed open_price or close_price")
    if prices["high_price"] < max(prices["open_price"], prices["close_price"]):
        raise ValueError("high_price cannot be below open_price or close_price")
    volume = float(row["volume"])
    if volume < 0:
        raise ValueError("volume cannot be negative")
    return MarketCandle(
        candle_id=row["candle_id"],
        asset_id=asset,
        symbol=row["symbol"],
        quote_currency=row["quote_currency"],
        interval=row["interval"],
        open_time=open_time,
        close_time=close_time,
        open_price=prices["open_price"],
        high_price=prices["high_price"],
        low_price=prices["low_price"],
        close_price=prices["close_price"],
        volume=volume,
        provider=row["provider"],
        retrieved_at=retrieved_at,
        is_synthetic=parse_bool(row["is_synthetic"]),
    )


def parse_annotation_row(row: dict[str, str]) -> HumanAnnotation:
    _require(
        row,
        (
            "annotation_id", "article_id", "target_asset_id", "sentiment_label",
            "event_label", "evidence_text", "annotator_id", "annotation_round",
            "dataset_split", "created_at", "schema_version",
        ),
    )
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {row['schema_version']!r}")
    annotation_round = int(row["annotation_round"])
    if annotation_round < 1:
        raise ValueError("annotation_round must be at least 1")
    return HumanAnnotation(
        annotation_id=row["annotation_id"],
        article_id=row["article_id"],
        target_asset_id=AssetId(row["target_asset_id"]),
        sentiment_label=SentimentLabel(row["sentiment_label"]),
        event_label=EventLabel(row["event_label"]),
        evidence_text=row["evidence_text"],
        annotator_id=row["annotator_id"],
        annotation_round=annotation_round,
        dataset_split=DatasetSplit(row["dataset_split"]),
        created_at=parse_utc(row["created_at"], "created_at"),
    )


def validate_prediction(prediction: SentimentPrediction) -> SentimentPrediction:
    probabilities = {
        SentimentLabel.NEGATIVE: prediction.negative_probability,
        SentimentLabel.NEUTRAL: prediction.neutral_probability,
        SentimentLabel.POSITIVE: prediction.positive_probability,
    }
    if any(value < 0 or value > 1 for value in probabilities.values()):
        raise ValueError("prediction probabilities must be between 0 and 1")
    if abs(sum(probabilities.values()) - 1.0) > 1e-6:
        raise ValueError("prediction probabilities must sum to 1.0")
    expected_label = max(probabilities, key=probabilities.get)
    if prediction.predicted_label != expected_label:
        raise ValueError("predicted_label must match the highest probability")
    if not prediction.evidence_text.strip():
        raise ValueError("prediction evidence_text is required")
    if prediction.predicted_at.tzinfo is None or prediction.predicted_at.utcoffset() != UTC.utcoffset(
        prediction.predicted_at
    ):
        raise ValueError("predicted_at must use the UTC timezone")
    return prediction


def validate_event_prediction(prediction: EventPrediction) -> EventPrediction:
    if not prediction.predicted_labels:
        raise ValueError("event prediction must contain at least one label")
    if prediction.primary_event_label != prediction.predicted_labels[0]:
        raise ValueError("primary_event_label must be the first predicted label")
    if len(set(prediction.predicted_labels)) != len(prediction.predicted_labels):
        raise ValueError("predicted event labels must be unique")
    if not 0 <= prediction.confidence <= 1:
        raise ValueError("event confidence must be between 0 and 1")
    if not 0 <= prediction.evidence_strength <= 1:
        raise ValueError("event evidence_strength must be between 0 and 1")
    if not prediction.evidence_text.strip():
        raise ValueError("event evidence_text is required")
    if prediction.primary_event_label == EventLabel.INSUFFICIENT_EVIDENCE:
        if prediction.matched_terms or prediction.confidence or prediction.evidence_strength:
            raise ValueError("insufficient evidence must not carry matches or positive scores")
    if prediction.predicted_at.tzinfo is None or prediction.predicted_at.utcoffset() != UTC.utcoffset(
        prediction.predicted_at
    ):
        raise ValueError("predicted_at must use the UTC timezone")
    return prediction


def validate_aggregate(aggregate: SentimentAggregate) -> SentimentAggregate:
    if not aggregate.window_start < aggregate.window_end <= aggregate.calculated_at:
        raise ValueError("aggregate times must satisfy window_start < window_end <= calculated_at")
    if not -1 <= aggregate.sentiment_index <= 1:
        raise ValueError("sentiment_index must be between -1 and 1")
    if not 0 <= aggregate.evidence_coverage <= 1:
        raise ValueError("evidence_coverage must be between 0 and 1")
    counts = (
        aggregate.evidence_count,
        aggregate.independent_source_count,
        aggregate.duplicate_group_count,
    )
    if any(value < 0 for value in counts):
        raise ValueError("aggregate counts cannot be negative")
    if aggregate.independent_source_count > aggregate.evidence_count:
        raise ValueError("independent_source_count cannot exceed evidence_count")
    if aggregate.duplicate_group_count > aggregate.evidence_count:
        raise ValueError("duplicate_group_count cannot exceed evidence_count")
    return aggregate


def load_csv(path: Path, parser, dataset: str):
    records = []
    issues: list[ValidationIssue] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row_number, row in enumerate(reader, start=2):
            try:
                records.append(parser(row))
            except (TypeError, ValueError, KeyError) as exc:
                issues.append(ValidationIssue(dataset, row_number, "row", str(exc)))
    if issues:
        raise DatasetValidationError(issues)
    return records


def validate_dataset_directory(data_dir: Path) -> dict[str, int]:
    news = load_csv(data_dir / "news_articles.csv", parse_news_row, "news_articles")
    candles = load_csv(data_dir / "market_candles.csv", parse_market_row, "market_candles")
    annotations = load_csv(data_dir / "annotations.csv", parse_annotation_row, "annotations")

    article_ids = [article.article_id for article in news]
    candle_ids = [candle.candle_id for candle in candles]
    annotation_ids = [annotation.annotation_id for annotation in annotations]
    issues: list[ValidationIssue] = []
    for dataset, values in (
        ("news_articles", article_ids),
        ("market_candles", candle_ids),
        ("annotations", annotation_ids),
    ):
        if len(values) != len(set(values)):
            issues.append(ValidationIssue(dataset, 0, "id", "identifiers must be unique"))
    known_articles = set(article_ids)
    article_assets = {article.article_id: set(article.asset_ids) for article in news}
    for row_number, annotation in enumerate(annotations, start=2):
        if annotation.article_id not in known_articles:
            issues.append(ValidationIssue("annotations", row_number, "article_id", "unknown article"))
        elif annotation.target_asset_id not in article_assets[annotation.article_id]:
            issues.append(
                ValidationIssue(
                    "annotations", row_number, "target_asset_id", "asset is not listed by the article",
                )
            )
    if issues:
        raise DatasetValidationError(issues)
    return {"news_articles": len(news), "market_candles": len(candles), "annotations": len(annotations)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CryptoPulse AI sample contracts")
    parser.add_argument("data_dir", nargs="?", type=Path, default=Path("data/sample"))
    args = parser.parse_args()
    counts = validate_dataset_directory(args.data_dir)
    print(
        "Validated CryptoPulse AI data: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )


if __name__ == "__main__":
    main()
