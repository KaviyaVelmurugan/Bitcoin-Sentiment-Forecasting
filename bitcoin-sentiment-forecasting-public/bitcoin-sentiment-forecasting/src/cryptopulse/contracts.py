"""Versioned data contracts used across CryptoPulse AI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


SCHEMA_VERSION = "1.0.0"


class AssetId(StrEnum):
    BITCOIN = "bitcoin"
    ETHEREUM = "ethereum"


class SentimentLabel(StrEnum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class EventLabel(StrEnum):
    REGULATION_LEGAL = "regulation_legal"
    SECURITY_INCIDENT = "security_incident"
    EXCHANGE_INCIDENT = "exchange_incident"
    ADOPTION_PARTNERSHIP = "adoption_partnership"
    PROTOCOL_TECHNICAL = "protocol_technical"
    LISTING_DELISTING = "listing_delisting"
    MACROECONOMIC = "macroeconomic"
    MARKET_COMMENTARY = "market_commentary"
    OTHER = "other"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class DatasetSplit(StrEnum):
    DEVELOPMENT = "development"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class NewsArticle:
    article_id: str
    headline: str
    summary: str
    source_name: str
    source_url: str
    language: str
    published_at: datetime
    retrieved_at: datetime
    processed_at: datetime
    asset_ids: tuple[AssetId, ...]
    duplicate_group_id: str
    content_license: str
    is_synthetic: bool
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class CleanedNewsArticle:
    article_id: str
    cleaned_headline: str
    cleaned_summary: str
    model_text: str
    detected_language: str
    language_confidence: float
    quality_flags: tuple[str, ...]
    preprocessing_version: str


@dataclass(frozen=True, slots=True)
class MarketCandle:
    candle_id: str
    asset_id: AssetId
    symbol: str
    quote_currency: str
    interval: str
    open_time: datetime
    close_time: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    provider: str
    retrieved_at: datetime
    is_synthetic: bool
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class HumanAnnotation:
    annotation_id: str
    article_id: str
    target_asset_id: AssetId
    sentiment_label: SentimentLabel
    event_label: EventLabel
    evidence_text: str
    annotator_id: str
    annotation_round: int
    dataset_split: DatasetSplit
    created_at: datetime
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class SentimentPrediction:
    prediction_id: str
    article_id: str
    target_asset_id: AssetId
    negative_probability: float
    neutral_probability: float
    positive_probability: float
    predicted_label: SentimentLabel
    evidence_text: str
    model_name: str
    model_version: str
    preprocessing_version: str
    evidence_version: str
    predicted_at: datetime
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class EventPrediction:
    prediction_id: str
    article_id: str
    target_asset_id: AssetId
    predicted_labels: tuple[EventLabel, ...]
    primary_event_label: EventLabel
    confidence: float
    evidence_strength: float
    matched_terms: tuple[str, ...]
    evidence_text: str
    classifier_name: str
    classifier_version: str
    preprocessing_version: str
    evidence_version: str
    predicted_at: datetime
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class VaderBaselineScore:
    prediction_id: str
    article_id: str
    target_asset_id: AssetId
    negative_proportion: float
    neutral_proportion: float
    positive_proportion: float
    compound_score: float
    predicted_label: SentimentLabel
    evidence_text: str
    model_name: str
    model_version: str
    preprocessing_version: str
    evidence_version: str
    predicted_at: datetime
    schema_version: str = SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class SentimentAggregate:
    aggregate_id: str
    asset_id: AssetId
    window_start: datetime
    window_end: datetime
    sentiment_index: float
    evidence_count: int
    independent_source_count: int
    duplicate_group_count: int
    evidence_coverage: float
    aggregation_version: str
    calculated_at: datetime
    schema_version: str = SCHEMA_VERSION
