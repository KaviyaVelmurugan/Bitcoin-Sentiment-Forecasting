"""Explainable multi-label market-event classification and evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from .contracts import AssetId, DatasetSplit, EventLabel, EventPrediction, HumanAnnotation
from .entity_resolution import ENTITY_RESOLUTION_VERSION, extract_target_evidence
from .preprocessing import clean_article, deduplicate_articles
from .sentiment import deduplicated_annotations
from .validation import load_csv, parse_annotation_row, parse_news_row, validate_event_prediction


CLASSIFIER_NAME = "weighted_event_rules"
CLASSIFIER_VERSION = "1.0.0"
MINIMUM_SCORE = 2.0
SECONDARY_RATIO = 0.60

RULES: dict[EventLabel, tuple[tuple[str, float], ...]] = {
    EventLabel.REGULATION_LEGAL: (
        ("regulation", 3), ("regulator", 3), ("lawsuit", 3), ("legal", 2.5),
        ("hearing", 2), ("custody rules", 3), ("committee", 1), ("ban", 3),
    ),
    EventLabel.SECURITY_INCIDENT: (
        ("exploit", 4), ("hack", 4), ("breach", 4), ("security incident", 4),
        ("stolen", 3), ("vulnerability", 3), ("affected contract", 2),
    ),
    EventLabel.EXCHANGE_INCIDENT: (
        ("exchange outage", 4), ("withdrawals suspended", 4), ("insolvency", 4),
        ("exchange incident", 4), ("trading halted", 3), ("liquidation", 2),
    ),
    EventLabel.ADOPTION_PARTNERSHIP: (
        ("partnership", 3), ("adoption", 3), ("settlement pilot", 4),
        ("merchant", 2), ("integrat", 2), ("payment provider", 2), ("launch", 1),
    ),
    EventLabel.PROTOCOL_TECHNICAL: (
        ("protocol", 2), ("upgrade", 2), ("client update", 4), ("client release", 3),
        ("test network", 2), ("test-network", 2), ("finality", 3), ("fork", 3),
    ),
    EventLabel.LISTING_DELISTING: (
        ("delist", 4), ("listed", 3), ("listing", 3), ("trading pair", 2),
    ),
    EventLabel.MACROECONOMIC: (
        ("inflation", 3), ("interest rate", 3), ("central bank", 3),
        ("federal reserve", 4), ("recession", 3), ("gdp", 3),
    ),
    EventLabel.MARKET_COMMENTARY: (
        ("liquidity", 3), ("market note", 3), ("market commentary", 4),
        ("trading volume", 3), ("price movement", 2), ("volatility", 3),
    ),
}


def _contains(text: str, term: str) -> bool:
    if term.endswith("at"):
        return bool(re.search(rf"\b{re.escape(term)}\w*", text, re.IGNORECASE))
    return bool(re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text, re.IGNORECASE))


class ExplainableEventClassifier:
    def score(
        self,
        article,
        target_asset_id: AssetId,
        *,
        predicted_at: datetime,
    ) -> EventPrediction:
        if predicted_at.tzinfo is None or predicted_at.utcoffset() != UTC.utcoffset(predicted_at):
            raise ValueError("predicted_at must use UTC")
        target_evidence = extract_target_evidence(article, target_asset_id)
        scores: dict[EventLabel, float] = {}
        matches: dict[EventLabel, list[str]] = {}
        for label, rules in RULES.items():
            found = [(term, weight) for term, weight in rules if _contains(target_evidence.evidence_text, term)]
            if found:
                scores[label] = sum(weight for _, weight in found)
                matches[label] = [term for term, _ in found]
        eligible = {label: score for label, score in scores.items() if score >= MINIMUM_SCORE}
        if not eligible:
            labels = (EventLabel.INSUFFICIENT_EVIDENCE,)
            primary = labels[0]
            confidence = evidence_strength = 0.0
            matched_terms: tuple[str, ...] = ()
        else:
            ordered = sorted(eligible, key=lambda label: (-eligible[label], label.value))
            primary = ordered[0]
            labels = tuple(
                label for label in ordered if eligible[label] >= eligible[primary] * SECONDARY_RATIO
            )
            total = sum(eligible.values())
            confidence = eligible[primary] / total
            possible = sum(weight for _, weight in RULES[primary])
            evidence_strength = min(1.0, eligible[primary] / possible)
            matched_terms = tuple(
                f"{label.value}:{term}" for label in labels for term in matches[label]
            )
        identity = (
            f"{article.article_id}|{target_asset_id.value}|{CLASSIFIER_NAME}|"
            f"{CLASSIFIER_VERSION}|{ENTITY_RESOLUTION_VERSION}"
        )
        return validate_event_prediction(EventPrediction(
            prediction_id=f"event_{hashlib.sha256(identity.encode()).hexdigest()[:20]}",
            article_id=article.article_id,
            target_asset_id=target_asset_id,
            predicted_labels=labels,
            primary_event_label=primary,
            confidence=round(confidence, 6),
            evidence_strength=round(evidence_strength, 6),
            matched_terms=matched_terms,
            evidence_text=target_evidence.evidence_text,
            classifier_name=CLASSIFIER_NAME,
            classifier_version=CLASSIFIER_VERSION,
            preprocessing_version=article.preprocessing_version,
            evidence_version=ENTITY_RESOLUTION_VERSION,
            predicted_at=predicted_at.astimezone(UTC),
        ))


@dataclass(frozen=True, slots=True)
class EventClassMetrics:
    support: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class EventEvaluation:
    sample_count: int
    primary_accuracy: float
    multi_label_hit_rate: float
    macro_f1: float
    per_class: dict[str, EventClassMetrics]
    confusion_matrix: dict[str, dict[str, int]]


def evaluate_events(
    annotations: Sequence[HumanAnnotation],
    predictions: dict[tuple[str, AssetId], EventPrediction],
) -> EventEvaluation:
    labels = tuple(EventLabel)
    confusion = {actual.value: {predicted.value: 0 for predicted in labels} for actual in labels}
    if not annotations:
        return EventEvaluation(0, 0.0, 0.0, 0.0, {}, confusion)
    hits = 0
    for annotation in annotations:
        prediction = predictions[(annotation.article_id, annotation.target_asset_id)]
        confusion[annotation.event_label.value][prediction.primary_event_label.value] += 1
        hits += annotation.event_label in prediction.predicted_labels
    f1_values = []
    per_class: dict[str, EventClassMetrics] = {}
    for label in labels:
        name = label.value
        support = sum(confusion[name].values())
        tp = confusion[name][name]
        fp = sum(confusion[actual.value][name] for actual in labels if actual != label)
        fn = sum(confusion[name][predicted.value] for predicted in labels if predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        if support:
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            f1_values.append(f1)
            per_class[name] = EventClassMetrics(
                support, round(precision, 4), round(recall, 4), round(f1, 4)
            )
    correct = sum(confusion[label.value][label.value] for label in labels)
    return EventEvaluation(
        len(annotations),
        round(correct / len(annotations), 4),
        round(hits / len(annotations), 4),
        round(sum(f1_values) / len(f1_values), 4),
        per_class,
        confusion,
    )


def generate_event_reports(
    news_path: Path,
    annotations_path: Path,
    output_dir: Path,
    *,
    predicted_at: datetime,
) -> dict[str, EventEvaluation]:
    articles = load_csv(news_path, parse_news_row, "news_articles")
    annotations = load_csv(annotations_path, parse_annotation_row, "annotations")
    cleaned = {article.article_id: clean_article(article) for article in articles}
    dedup = deduplicate_articles(articles, cleaned)
    classifier = ExplainableEventClassifier()
    keys = sorted(
        {(item.article_id, item.target_asset_id) for item in annotations},
        key=lambda item: (item[0], item[1].value),
    )
    predictions = {
        key: classifier.score(cleaned[key[0]], key[1], predicted_at=predicted_at) for key in keys
    }
    slices = {
        "all_records": annotations,
        "deduplicated_records": deduplicated_annotations(annotations, dedup.group_by_article_id),
        "locked_test_split": [item for item in annotations if item.dataset_split == DatasetSplit.TEST],
    }
    metrics = {name: evaluate_events(rows, predictions) for name, rows in slices.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "event_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "prediction_id", "article_id", "target_asset_id", "primary_event_label",
            "predicted_labels", "confidence", "evidence_strength", "matched_terms",
            "evidence_text", "classifier_name", "classifier_version", "preprocessing_version",
            "evidence_version", "predicted_at",
        ))
        writer.writeheader()
        for item in predictions.values():
            writer.writerow({
                "prediction_id": item.prediction_id,
                "article_id": item.article_id,
                "target_asset_id": item.target_asset_id.value,
                "primary_event_label": item.primary_event_label.value,
                "predicted_labels": "|".join(label.value for label in item.predicted_labels),
                "confidence": f"{item.confidence:.6f}",
                "evidence_strength": f"{item.evidence_strength:.6f}",
                "matched_terms": "|".join(item.matched_terms),
                "evidence_text": item.evidence_text,
                "classifier_name": item.classifier_name,
                "classifier_version": item.classifier_version,
                "preprocessing_version": item.preprocessing_version,
                "evidence_version": item.evidence_version,
                "predicted_at": item.predicted_at.isoformat().replace("+00:00", "Z"),
            })
    with (output_dir / "event_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump({
            "warning": "Synthetic sample metrics validate plumbing only; they are not performance claims.",
            "classifier": {"name": CLASSIFIER_NAME, "version": CLASSIFIER_VERSION},
            "evaluations": {name: asdict(value) for name, value in metrics.items()},
        }, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with (output_dir / "event_errors.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "article_id", "target_asset_id", "actual_event", "primary_event",
            "predicted_labels", "matched_terms", "evidence_text", "error_type",
        ))
        writer.writeheader()
        for annotation in annotations:
            item = predictions[(annotation.article_id, annotation.target_asset_id)]
            if item.primary_event_label == annotation.event_label:
                continue
            writer.writerow({
                "article_id": item.article_id,
                "target_asset_id": item.target_asset_id.value,
                "actual_event": annotation.event_label.value,
                "primary_event": item.primary_event_label.value,
                "predicted_labels": "|".join(label.value for label in item.predicted_labels),
                "matched_terms": "|".join(item.matched_terms),
                "evidence_text": item.evidence_text,
                "error_type": "rule_priority_or_vocabulary_gap",
            })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate explainable market-event reports")
    parser.add_argument("--news", type=Path, default=Path("data/sample/news_articles.csv"))
    parser.add_argument("--annotations", type=Path, default=Path("data/sample/annotations.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/events"))
    args = parser.parse_args()
    annotations = load_csv(args.annotations, parse_annotation_row, "annotations")
    metrics = generate_event_reports(
        args.news, args.annotations, args.output,
        predicted_at=max(item.created_at for item in annotations),
    )
    result = metrics["deduplicated_records"]
    print(
        f"Event baseline: samples={result.sample_count}, primary_accuracy={result.primary_accuracy:.4f}, "
        f"multi_label_hit_rate={result.multi_label_hit_rate:.4f}"
    )


if __name__ == "__main__":
    main()
