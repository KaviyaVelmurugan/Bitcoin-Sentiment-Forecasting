"""VADER baseline inference and transparent classification evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .contracts import (
    AssetId,
    CleanedNewsArticle,
    DatasetSplit,
    HumanAnnotation,
    SentimentLabel,
    VaderBaselineScore,
)
from .entity_resolution import ENTITY_RESOLUTION_VERSION, extract_target_evidence
from .preprocessing import clean_article, deduplicate_articles
from .validation import load_csv, parse_annotation_row, parse_news_row


VADER_MODEL_NAME = "vaderSentiment"
VADER_MODEL_VERSION = version("vaderSentiment")
NEGATIVE_THRESHOLD = -0.05
POSITIVE_THRESHOLD = 0.05
LABELS = tuple(SentimentLabel)


class LabelledPrediction(Protocol):
    predicted_label: SentimentLabel


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    support: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    sample_count: int
    accuracy: float
    macro_f1: float
    per_class: dict[str, ClassMetrics]
    confusion_matrix: dict[str, dict[str, int]]


def label_from_compound(compound: float) -> SentimentLabel:
    if not -1 <= compound <= 1:
        raise ValueError("VADER compound score must be between -1 and 1")
    if compound >= POSITIVE_THRESHOLD:
        return SentimentLabel.POSITIVE
    if compound <= NEGATIVE_THRESHOLD:
        return SentimentLabel.NEGATIVE
    return SentimentLabel.NEUTRAL


class VaderBaseline:
    """Thin versioned wrapper around the official VADER scoring rules."""

    def __init__(self, analyzer: SentimentIntensityAnalyzer | None = None) -> None:
        self.analyzer = analyzer or SentimentIntensityAnalyzer()

    def score(
        self,
        article: CleanedNewsArticle,
        target_asset_id: AssetId,
        *,
        predicted_at: datetime,
    ) -> VaderBaselineScore:
        if predicted_at.tzinfo is None or predicted_at.utcoffset() != UTC.utcoffset(predicted_at):
            raise ValueError("predicted_at must use UTC")
        target_evidence = extract_target_evidence(article, target_asset_id)
        values = self.analyzer.polarity_scores(target_evidence.evidence_text)
        required = {"neg", "neu", "pos", "compound"}
        if not required.issubset(values):
            raise ValueError("VADER response is missing required scores")
        proportions = [float(values[name]) for name in ("neg", "neu", "pos")]
        if abs(sum(proportions) - 1.0) > 0.002:
            raise ValueError("VADER negative, neutral, and positive proportions must sum to 1")
        compound = float(values["compound"])
        identity = (
            f"{article.article_id}|{target_asset_id.value}|{VADER_MODEL_NAME}|"
            f"{VADER_MODEL_VERSION}|{ENTITY_RESOLUTION_VERSION}"
        )
        prediction_id = f"vader_{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}"
        return VaderBaselineScore(
            prediction_id=prediction_id,
            article_id=article.article_id,
            target_asset_id=target_asset_id,
            negative_proportion=proportions[0],
            neutral_proportion=proportions[1],
            positive_proportion=proportions[2],
            compound_score=compound,
            predicted_label=label_from_compound(compound),
            evidence_text=target_evidence.evidence_text,
            model_name=VADER_MODEL_NAME,
            model_version=VADER_MODEL_VERSION,
            preprocessing_version=article.preprocessing_version,
            evidence_version=ENTITY_RESOLUTION_VERSION,
            predicted_at=predicted_at.astimezone(UTC),
        )


def evaluate(
    annotations: list[HumanAnnotation],
    predictions: dict[tuple[str, AssetId], LabelledPrediction],
) -> EvaluationMetrics:
    if not annotations:
        return EvaluationMetrics(
            0,
            0.0,
            0.0,
            {label.value: ClassMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0) for label in LABELS},
            {actual.value: {predicted.value: 0 for predicted in LABELS} for actual in LABELS},
        )
    confusion = {actual.value: {predicted.value: 0 for predicted in LABELS} for actual in LABELS}
    for annotation in annotations:
        key = (annotation.article_id, annotation.target_asset_id)
        if key not in predictions:
            raise ValueError(f"missing prediction for {annotation.article_id}/{annotation.target_asset_id.value}")
        confusion[annotation.sentiment_label.value][predictions[key].predicted_label.value] += 1

    per_class: dict[str, ClassMetrics] = {}
    for label in LABELS:
        name = label.value
        tp = confusion[name][name]
        fp = sum(confusion[actual.value][name] for actual in LABELS if actual != label)
        fn = sum(confusion[name][predicted.value] for predicted in LABELS if predicted != label)
        support = sum(confusion[name].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = ClassMetrics(
            support, tp, fp, fn, round(precision, 4), round(recall, 4), round(f1, 4)
        )
    correct = sum(confusion[label.value][label.value] for label in LABELS)
    return EvaluationMetrics(
        sample_count=len(annotations),
        accuracy=round(correct / len(annotations), 4),
        macro_f1=round(sum(metric.f1 for metric in per_class.values()) / len(LABELS), 4),
        per_class=per_class,
        confusion_matrix=confusion,
    )


def deduplicated_annotations(
    annotations: list[HumanAnnotation],
    group_by_article_id: dict[str, str],
) -> list[HumanAnnotation]:
    """Keep one label per duplicate group, target asset, and split using stable annotation order."""
    chosen: dict[tuple[str, AssetId, DatasetSplit], HumanAnnotation] = {}
    for annotation in sorted(annotations, key=lambda item: (item.created_at, item.annotation_id)):
        key = (
            group_by_article_id[annotation.article_id],
            annotation.target_asset_id,
            annotation.dataset_split,
        )
        chosen.setdefault(key, annotation)
    return list(chosen.values())


def _metrics_dict(metrics: EvaluationMetrics) -> dict[str, object]:
    return {
        "sample_count": metrics.sample_count,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "per_class": {name: asdict(value) for name, value in metrics.per_class.items()},
        "confusion_matrix": metrics.confusion_matrix,
    }


def generate_vader_reports(
    news_path: Path,
    annotations_path: Path,
    output_dir: Path,
    *,
    predicted_at: datetime,
) -> dict[str, EvaluationMetrics]:
    articles = load_csv(news_path, parse_news_row, "news_articles")
    annotations = load_csv(annotations_path, parse_annotation_row, "annotations")
    cleaned_by_id = {article.article_id: clean_article(article) for article in articles}
    deduplication = deduplicate_articles(articles, cleaned_by_id)
    baseline = VaderBaseline()
    predictions: dict[tuple[str, AssetId], VaderBaselineScore] = {}
    for annotation in annotations:
        key = (annotation.article_id, annotation.target_asset_id)
        predictions.setdefault(
            key,
            baseline.score(cleaned_by_id[annotation.article_id], annotation.target_asset_id, predicted_at=predicted_at),
        )

    deduplicated = deduplicated_annotations(annotations, deduplication.group_by_article_id)
    test_annotations = [item for item in annotations if item.dataset_split == DatasetSplit.TEST]
    metrics = {
        "all_records": evaluate(annotations, predictions),
        "deduplicated_records": evaluate(deduplicated, predictions),
        "locked_test_split": evaluate(test_annotations, predictions),
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "vader_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        fieldnames = (
            "prediction_id", "article_id", "target_asset_id", "negative_proportion",
            "neutral_proportion", "positive_proportion", "compound_score", "predicted_label",
            "model_name", "model_version", "preprocessing_version", "evidence_version",
            "predicted_at",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in sorted(predictions.values(), key=lambda item: (item.article_id, item.target_asset_id)):
            writer.writerow(
                {
                    "prediction_id": prediction.prediction_id,
                    "article_id": prediction.article_id,
                    "target_asset_id": prediction.target_asset_id.value,
                    "negative_proportion": f"{prediction.negative_proportion:.4f}",
                    "neutral_proportion": f"{prediction.neutral_proportion:.4f}",
                    "positive_proportion": f"{prediction.positive_proportion:.4f}",
                    "compound_score": f"{prediction.compound_score:.4f}",
                    "predicted_label": prediction.predicted_label.value,
                    "model_name": prediction.model_name,
                    "model_version": prediction.model_version,
                    "preprocessing_version": prediction.preprocessing_version,
                    "evidence_version": prediction.evidence_version,
                    "predicted_at": prediction.predicted_at.isoformat().replace("+00:00", "Z"),
                }
            )

    with (output_dir / "vader_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "warning": "Synthetic sample metrics validate the pipeline; they are not performance claims.",
                "thresholds": {"negative_max": NEGATIVE_THRESHOLD, "positive_min": POSITIVE_THRESHOLD},
                "evaluations": {name: _metrics_dict(value) for name, value in metrics.items()},
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")

    with (output_dir / "vader_confusion_matrix.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["evaluation", "actual_label", *(label.value for label in LABELS)])
        for evaluation_name, evaluation in metrics.items():
            for actual in LABELS:
                writer.writerow(
                    [evaluation_name, actual.value]
                    + [evaluation.confusion_matrix[actual.value][predicted.value] for predicted in LABELS]
                )

    with (output_dir / "vader_error_analysis.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "article_id", "target_asset_id", "actual_label", "predicted_label", "compound_score",
                "event_label", "evidence_text", "model_text", "error_type",
            ),
        )
        writer.writeheader()
        for annotation in sorted(annotations, key=lambda item: (item.article_id, item.target_asset_id, item.annotation_id)):
            prediction = predictions[(annotation.article_id, annotation.target_asset_id)]
            if annotation.sentiment_label == prediction.predicted_label:
                continue
            writer.writerow(
                {
                    "article_id": prediction.article_id,
                    "target_asset_id": prediction.target_asset_id.value,
                    "actual_label": annotation.sentiment_label.value,
                    "predicted_label": prediction.predicted_label.value,
                    "compound_score": f"{prediction.compound_score:.4f}",
                    "event_label": annotation.event_label.value,
                    "evidence_text": annotation.evidence_text,
                    "model_text": prediction.evidence_text,
                    "error_type": "target_context_or_lexicon_mismatch",
                }
            )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the CryptoPulse AI VADER baseline reports")
    parser.add_argument("--news", type=Path, default=Path("data/sample/news_articles.csv"))
    parser.add_argument("--annotations", type=Path, default=Path("data/sample/annotations.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/sentiment"))
    args = parser.parse_args()
    annotation_rows = load_csv(args.annotations, parse_annotation_row, "annotations")
    predicted_at = max(item.created_at for item in annotation_rows)
    metrics = generate_vader_reports(
        args.news, args.annotations, args.output, predicted_at=predicted_at
    )
    summary = metrics["deduplicated_records"]
    print(
        f"VADER baseline: deduplicated_samples={summary.sample_count}, "
        f"accuracy={summary.accuracy:.4f}, macro_f1={summary.macro_f1:.4f}"
    )


if __name__ == "__main__":
    main()
