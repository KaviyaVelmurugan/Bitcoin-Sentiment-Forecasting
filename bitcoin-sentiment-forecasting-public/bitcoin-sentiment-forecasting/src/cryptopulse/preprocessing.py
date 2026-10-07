"""Deterministic text cleaning, quality checks, and explainable deduplication."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import timedelta
from difflib import SequenceMatcher
from pathlib import Path

from .contracts import CleanedNewsArticle, NewsArticle
from .validation import load_csv, parse_news_row


PREPROCESSING_VERSION = "1.0.0"
TRUNCATION_PATTERN = re.compile(r"\s*(?:…|\.\.\.)?\s*\[\+\d+\s+chars\]\s*$", re.IGNORECASE)
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "with",
    "fictional", "report", "reported", "reports", "described", "same",
}
ENGLISH_MARKERS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "is", "of", "on",
    "or", "that", "the", "this", "to", "was", "with", "while",
}
ALIASES = {"btc": "bitcoin", "eth": "ethereum", "settlements": "settlement", "merchants": "merchant"}


@dataclass(frozen=True, slots=True)
class DuplicateMatch:
    left_article_id: str
    right_article_id: str
    match_type: str
    sequence_similarity: float
    token_jaccard: float
    hours_apart: float
    explanation: str


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    group_by_article_id: dict[str, str]
    matches: tuple[DuplicateMatch, ...]


@dataclass(frozen=True, slots=True)
class DataQualitySummary:
    total_articles: int
    articles_with_warnings: int
    unknown_language_articles: int
    removed_truncation_markers: int
    duplicate_groups: int
    duplicate_articles: int


def clean_text(value: str) -> tuple[str, bool]:
    """Return model-safe text and whether a provider truncation marker was removed."""
    normalised = unicodedata.normalize("NFKC", html.unescape(value or ""))
    normalised = normalised.replace("\u200b", "").replace("\ufeff", "")
    had_truncation = bool(TRUNCATION_PATTERN.search(normalised))
    normalised = TRUNCATION_PATTERN.sub("", normalised)
    normalised = "".join(character for character in normalised if character.isprintable())
    return WHITESPACE_PATTERN.sub(" ", normalised).strip(), had_truncation


def detect_language(text: str) -> tuple[str, float]:
    """Conservative English/unknown baseline; not a general language classifier."""
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return "unknown", 0.0
    ascii_ratio = sum(character.isascii() for character in letters) / len(letters)
    tokens = TOKEN_PATTERN.findall(text.lower())
    marker_ratio = sum(token in ENGLISH_MARKERS for token in tokens) / max(len(tokens), 1)
    confidence = min(1.0, 0.75 * ascii_ratio + 2.0 * marker_ratio)
    return ("en", confidence) if ascii_ratio >= 0.9 and marker_ratio >= 0.02 else ("unknown", confidence)


def clean_article(article: NewsArticle) -> CleanedNewsArticle:
    headline, headline_truncated = clean_text(article.headline)
    summary, summary_truncated = clean_text(article.summary)
    separator = " " if headline.endswith((".", "!", "?")) else ". "
    model_text = f"{headline}{separator}{summary}".strip()
    language, confidence = detect_language(model_text)
    flags: list[str] = []
    if headline_truncated or summary_truncated:
        flags.append("provider_truncation_removed")
    if len(headline) < 12:
        flags.append("short_headline")
    if len(summary) < 40:
        flags.append("short_summary")
    if language == "unknown":
        flags.append("language_unconfirmed")
    if article.retrieved_at - article.published_at > timedelta(hours=24):
        flags.append("retrieval_lag_over_24h")
    if not article.published_at <= article.retrieved_at <= article.processed_at:
        flags.append("invalid_timestamp_order")
    return CleanedNewsArticle(
        article_id=article.article_id,
        cleaned_headline=headline,
        cleaned_summary=summary,
        model_text=model_text,
        detected_language=language,
        language_confidence=round(confidence, 4),
        quality_flags=tuple(flags),
        preprocessing_version=PREPROCESSING_VERSION,
    )


def _comparison_tokens(text: str) -> list[str]:
    tokens = []
    for token in TOKEN_PATTERN.findall(text.lower()):
        token = ALIASES.get(token, token)
        if token not in STOPWORDS and len(token) > 1:
            tokens.append(token)
    return tokens


def _comparison_text(cleaned: CleanedNewsArticle) -> str:
    return " ".join(_comparison_tokens(cleaned.model_text))


def _token_jaccard(left: str, right: str) -> float:
    left_tokens, right_tokens = set(left.split()), set(right.split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _canonical_url(value: str) -> str:
    return value.strip().lower().rstrip("/")


def _group_id(member_ids: list[str]) -> str:
    material = "|".join(sorted(member_ids))
    return f"dup_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:20]}"


def deduplicate_articles(
    articles: list[NewsArticle],
    cleaned_by_id: dict[str, CleanedNewsArticle],
    *,
    max_hours_apart: float = 48.0,
    sequence_threshold: float = 0.44,
    jaccard_threshold: float = 0.20,
) -> DeduplicationResult:
    """Group exact URLs and explainable near duplicates using pairwise baseline metrics."""
    if not 0 <= sequence_threshold <= 1 or not 0 <= jaccard_threshold <= 1:
        raise ValueError("similarity thresholds must be between 0 and 1")
    parent = {article.article_id: article.article_id for article in articles}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    matches: list[DuplicateMatch] = []
    ordered = sorted(articles, key=lambda item: (item.published_at, item.article_id))
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            hours_apart = abs((right.published_at - left.published_at).total_seconds()) / 3600
            if hours_apart > max_hours_apart:
                break
            if not set(left.asset_ids) & set(right.asset_ids):
                continue
            if _canonical_url(left.source_url) == _canonical_url(right.source_url):
                sequence_similarity = token_jaccard = 1.0
                match_type = "exact_url"
                explanation = "Canonical source URLs are identical."
            else:
                left_text = _comparison_text(cleaned_by_id[left.article_id])
                right_text = _comparison_text(cleaned_by_id[right.article_id])
                sequence_similarity = SequenceMatcher(None, left_text, right_text).ratio()
                token_jaccard = _token_jaccard(left_text, right_text)
                if sequence_similarity < sequence_threshold or token_jaccard < jaccard_threshold:
                    continue
                match_type = "near_duplicate"
                explanation = (
                    f"Shared asset within {hours_apart:.2f}h; sequence similarity "
                    f"{sequence_similarity:.3f} >= {sequence_threshold:.3f} and token Jaccard "
                    f"{token_jaccard:.3f} >= {jaccard_threshold:.3f}."
                )
            union(left.article_id, right.article_id)
            matches.append(
                DuplicateMatch(
                    left.article_id,
                    right.article_id,
                    match_type,
                    round(sequence_similarity, 4),
                    round(token_jaccard, 4),
                    round(hours_apart, 4),
                    explanation,
                )
            )

    members_by_root: dict[str, list[str]] = {}
    for article_id in parent:
        members_by_root.setdefault(find(article_id), []).append(article_id)
    group_by_article_id = {
        article_id: _group_id(member_ids)
        for member_ids in members_by_root.values()
        for article_id in member_ids
    }
    return DeduplicationResult(group_by_article_id, tuple(matches))


def build_quality_summary(
    cleaned: list[CleanedNewsArticle],
    deduplication: DeduplicationResult,
) -> DataQualitySummary:
    group_sizes: dict[str, int] = {}
    for group_id in deduplication.group_by_article_id.values():
        group_sizes[group_id] = group_sizes.get(group_id, 0) + 1
    return DataQualitySummary(
        total_articles=len(cleaned),
        articles_with_warnings=sum(bool(article.quality_flags) for article in cleaned),
        unknown_language_articles=sum(article.detected_language == "unknown" for article in cleaned),
        removed_truncation_markers=sum(
            "provider_truncation_removed" in article.quality_flags for article in cleaned
        ),
        duplicate_groups=sum(size > 1 for size in group_sizes.values()),
        duplicate_articles=sum(size for size in group_sizes.values() if size > 1),
    )


def generate_quality_reports(data_path: Path, output_dir: Path) -> DataQualitySummary:
    articles = load_csv(data_path, parse_news_row, "news_articles")
    cleaned = [clean_article(article) for article in articles]
    cleaned_by_id = {article.article_id: article for article in cleaned}
    deduplication = deduplicate_articles(articles, cleaned_by_id)
    summary = build_quality_summary(cleaned, deduplication)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "article_quality.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "article_id", "detected_language", "language_confidence", "headline_length",
                "summary_length", "quality_status", "quality_flags", "duplicate_group_id",
                "preprocessing_version",
            ),
        )
        writer.writeheader()
        for article in cleaned:
            writer.writerow(
                {
                    "article_id": article.article_id,
                    "detected_language": article.detected_language,
                    "language_confidence": f"{article.language_confidence:.4f}",
                    "headline_length": len(article.cleaned_headline),
                    "summary_length": len(article.cleaned_summary),
                    "quality_status": "warning" if article.quality_flags else "pass",
                    "quality_flags": "|".join(article.quality_flags),
                    "duplicate_group_id": deduplication.group_by_article_id[article.article_id],
                    "preprocessing_version": article.preprocessing_version,
                }
            )

    with (output_dir / "duplicate_matches.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "left_article_id", "right_article_id", "match_type", "sequence_similarity",
                "token_jaccard", "hours_apart", "explanation",
            ),
        )
        writer.writeheader()
        for match in deduplication.matches:
            writer.writerow(match.__dict__ if hasattr(match, "__dict__") else {
                field: getattr(match, field) for field in writer.fieldnames
            })

    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {field: getattr(summary, field) for field in summary.__dataclass_fields__},
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CryptoPulse AI data-quality reports")
    parser.add_argument("--input", type=Path, default=Path("data/sample/news_articles.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/data_quality"))
    args = parser.parse_args()
    summary = generate_quality_reports(args.input, args.output)
    print(
        f"Quality report: articles={summary.total_articles}, "
        f"warnings={summary.articles_with_warnings}, "
        f"duplicate_groups={summary.duplicate_groups}, "
        f"duplicate_articles={summary.duplicate_articles}"
    )


if __name__ == "__main__":
    main()
