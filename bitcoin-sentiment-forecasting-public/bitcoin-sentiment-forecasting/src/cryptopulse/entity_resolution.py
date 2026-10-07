"""Explainable crypto-asset entity resolution and target evidence extraction."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from .contracts import AssetId, CleanedNewsArticle
from .preprocessing import clean_article
from .validation import load_csv, parse_news_row


ENTITY_RESOLUTION_VERSION = "1.0.0"
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")
CLAUSE_PATTERN = re.compile(r"\s+(?:while|whereas|but)\s+|;\s*", re.IGNORECASE)
CRYPTO_CONTEXT = re.compile(
    r"\b(?:crypto(?:currency)?|digital asset|token|blockchain|coin|liquidity|market|protocol|"
    r"network|wallet|exchange|settlement|price)\b",
    re.IGNORECASE,
)

ASSET_PATTERNS: dict[AssetId, tuple[tuple[str, re.Pattern[str], bool], ...]] = {
    AssetId.BITCOIN: (
        ("bitcoin", re.compile(r"\bbitcoin\b", re.IGNORECASE), False),
        ("BTC", re.compile(r"(?<![A-Za-z0-9])\$?BTC(?![A-Za-z0-9])"), True),
        ("XBT", re.compile(r"(?<![A-Za-z0-9])XBT(?![A-Za-z0-9])"), True),
    ),
    AssetId.ETHEREUM: (
        ("ethereum", re.compile(r"\bethereum\b", re.IGNORECASE), False),
        ("ether", re.compile(r"\bether\b", re.IGNORECASE), False),
        ("ETH", re.compile(r"(?<![A-Za-z0-9])\$?ETH(?![A-Za-z0-9])"), True),
    ),
}


@dataclass(frozen=True, slots=True)
class EntityMention:
    asset_id: AssetId
    alias: str
    matched_text: str
    start: int
    end: int
    ticker_only: bool


@dataclass(frozen=True, slots=True)
class TargetEvidence:
    article_id: str
    target_asset_id: AssetId
    evidence_text: str
    mentions: tuple[EntityMention, ...]
    evidence_segment_count: int
    resolution_status: str
    quality_flags: tuple[str, ...]
    resolution_version: str = ENTITY_RESOLUTION_VERSION


def find_mentions(text: str) -> tuple[EntityMention, ...]:
    mentions: list[EntityMention] = []
    for asset_id, patterns in ASSET_PATTERNS.items():
        for alias, pattern, ticker_only in patterns:
            for match in pattern.finditer(text):
                mentions.append(
                    EntityMention(asset_id, alias, match.group(), match.start(), match.end(), ticker_only)
                )
    return tuple(sorted(mentions, key=lambda item: (item.start, item.end, item.asset_id.value)))


def _segments(text: str) -> list[str]:
    sentences = SENTENCE_PATTERN.split(text.strip())
    return [
        part.strip(" ,.;")
        for sentence in sentences
        for part in CLAUSE_PATTERN.split(sentence)
        if part.strip(" ,.;")
    ]


def extract_target_evidence(
    article: CleanedNewsArticle,
    target_asset_id: AssetId,
) -> TargetEvidence:
    all_mentions = find_mentions(article.model_text)
    target_mentions = tuple(item for item in all_mentions if item.asset_id == target_asset_id)
    selected = [segment for segment in _segments(article.model_text) if any(
        mention.asset_id == target_asset_id for mention in find_mentions(segment)
    )]
    flags: list[str] = []
    if not target_mentions:
        flags.append("target_not_mentioned")
        return TargetEvidence(
            article.article_id,
            target_asset_id,
            article.model_text,
            (),
            0,
            "not_found_fallback_full_text",
            tuple(flags),
        )
    ticker_mentions = [item for item in target_mentions if item.ticker_only]
    named_mentions = [item for item in target_mentions if not item.ticker_only]
    if ticker_mentions and not named_mentions and not CRYPTO_CONTEXT.search(article.model_text):
        flags.append("ticker_context_ambiguous")
        status = "ambiguous_ticker"
    else:
        status = "resolved"
    other_assets = {item.asset_id for item in all_mentions if item.asset_id != target_asset_id}
    if other_assets:
        flags.append("multi_asset_article")
    evidence_text = ". ".join(dict.fromkeys(selected))
    return TargetEvidence(
        article.article_id,
        target_asset_id,
        evidence_text,
        target_mentions,
        len(selected),
        status,
        tuple(flags),
    )


def generate_entity_report(news_path: Path, output_path: Path) -> list[TargetEvidence]:
    articles = load_csv(news_path, parse_news_row, "news_articles")
    evidence = [
        extract_target_evidence(clean_article(article), asset_id)
        for article in articles
        for asset_id in article.asset_ids
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "article_id", "target_asset_id", "resolution_status", "mention_count",
                "aliases", "evidence_segment_count", "evidence_text", "quality_flags",
                "resolution_version",
            ),
        )
        writer.writeheader()
        for item in evidence:
            writer.writerow(
                {
                    "article_id": item.article_id,
                    "target_asset_id": item.target_asset_id.value,
                    "resolution_status": item.resolution_status,
                    "mention_count": len(item.mentions),
                    "aliases": "|".join(dict.fromkeys(mention.alias for mention in item.mentions)),
                    "evidence_segment_count": item.evidence_segment_count,
                    "evidence_text": item.evidence_text,
                    "quality_flags": "|".join(item.quality_flags),
                    "resolution_version": item.resolution_version,
                }
            )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate target-specific entity evidence")
    parser.add_argument("--news", type=Path, default=Path("data/sample/news_articles.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/entity_resolution/target_evidence.csv"),
    )
    args = parser.parse_args()
    evidence = generate_entity_report(args.news, args.output)
    resolved = sum(item.resolution_status == "resolved" for item in evidence)
    print(f"Entity evidence: targets={len(evidence)}, resolved={resolved}")


if __name__ == "__main__":
    main()
