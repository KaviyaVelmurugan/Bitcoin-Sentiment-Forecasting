"""Reuse CryptoPulse scoring locally; retain private text and real scoring times."""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


def process(news, source, output):
    sys.path.insert(0, str(source))
    from cryptopulse.contracts import AssetId
    from cryptopulse.validation import load_csv, parse_news_row
    from cryptopulse.preprocessing import clean_article, deduplicate_articles
    from cryptopulse.entity_resolution import extract_target_evidence
    from cryptopulse.sentiment import VaderBaseline

    articles = load_csv(news, parse_news_row, "news_articles")
    if any(a.is_synthetic for a in articles):
        raise ValueError("Synthetic articles are prohibited in this real-data processing run")
    if len({a.article_id for a in articles}) != len(articles):
        raise ValueError("Duplicate article identifiers")
    cleaned = {a.article_id: clean_article(a) for a in articles}
    duplicate = deduplicate_articles(articles, cleaned)
    analyzer = VaderBaseline()
    output.mkdir(parents=True, exist_ok=True)
    cache_path = output / "scoring_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    results, unscored = [], []
    for article in articles:
        evidence = extract_target_evidence(cleaned[article.article_id], AssetId.BITCOIN)
        if evidence.resolution_status != "resolved":
            unscored.append({"article_id": article.article_id, "resolution_status": evidence.resolution_status})
            continue
        # Include implementation hashes so a changed scorer cannot reuse an old prediction.
        material = article.article_id + cleaned[article.article_id].model_text
        for file in ("sentiment.py", "preprocessing.py", "entity_resolution.py"):
            material += hashlib.sha256((source / "cryptopulse" / file).read_bytes()).hexdigest()
        key = hashlib.sha256(material.encode()).hexdigest()
        if key not in cache:
            prediction = analyzer.score(cleaned[article.article_id], AssetId.BITCOIN,
                                        predicted_at=datetime.now(timezone.utc))
            scored_at = datetime.now(timezone.utc)  # actual scoring completion
            data = asdict(prediction)
            data["predicted_at"] = scored_at.isoformat()
            data["available_at"] = max(article.retrieved_at, article.processed_at, scored_at).isoformat()
            cache[key] = data
        row = dict(cache[key])
        row.update({"published_at": article.published_at.isoformat(),
                    "retrieved_at": article.retrieved_at.isoformat(),
                    "duplicate_group_id": duplicate.group_by_article_id[article.article_id],
                    "source_name": article.source_name, "source_url": article.source_url,
                    "content_license": article.content_license,
                    "training_eligible": False, "entity_status": evidence.resolution_status,
                    "quality_flags": "|".join(cleaned[article.article_id].quality_flags + evidence.quality_flags)})
        results.append(row)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    if results:
        with (output / "bitcoin_sentiment.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    (output / "unscored.json").write_text(json.dumps(unscored, indent=2), encoding="utf-8")
    groups = Counter(duplicate.group_by_article_id.values())
    summary = {"input_articles": len(articles), "bitcoin_scored": len(results),
               "unresolved_or_ambiguous": len(unscored),
               "labels": dict(Counter(r["predicted_label"] for r in results)),
               "duplicate_groups": len(groups), "multi_article_groups": sum(n > 1 for n in groups.values()),
               "largest_group_articles": max(groups.values(), default=0),
               "publication_range": [min(a.published_at for a in articles).isoformat(), max(a.published_at for a in articles).isoformat()] if articles else [],
               "raw_input_sha256": hashlib.sha256(news.read_bytes()).hexdigest(),
               "processed_at": datetime.now(timezone.utc).isoformat(),
               "status": "local development scoring only; no training or public-release eligibility established",
               "limits": ["VADER labels language, not expected returns; no real-data accuracy labels available",
                          "Rule-based duplicate groups require review and can merge distinct related stories",
                          "Scored now, not at publication time; historical labels cannot become past live signals",
                          "No historical daily forecasting index or model was created",
                          "First 100 search results give incomplete source coverage"]}
    (output / "processing_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--news", type=Path, default=root / "work/private_news/news_articles.csv")
    parser.add_argument("--source", type=Path, default=root / "src")
    parser.add_argument("--output", type=Path, default=root / "work/private_news/processed")
    args = parser.parse_args()
    process(args.news, args.source, args.output)
