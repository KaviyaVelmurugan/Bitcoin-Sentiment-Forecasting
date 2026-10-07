"""Store human annotations separately from unchanged model predictions."""
import argparse
import csv
import json
from pathlib import Path


def run(path, root):
    payload = json.loads(path.read_text(encoding="utf-8"))
    folder = root / "work/private_news"
    known = {r["article_id"] for r in csv.DictReader((folder / "news_articles.csv").open(encoding="utf-8"))}
    scored = {r["article_id"]: r for r in csv.DictReader((folder / "processed/bitcoin_sentiment.csv").open(encoding="utf-8"))}
    seen, reviewed = set(), []
    for row in payload["records"]:
        identity = row["article_id"]
        if identity not in known or identity in seen:
            raise ValueError("Unknown or repeated article identifier")
        seen.add(identity)
        if row["relevance"] not in {"pending", "relevant", "unclear", "irrelevant"}:
            raise ValueError("Invalid relevance judgment")
        if row["human_sentiment"] not in {"pending", "positive", "negative", "neutral", "mixed", "insufficient_evidence"}:
            raise ValueError("Invalid sentiment judgment")
        if row["duplicate_judgment"] not in {"pending", "same_story", "different_story", "not_applicable"}:
            raise ValueError("Invalid duplicate judgment")
        if row["relevance"] != "pending" or row["human_sentiment"] != "pending" or row["duplicate_judgment"] != "pending":
            comparable = row["relevance"] == "relevant" and row["human_sentiment"] in {"positive", "negative", "neutral"} and identity in scored
            reviewed.append({**row, "reviewed_at": payload["reviewed_at"],
                             "model_label": scored.get(identity, {}).get("predicted_label"),
                             "comparison_status": "comparable" if comparable else "requires_review_or_abstention",
                             "eligible_for_confirmed_label": comparable})
    dest = folder / "human_reviews"
    dest.mkdir(parents=True, exist_ok=True)
    # Preserve imports as versions rather than overwriting prior reviewer judgments.
    name = payload["reviewed_at"].replace(":", "-").replace("/", "-").replace("\\", "-")
    (dest / ("review-" + name + ".json")).write_text(json.dumps({"reviewed_at": payload["reviewed_at"], "records": reviewed}, indent=2), encoding="utf-8")
    summary = {"annotations_imported": len(reviewed), "untouched_examples": len(seen) - len(reviewed),
               "comparable_sentiment_labels": sum(r["comparison_status"] == "comparable" for r in reviewed),
               "original_model_predictions_changed": False,
               "accuracy_estimated": False}
    (dest / "latest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("review", type=Path)
    args = parser.parse_args()
    run(args.review, Path(__file__).resolve().parents[1])
