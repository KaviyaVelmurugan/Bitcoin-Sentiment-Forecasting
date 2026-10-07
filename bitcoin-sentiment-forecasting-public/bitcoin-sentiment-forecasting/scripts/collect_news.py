"""Manual NewsAPI local-development ingestion. No scoring or trading."""
import argparse
import csv
import getpass
import hashlib
import json
import os
import sqlite3
import sys
import re
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

UTC = timezone.utc
URL = "https://newsapi.org/v2/everything"
QUERY = '("bitcoin" OR BTC) AND (crypto OR cryptocurrency)'


def iso(value):
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def api_time(value):
    # NewsAPI documents second-resolution request dates; retain full precision in records.
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_key(value):
    value = value.strip()
    if not re.fullmatch(r"[0-9a-fA-F]{32}", value):
        raise RuntimeError("The pasted key does not have NewsAPI's expected 32-character format. Copy only the key from your account, without quotes, spaces or other text, then run again. Your key has not been sent.")
    return value


def prompt_key():
    # Windows getpass reads individual console characters and may mishandle paste shortcuts.
    if sys.platform != "win32":
        return getpass.getpass("NewsAPI key (hidden): ").strip()
    try:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            value = simpledialog.askstring(
                "Bitcoin Sentiment Forecasting",
                "Paste your NewsAPI key here (Ctrl+V), then click OK.\nIt will not be saved or displayed.",
                show="*", parent=root,
            )
        finally:
            root.destroy()
    except ImportError:
        raise RuntimeError("The password dialog requires tkinter. This Python installation does not include it.") from None
    if value is None:
        raise RuntimeError("Key entry cancelled. No request was sent.")
    return value.strip()


def provider_error(error, key):
    try:
        payload = json.loads(error.read(8192))
        code = str(payload.get("code", "unknown"))
        message = str(payload.get("message", "No provider explanation supplied"))
    except (ValueError, AttributeError, TypeError):
        code = "non_json_response"
        message = "The server or a network intermediary returned a non-JSON response, rather than a normal NewsAPI error. Try the same command from a different network if this persists"
    # Never expose an API key, including one echoed by an error response.
    explanation = (code + ": " + message).replace(key, "[REDACTED]") if key else code + ": " + message
    explanation = re.sub(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s&]+", r"\1[REDACTED]", explanation)
    explanation = " ".join(explanation.split())[:500]
    return f"NewsAPI HTTP {error.code} — {explanation}. No retry was made."


def parse_time(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Publication timestamp has no timezone")
    return value.astimezone(UTC)


def map_article(item, retrieved_at):
    if not isinstance(item, dict):
        raise ValueError("Invalid article object")
    title, summary = str(item.get("title") or "").strip(), str(item.get("description") or "").strip()
    source = item.get("source")
    source_name = str(source.get("name") or "").strip() if isinstance(source, dict) else ""
    parts = urlsplit(str(item.get("url") or "").strip())
    if parts.scheme not in ("http", "https") or not parts.netloc or parts.username or parts.password:
        raise ValueError("Invalid source URL")
    canonical = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))
    published = parse_time(str(item.get("publishedAt") or ""))
    if not title or not summary or not source_name or title == "[Removed]":
        raise ValueError("Required evidence is missing")
    if published > retrieved_at:
        raise ValueError("Publication is in the future")
    identity = hashlib.sha256(canonical.encode()).hexdigest()[:20]
    # Search relevance is a candidate, not verified entity resolution.
    return {"article_id": "news_" + identity, "headline": title, "summary": summary,
            "source_name": source_name, "source_url": canonical, "language": "en",
            "published_at": iso(published), "retrieved_at": iso(retrieved_at),
            "processed_at": iso(datetime.now(UTC)), "asset_ids": "bitcoin",
            "duplicate_group_id": "dup_" + identity,
            "content_license": "provider_terms_review_required", "is_synthetic": "false",
            "schema_version": "1.0.0"}


def open_store(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE IF NOT EXISTS articles (article_id TEXT PRIMARY KEY, record TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS runs (run_at TEXT, requests INTEGER, inserted INTEGER, skipped INTEGER, truncated INTEGER)")
    return connection


def store_article(connection, article):
    # Retain first-seen evidence and retrieval time on repeated ingestion.
    cursor = connection.execute("INSERT OR IGNORE INTO articles VALUES (?, ?)",
                                (article["article_id"], json.dumps(article)))
    return cursor.rowcount


def fetch_page(key, start, end, page):
    params = {"q": QUERY, "searchIn": "title,description", "language": "en",
              "sortBy": "publishedAt", "from": api_time(start), "to": api_time(end),
              "pageSize": 100, "page": page}
    request = Request(URL + "?" + urlencode(params), headers={"X-Api-Key": key,
                      "User-Agent": "BitcoinSentimentResearch/local-development"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        received = datetime.now(UTC)  # after the response, never before it
    except HTTPError as error:
        raise RuntimeError(provider_error(error, key)) from None
    except URLError:
        raise RuntimeError("NewsAPI connection failed; TLS validation remains enabled.") from None
    if not isinstance(payload, dict) or payload.get("status") != "ok" or not isinstance(payload.get("articles"), list):
        raise RuntimeError("NewsAPI returned an invalid or unsuccessful response")
    return payload, received


def export(connection, path):
    rows = [json.loads(item[0]) for item in connection.execute("SELECT record FROM articles ORDER BY article_id")]
    if not rows:
        return 0
    rows.sort(key=lambda row: (row["published_at"], row["article_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument("--prompt-key", action="store_true")
    parser.add_argument("--database", type=Path, default=Path("work/private_news/news.sqlite"))
    parser.add_argument("--export", type=Path, default=Path("work/private_news/news_articles.csv"))
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--max-pages", type=int, default=1)
    args = parser.parse_args()
    key = os.environ.get("NEWS_API_KEY", "").strip()
    if args.check_config:
        print(json.dumps({"key_present": bool(key), "mode": "manual_local_development",
                          "real_records_collected_by_this_command": 0}))
        return
    if not 1 <= args.days <= 28 or args.max_pages != 1:
        parser.error("days must be 1–28; the Developer-plan collector supports only max-pages=1 (100 results)")
    if not key and args.prompt_key:
        key = prompt_key()
    if not key:
        parser.error("Configure NEWS_API_KEY locally or use --prompt-key. Do not share your key in chat.")
    key = validate_key(key)
    now = datetime.now(UTC)
    end = now - timedelta(hours=25)
    start = end - timedelta(days=args.days)
    inserted = skipped = requests = 0
    truncated = False
    with closing(open_store(args.database)) as connection, connection:
        for page in range(1, args.max_pages + 1):
            payload, received = fetch_page(key, start, end, page)
            requests += 1
            articles = payload["articles"]
            for item in articles:
                try:
                    row = map_article(item, received)
                    if not start <= parse_time(row["published_at"]) <= end:
                        raise ValueError("Outside requested publication window")
                except (TypeError, ValueError, OverflowError):
                    skipped += 1
                    continue
                inserted += store_article(connection, row)
            total = int(payload.get("totalResults", 0))
            if len(articles) < 100 or page * 100 >= total:
                break
            truncated = page == args.max_pages
        connection.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
                           (iso(now), requests, inserted, skipped, int(truncated)))
        count = export(connection, args.export)
    print(json.dumps({"requests": requests, "new_records": inserted, "stored_records": count,
                      "skipped": skipped, "pagination_truncated": truncated,
                      "public_release_allowed": "not established; provider and publisher terms require review",
                      "sentiment_scoring": "not performed"}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCollection cancelled. Run again when ready.", file=sys.stderr)
        sys.exit(130)
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
