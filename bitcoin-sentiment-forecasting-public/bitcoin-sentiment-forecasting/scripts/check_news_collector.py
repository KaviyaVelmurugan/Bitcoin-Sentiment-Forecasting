"""Offline ingestion checks with explicitly fictional article metadata."""
import tempfile
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collect_news import map_article, open_store, store_article, export, api_time, provider_error, validate_key
from urllib.error import HTTPError
from io import BytesIO

root = Path(__file__).resolve().parents[1]
received = datetime.now(timezone.utc)
assert validate_key("  " + "a" * 32 + "  ") == "a" * 32
for bad_key in ['"' + "a" * 32 + '"', "a" * 31, "a" * 15 + " " + "a" * 16, "a" * 31 + "\x1b"]:
    try:
        validate_key(bad_key)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Malformed key accepted")
assert "." not in api_time(received)
assert api_time(received).endswith("Z")
error = HTTPError("https://newsapi.org/", 400, "bad request", {},
                  BytesIO(b'{"code":"parameterInvalid", "message":"test-secret invalid from parameter"}'))
message = provider_error(error, "test-secret")
assert "test-secret" not in message
assert "parameterInvalid" in message and "invalid from parameter" in message
fixture = {"title": "Fictional Bitcoin test headline", "description": "Fictional test description",
           "source": {"name": "Synthetic fixture"}, "url": "https://example.com/test#fragment",
           "publishedAt": (received - timedelta(days=2)).isoformat()}
row = map_article(fixture, received)
assert row["source_url"] == "https://example.com/test"
assert row["retrieved_at"] == received.isoformat().replace("+00:00", "Z")
with tempfile.TemporaryDirectory(dir=root / "work") as folder:
    with closing(open_store(Path(folder) / "news.sqlite")) as connection, connection:
        assert store_article(connection, row) == 1
        later = map_article(fixture, received + timedelta(hours=1))
        assert store_article(connection, later) == 0
        assert export(connection, Path(folder) / "export.csv") == 1
        saved = connection.execute("SELECT record FROM articles").fetchone()[0]
        assert row["retrieved_at"] in saved
        assert later["retrieved_at"] not in saved
bad_cases = [dict(fixture, publishedAt=(received + timedelta(days=1)).isoformat()),
             dict(fixture, publishedAt="2026-01-01T00:00:00"),
             dict(fixture, url="file:///private"), dict(fixture, description=None)]
for item in bad_cases:
    try:
        map_article(item, received)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid article was accepted")
print("Passed offline checks: first-seen timestamp preservation, duplicate ingestion, CSV export, future/timezone rejection, URL checks and missing-evidence rejection. No live API requests.")
