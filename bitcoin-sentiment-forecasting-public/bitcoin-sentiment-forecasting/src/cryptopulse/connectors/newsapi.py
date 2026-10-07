"""NewsAPI adapter for traceable BTC and ETH article metadata."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from urllib.parse import urlsplit, urlunsplit

from cryptopulse.contracts import AssetId, NewsArticle
from cryptopulse.validation import parse_utc

from .base import FetchResult, JsonTransport, ProviderResponseError, UrllibJsonTransport


NEWS_API_URL = "https://newsapi.org/v2/everything"
ASSET_QUERIES = {
    AssetId.BITCOIN: '("bitcoin" OR BTC) AND (crypto OR cryptocurrency)',
    AssetId.ETHEREUM: '("ethereum" OR ETH) AND (crypto OR cryptocurrency OR blockchain)',
}


def _canonical_url(value: str) -> str:
    parts = urlsplit(value.strip())
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, ""))


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


class NewsApiConnector:
    def __init__(
        self,
        api_key: str,
        *,
        transport: JsonTransport | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("NEWS_API_KEY is required")
        self.api_key = api_key.strip()
        self.transport = transport or UrllibJsonTransport()
        self.clock = clock or (lambda: datetime.now(UTC))

    def fetch(
        self,
        asset_id: AssetId,
        *,
        from_time: datetime,
        to_time: datetime,
        page_size: int = 100,
        max_pages: int = 3,
    ) -> FetchResult[NewsArticle]:
        self._validate_window(from_time, to_time, page_size, max_pages)
        retrieved_at = self.clock().astimezone(UTC)
        records: dict[str, NewsArticle] = {}
        warnings: list[str] = []

        for page in range(1, max_pages + 1):
            payload = self.transport.get_json(
                NEWS_API_URL,
                params={
                    "q": ASSET_QUERIES[asset_id],
                    "searchIn": "title,description",
                    "language": "en",
                    "sortBy": "publishedAt",
                    "from": from_time.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                    "to": to_time.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                    "pageSize": page_size,
                    "page": page,
                },
                headers={"X-Api-Key": self.api_key},
            )
            if not isinstance(payload, dict):
                raise ProviderResponseError("NewsAPI response must be an object")
            if payload.get("status") != "ok":
                code = payload.get("code", "unknown")
                message = payload.get("message", "provider returned an error")
                raise ProviderResponseError(f"NewsAPI error {code}: {message}")
            articles = payload.get("articles")
            if not isinstance(articles, list):
                raise ProviderResponseError("NewsAPI articles must be a list")

            for index, item in enumerate(articles, start=1):
                try:
                    article = self._map_article(item, asset_id, retrieved_at)
                except (KeyError, TypeError, ValueError) as exc:
                    warnings.append(f"page {page} article {index} skipped: {exc}")
                    continue
                records.setdefault(article.article_id, article)

            total_results = payload.get("totalResults", 0)
            if len(articles) < page_size or page * page_size >= int(total_results):
                break
        else:
            warnings.append(f"result collection stopped at configured max_pages={max_pages}")

        return FetchResult(tuple(sorted(records.values(), key=lambda item: item.published_at)), tuple(warnings))

    @staticmethod
    def _validate_window(
        from_time: datetime,
        to_time: datetime,
        page_size: int,
        max_pages: int,
    ) -> None:
        for name, value in (("from_time", from_time), ("to_time", to_time)):
            if value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")
        if from_time >= to_time:
            raise ValueError("from_time must be earlier than to_time")
        if not 1 <= page_size <= 100:
            raise ValueError("page_size must be between 1 and 100")
        if not 1 <= max_pages <= 10:
            raise ValueError("max_pages must be between 1 and 10")

    @staticmethod
    def _map_article(item: object, asset_id: AssetId, retrieved_at: datetime) -> NewsArticle:
        if not isinstance(item, dict):
            raise TypeError("article must be an object")
        headline = str(item.get("title") or "").strip()
        summary = str(item.get("description") or "").strip()
        source_url = _canonical_url(str(item.get("url") or ""))
        published_at = parse_utc(str(item.get("publishedAt") or ""), "publishedAt")
        source = item.get("source")
        source_name = str(source.get("name") or "").strip() if isinstance(source, dict) else ""
        if not all((headline, summary, source_name, source_url)):
            raise ValueError("title, description, source name, and URL are required")
        if published_at > retrieved_at:
            raise ValueError("publishedAt cannot be later than retrieval time")
        article_id = _stable_id("news", source_url)
        return NewsArticle(
            article_id=article_id,
            headline=headline,
            summary=summary,
            source_name=source_name,
            source_url=source_url,
            language="en",
            published_at=published_at,
            retrieved_at=retrieved_at,
            processed_at=retrieved_at,
            asset_ids=(asset_id,),
            duplicate_group_id=_stable_id("dup", source_url),
            content_license="provider_terms_review_required",
            is_synthetic=False,
        )
