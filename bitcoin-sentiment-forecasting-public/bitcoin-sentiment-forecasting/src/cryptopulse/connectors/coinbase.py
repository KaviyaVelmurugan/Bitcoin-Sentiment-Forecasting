"""Coinbase Exchange public candle adapter for exchange-specific OHLCV."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from cryptopulse.contracts import AssetId, MarketCandle

from .base import FetchResult, JsonTransport, ProviderResponseError, UrllibJsonTransport


COINBASE_URL = "https://api.exchange.coinbase.com/products/{product_id}/candles"
PRODUCTS = {AssetId.BITCOIN: "BTC-USD", AssetId.ETHEREUM: "ETH-USD"}
GRANULARITY_SECONDS = {"1h": 3600, "1d": 86400}


def _candle_id(product_id: str, start: int, interval: str) -> str:
    key = f"coinbase_exchange|{product_id}|{start}|{interval}"
    return f"candle_{hashlib.sha256(key.encode('utf-8')).hexdigest()[:20]}"


class CoinbaseCandleConnector:
    def __init__(
        self,
        *,
        transport: JsonTransport | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.transport = transport or UrllibJsonTransport()
        self.clock = clock or (lambda: datetime.now(UTC))

    def fetch(
        self,
        asset_id: AssetId,
        *,
        from_time: datetime,
        to_time: datetime,
        interval: str = "1h",
    ) -> FetchResult[MarketCandle]:
        granularity = self._validate_window(from_time, to_time, interval)
        retrieved_at = self.clock().astimezone(UTC)
        product_id = PRODUCTS[asset_id]
        payload = self.transport.get_json(
            COINBASE_URL.format(product_id=product_id),
            params={
                "start": from_time.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "end": to_time.astimezone(UTC).isoformat().replace("+00:00", "Z"),
                "granularity": granularity,
            },
            headers={},
        )
        if not isinstance(payload, list):
            raise ProviderResponseError("Coinbase candle response must be a list")

        records: dict[str, MarketCandle] = {}
        warnings: list[str] = []
        for index, row in enumerate(payload, start=1):
            try:
                candle = self._map_candle(row, asset_id, interval, retrieved_at)
            except (TypeError, ValueError) as exc:
                warnings.append(f"candle {index} skipped: {exc}")
                continue
            if candle.close_time > retrieved_at:
                warnings.append(f"candle {index} skipped: interval was not complete at retrieval time")
                continue
            records.setdefault(candle.candle_id, candle)
        ordered = tuple(sorted(records.values(), key=lambda item: item.open_time))
        return FetchResult(ordered, tuple(warnings))

    @staticmethod
    def _validate_window(from_time: datetime, to_time: datetime, interval: str) -> int:
        for name, value in (("from_time", from_time), ("to_time", to_time)):
            if value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")
        if from_time >= to_time:
            raise ValueError("from_time must be earlier than to_time")
        if interval not in GRANULARITY_SECONDS:
            raise ValueError("interval must be '1h' or '1d'")
        granularity = GRANULARITY_SECONDS[interval]
        bucket_count = (to_time - from_time).total_seconds() / granularity
        if bucket_count > 300:
            raise ValueError("Coinbase requests are limited to at most 300 candle buckets")
        return granularity

    @staticmethod
    def _map_candle(
        row: object,
        asset_id: AssetId,
        interval: str,
        retrieved_at: datetime,
    ) -> MarketCandle:
        if not isinstance(row, list) or len(row) < 6:
            raise ValueError("expected [time, low, high, open, close, volume]")
        start = int(row[0])
        low, high, open_price, close_price, volume = map(float, row[1:6])
        open_time = datetime.fromtimestamp(start, tz=UTC)
        close_time = open_time + timedelta(seconds=GRANULARITY_SECONDS[interval])
        if min(low, high, open_price, close_price) <= 0:
            raise ValueError("prices must be greater than zero")
        if low > min(open_price, close_price) or high < max(open_price, close_price):
            raise ValueError("OHLC values are inconsistent")
        if volume < 0:
            raise ValueError("volume cannot be negative")
        product_id = PRODUCTS[asset_id]
        return MarketCandle(
            candle_id=_candle_id(product_id, start, interval),
            asset_id=asset_id,
            symbol=product_id.split("-")[0],
            quote_currency="USD",
            interval=interval,
            open_time=open_time,
            close_time=close_time,
            open_price=open_price,
            high_price=high,
            low_price=low,
            close_price=close_price,
            volume=volume,
            provider="coinbase_exchange",
            retrieved_at=retrieved_at,
            is_synthetic=False,
        )
