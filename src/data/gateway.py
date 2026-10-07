import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import ccxt.pro as ccxtpro

log = logging.getLogger(__name__)

OHLCV = list[list[float]]  # [[ts_ms, open, high, low, close, volume], ...]


class ExchangeGateway:
    """Single ccxt.pro client: REST for history, WebSockets for live data."""

    def __init__(self, exchange_id: str = "binance") -> None:
        self.exchange: Any = getattr(ccxtpro, exchange_id)({"enableRateLimit": True})
        self.prices: dict[str, float] = {}  # in-memory price book: symbol -> last price

    async def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int) -> OHLCV:
        result: OHLCV = await self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        return result

    async def watch_ohlcv(self, symbol: str, timeframe: str) -> OHLCV:
        result: OHLCV = await self.exchange.watch_ohlcv(symbol, timeframe)
        return result

    def timeframe_ms(self, timeframe: str) -> int:
        return int(self.exchange.parse_timeframe(timeframe)) * 1000

    async def stream_prices(
        self, symbol: str, on_tick: Callable[[str, float], Awaitable[None]] | None = None
    ) -> None:
        """Keep `self.prices[symbol]` fresh forever (~1 update/s); run as a background task."""
        while True:
            try:
                ticker = await self.exchange.watch_ticker(symbol)
                self.prices[symbol] = price = float(ticker["last"])
                if on_tick is not None:
                    await on_tick(symbol, price)
            except Exception:  # ccxt.pro reconnects on the next watch call
                log.exception("ticker stream error for %s, retrying", symbol)
                await asyncio.sleep(1)

    async def close(self) -> None:
        await self.exchange.close()
