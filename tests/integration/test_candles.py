import time
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.candles import backfill, to_frame, upsert_candles
from src.models import MarketCandle

SYM, TF, HOUR = "TEST/USDT", "1h", 3_600_000


class FakeGateway:
    """REST stand-in serving a fixed hourly series, including a still-forming last candle."""

    def __init__(self, n: int) -> None:
        live = int(time.time() * 1000) // HOUR * HOUR  # open time of the forming candle
        self.data = [[live - i * HOUR, 1.0, 2.0, 0.5, 1.5, 10.0] for i in range(n - 1, -1, -1)]
        self.calls = 0

    def timeframe_ms(self, timeframe: str) -> int:
        return HOUR

    async def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int) -> Any:
        self.calls += 1
        return [c for c in self.data if c[0] >= since][:limit]


async def _count(session: AsyncSession) -> int:
    q = select(func.count()).select_from(MarketCandle).where(MarketCandle.symbol == SYM)
    return int(await session.scalar(q) or 0)


async def test_upsert_dedupes_and_updates(session: AsyncSession) -> None:
    ts = 1_700_000_000_000
    await upsert_candles(session, to_frame([[ts, 1, 2, 0.5, 1.5, 10]], SYM, TF))
    await upsert_candles(session, to_frame([[ts, 1, 2, 0.5, 1.75, 12]], SYM, TF))
    assert await _count(session) == 1
    row = await session.scalar(select(MarketCandle).where(MarketCandle.symbol == SYM))
    assert row is not None
    assert row.close == Decimal("1.75") and row.volume == Decimal("12")


async def test_backfill_paginates_skips_forming_candle_and_is_idempotent(
    session: AsyncSession,
) -> None:
    gw = FakeGateway(n=2500)
    written = await backfill(gw, session, SYM, TF, days=200)  # type: ignore[arg-type]
    assert written == 2499  # forming candle excluded
    assert gw.calls == 3  # 1000 + 1000 + 500
    assert await _count(session) == 2499

    # Re-run resumes from the newest stored candle and writes no duplicates.
    await backfill(gw, session, SYM, TF, days=200)  # type: ignore[arg-type]
    assert await _count(session) == 2499
