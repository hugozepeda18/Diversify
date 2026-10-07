import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime

import pandas as pd
from sqlalchemy import ColumnElement, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.data.gateway import OHLCV, ExchangeGateway
from src.models import MarketCandle

log = logging.getLogger(__name__)

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
PAGE = 1000  # Binance max klines per REST call; 8 cols * 1000 rows stays under PG's param limit

OnClose = Callable[[str, str, pd.DataFrame], Awaitable[None]]


def to_frame(ohlcv: OHLCV, symbol: str, timeframe: str) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["symbol"] = symbol
    df["timeframe"] = timeframe
    return df


async def upsert_candles(session: AsyncSession, df: pd.DataFrame) -> None:
    """Insert candles; re-seen (timestamp, symbol, timeframe) rows overwrite OHLCV values."""
    if df.empty:
        return
    stmt = insert(MarketCandle).values(df.to_dict("records"))
    stmt = stmt.on_conflict_do_update(
        index_elements=["timestamp", "symbol", "timeframe"],
        set_={c: stmt.excluded[c] for c in COLUMNS[1:]},
    )
    await session.execute(stmt)


def _where(symbol: str, timeframe: str) -> tuple[ColumnElement[bool], ...]:
    return (MarketCandle.symbol == symbol, MarketCandle.timeframe == timeframe)


async def latest_timestamp(session: AsyncSession, symbol: str, timeframe: str) -> datetime | None:
    return await session.scalar(
        select(func.max(MarketCandle.timestamp)).where(*_where(symbol, timeframe))
    )


async def load_candles(
    session: AsyncSession, symbol: str, timeframe: str, days: int
) -> pd.DataFrame:
    """Stored candles for the last `days`, as floats indexed by timestamp."""
    start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    q = (
        select(MarketCandle.timestamp, *(getattr(MarketCandle, c) for c in COLUMNS[1:]))
        .where(*_where(symbol, timeframe), MarketCandle.timestamp >= start)
        .order_by(MarketCandle.timestamp)
    )
    rows = (await session.execute(q)).all()
    df = pd.DataFrame(rows, columns=COLUMNS).set_index("timestamp")
    return df.astype(float)


async def backfill(
    gw: ExchangeGateway, session: AsyncSession, symbol: str, timeframe: str, days: int
) -> int:
    """Seed closed candles via REST, resuming from the newest stored one. Returns rows written."""
    tf_ms = gw.timeframe_ms(timeframe)
    now = int(time.time() * 1000)
    start = now - days * 86_400_000
    span = (
        await session.execute(
            select(func.min(MarketCandle.timestamp), func.max(MarketCandle.timestamp)).where(
                *_where(symbol, timeframe)
            )
        )
    ).one()
    # Resume after the newest candle if storage already covers `start`; otherwise fetch from
    # `start` (overlapping rows are upserted, so re-fetching is harmless).
    covered = span[0] is not None and span[0].timestamp() * 1000 <= start + tf_ms
    since = int(span[1].timestamp() * 1000) if covered else start
    written = 0
    while since < now:
        page = await gw.fetch_ohlcv(symbol, timeframe, since, PAGE)
        if not page:
            break
        df = to_frame(page, symbol, timeframe)
        # Drop the still-forming candle; only closed candles are persisted.
        df = df[
            df["timestamp"] + pd.Timedelta(milliseconds=tf_ms)
            <= pd.Timestamp(now, unit="ms", tz="UTC")
        ]
        await upsert_candles(session, df)
        await session.commit()
        written += len(df)
        since = int(page[-1][0]) + tf_ms
        if len(page) < PAGE:
            break
    log.info("backfilled %d %s %s candles", written, symbol, timeframe)
    return written


def split_closed(
    pending: pd.DataFrame, update: pd.DataFrame, last_closed: pd.Timestamp | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge a websocket update into the forming candle(s); return (newly closed, still forming).

    ccxt.pro only sends new updates, so a candle counts as closed once a newer one appears.
    """
    merged = pd.concat([pending, update]).drop_duplicates("timestamp", keep="last")
    live = merged["timestamp"].max()
    closed = merged[merged["timestamp"] < live]
    if last_closed is not None:
        closed = closed[closed["timestamp"] > last_closed]
    return closed, merged[merged["timestamp"] == live]


async def ingest_live(
    gw: ExchangeGateway,
    sessions: async_sessionmaker[AsyncSession],
    symbol: str,
    timeframe: str,
    on_close: OnClose,
) -> None:
    """Persist each candle as it closes and hand the closed batch to `on_close`."""
    async with sessions() as session:
        latest = await latest_timestamp(session, symbol, timeframe)
    last_closed = pd.Timestamp(latest) if latest else None
    pending = to_frame([], symbol, timeframe)
    while True:
        try:
            ohlcv = await gw.watch_ohlcv(symbol, timeframe)
        except Exception:  # ccxt.pro reconnects on the next watch call
            log.exception("ohlcv stream error for %s %s, retrying", symbol, timeframe)
            await asyncio.sleep(1)
            continue
        # ponytail: closed candle = last WS version seen; Binance sends the final (x=true) kline
        # before the next one opens. Re-fetch via REST here if exact finals ever mismatch.
        closed, pending = split_closed(pending, to_frame(ohlcv, symbol, timeframe), last_closed)
        if closed.empty:
            continue
        async with sessions() as session:
            await upsert_candles(session, closed)
            await session.commit()
        last_closed = closed["timestamp"].iloc[-1]
        await on_close(symbol, timeframe, closed)
