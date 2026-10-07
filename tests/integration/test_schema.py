from collections.abc import AsyncIterator
from decimal import Decimal

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.db import engine
from src.core.recovery import load_active_positions
from src.models import ActivePosition


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    # Each test runs inside a transaction that is rolled back.
    async with engine.connect() as conn:
        trans = await conn.begin()
        async with AsyncSession(bind=conn, join_transaction_mode="create_savepoint") as s:
            yield s
        await trans.rollback()


def _position(**kw: str) -> ActivePosition:
    d = Decimal("1")
    return ActivePosition(
        strategy_name=kw.get("strategy_name", "DoubleEmaCross"),
        symbol="BTC/USDT",
        timeframe="1h",
        side="LONG",
        entry_price=Decimal("64250.5"),
        stop_loss=d,
        take_profit=d,
        position_size_usd=d,
    )


async def test_market_candles_is_hypertable(session: AsyncSession) -> None:
    names = await session.scalars(
        text("SELECT hypertable_name FROM timescaledb_information.hypertables")
    )
    assert "market_candles" in names.all()


async def test_recovery_rebuilds_positions(session: AsyncSession) -> None:
    session.add_all([_position(), _position(strategy_name="RsiThreshold")])
    await session.flush()
    state = await load_active_positions(session)
    assert set(state) == {
        ("DoubleEmaCross", "BTC/USDT", "1h"),
        ("RsiThreshold", "BTC/USDT", "1h"),
    }
    assert state[("DoubleEmaCross", "BTC/USDT", "1h")].entry_price == Decimal("64250.5")


async def test_duplicate_position_rejected(session: AsyncSession) -> None:
    session.add_all([_position(), _position()])
    with pytest.raises(IntegrityError):
        await session.flush()
