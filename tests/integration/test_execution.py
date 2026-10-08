from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.risk import RiskManager, RiskPlan
from src.execution import Execution
from src.models import ActivePosition, TradeSignal
from src.signals import SignalEngine, process_close
from src.strategies import DoubleEmaCross
from tests.unit.test_strategies import candles

SYM, TF = "TEST/USDT", "1h"
STRAT, RISK = DoubleEmaCross(fast=5, slow=20), RiskManager()
# down, up, down: one live entry on the way up, one exit signal on the way down
WAVE = candles(
    np.r_[np.linspace(200, 100, 60), np.linspace(100, 200, 60), np.linspace(200, 100, 60)]
)
LIVE = STRAT.evaluate(WAVE, lag=0)
K = int(LIVE.entries.to_numpy().argmax())
J = K + int(LIVE.exits.iloc[K:].to_numpy().argmax())


class FakeBroker:
    def __init__(self, fail: bool = False) -> None:
        self.fail, self.calls = fail, []  # type: ignore[var-annotated]
        self.status: tuple[str, float] | None = None

    async def free_usdt(self) -> float:
        return 5_000.0

    async def enter(self, symbol: str, plan: RiskPlan, signal_price: float) -> Execution:
        self.calls.append(("enter", symbol))
        if self.fail:
            raise RuntimeError("exchange down")
        price = signal_price * 1.001  # slippage
        return Execution(
            price,
            plan.position_size_usd / price,
            plan.position_size_usd,
            price - (signal_price - plan.stop_loss),
            price + (plan.take_profit - signal_price),
            "oco-1",
        )

    async def exit(self, symbol: str, qty: float, ref: str) -> float:
        self.calls.append(("exit", symbol, qty, ref))
        return 123.0

    async def check(self, symbol: str, ref: str) -> tuple[str, float] | None:
        return self.status

    async def place_oco(self, symbol: str, qty: float, sl: float, tp: float) -> str:
        self.calls.append(("place_oco", symbol))
        return "oco-2"


async def _close(session: AsyncSession, upto: int, broker: FakeBroker) -> dict[str, Any] | None:
    c = WAVE.iloc[: upto + 1]
    price = float(c["close"].iloc[-1])
    return await process_close(
        session,
        STRAT,
        RISK,
        SYM,
        TF,
        c,
        price,
        10_000.0,
        broker=broker,  # type: ignore[arg-type]
    )


async def _position(session: AsyncSession) -> ActivePosition | None:
    return await session.scalar(select(ActivePosition).where(ActivePosition.symbol == SYM))


async def test_entry_executes_and_stores_exchange_state(session: AsyncSession) -> None:
    out = await _close(session, K, FakeBroker())
    assert out is not None and out["action"] == "ENTER_LONG"
    ex = out["execution"]
    assert ex["ref"] == "oco-1" and ex["price"] > out["trigger_price"]  # real fill recorded
    pos = await _position(session)
    assert pos is not None and pos.exchange_ref == "oco-1"
    assert float(pos.entry_price) == pytest.approx(ex["price"])
    assert float(pos.quantity or 0) == pytest.approx(ex["quantity"])
    # stops re-anchored to the fill, same distance as planned
    assert float(pos.entry_price) - float(pos.stop_loss) == pytest.approx(
        out["trigger_price"] - (out["trigger_price"] - (ex["price"] - float(pos.stop_loss)))
    )
    assert out["risk_management"]["stop_loss"] == pytest.approx(ex["stop_loss"])


async def test_failed_order_records_nothing(session: AsyncSession) -> None:
    with pytest.raises(RuntimeError):
        await _close(session, K, FakeBroker(fail=True))
    assert await _position(session) is None
    count = select(func.count()).select_from(TradeSignal).where(TradeSignal.symbol == SYM)
    assert await session.scalar(count) == 0


async def test_exchange_position_ignores_candle_stops_but_sells_on_exit_signal(
    session: AsyncSession,
) -> None:
    broker = FakeBroker()
    await _close(session, K, broker)
    pos = await _position(session)
    assert pos is not None
    crash = WAVE.iloc[: K + 2].copy()
    crash.iloc[-1, crash.columns.get_loc("low")] = float(pos.stop_loss) - 10  # wick through SL
    assert (
        await process_close(
            session,
            STRAT,
            RISK,
            SYM,
            TF,
            crash,
            150.0,
            10_000.0,
            broker=broker,  # type: ignore[arg-type]
        )
        is None
    ), "exchange OCO owns the stop; the bot must not book a virtual exit"
    out = await _close(session, J, broker)
    assert out is not None and out["reason"] == "signal" and out["execution"]["price"] == 123.0
    assert broker.calls[-1] == ("exit", SYM, pytest.approx(float(pos.quantity or 0)), "oco-1")
    assert await _position(session) is None


def _engine(session: AsyncSession, broker: FakeBroker) -> SignalEngine:
    @asynccontextmanager
    async def sessions() -> AsyncIterator[AsyncSession]:
        yield session

    return SignalEngine(sessions, STRAT, RISK, TF, 10_000.0, 30, broker=broker)  # type: ignore[arg-type]


async def test_sync_records_exchange_stop_fill(session: AsyncSession) -> None:
    broker = FakeBroker()
    await _close(session, K, broker)
    eng = _engine(session, broker)
    await eng.load()
    await eng.sync_exchange()  # OCO still open
    assert await _position(session) is not None
    broker.status = ("stop_loss", 95.5)  # fired on the exchange (maybe while we were offline)
    await eng.sync_exchange()
    assert await _position(session) is None and SYM not in eng.book
    last = await session.scalar(
        select(TradeSignal).where(TradeSignal.action == "EXIT_LONG", TradeSignal.symbol == SYM)
    )
    assert last is not None and last.payload["reason"] == "stop_loss"
    assert last.payload["trigger_price"] == 95.5


async def test_sync_replaces_manually_cancelled_oco(session: AsyncSession) -> None:
    broker = FakeBroker()
    await _close(session, K, broker)
    broker.status = ("cancelled", 0.0)
    await _engine(session, broker).sync_exchange()
    pos = await _position(session)
    assert pos is not None and pos.exchange_ref == "oco-2"
    assert ("place_oco", SYM) in broker.calls


async def test_ticks_do_not_exit_when_exchange_holds_stops(session: AsyncSession) -> None:
    broker = FakeBroker()
    await _close(session, K, broker)
    eng = _engine(session, broker)
    await eng.load()
    await eng.on_tick(SYM, 0.01)  # far below SL
    assert await _position(session) is not None
