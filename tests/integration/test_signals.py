import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.risk import RiskManager
from src.models import ActivePosition, TradeSignal
from src.signals import process_close
from src.strategies import DoubleEmaCross
from tests.unit.test_strategies import V

SYM, TF = "TEST/USDT", "1h"
STRAT, RISK = DoubleEmaCross(fast=5, slow=20), RiskManager()


async def _run(session: AsyncSession, candles: pd.DataFrame) -> dict | None:  # type: ignore[type-arg]
    price = float(candles["close"].iloc[-1])
    return await process_close(session, STRAT, RISK, SYM, TF, candles, price, 10_000.0)


async def test_entry_then_stop_loss_round_trip(session: AsyncSession) -> None:
    k = int(STRAT.evaluate(V, lag=0).entries.to_numpy().argmax())  # first live entry candle
    assert await _run(session, V.iloc[:k]) is None  # nothing before the cross

    enter = await _run(session, V.iloc[: k + 1])
    assert enter is not None and enter["action"] == "ENTER_LONG"
    rm = enter["risk_management"]
    assert rm["stop_loss"] < enter["trigger_price"] < rm["take_profit"]
    assert rm["risk_reward_ratio"] == RISK.reward_ratio
    assert set(enter["indicators_snapshot"]) == {"ema_fast", "ema_slow"}
    assert enter["timestamp"] == (V.index[k] + pd.Timedelta("1h")).strftime("%Y-%m-%dT%H:%M:%SZ")
    pos = await session.scalar(select(ActivePosition).where(ActivePosition.symbol == SYM))
    assert pos is not None and float(pos.stop_loss) == rm["stop_loss"]

    assert await _run(session, V.iloc[: k + 1]) is None  # already long: no duplicate entry

    crash = V.iloc[[k]].copy()
    crash.index += pd.Timedelta("1h")
    crash["low"] = rm["stop_loss"] - 1
    exit_ = await _run(session, pd.concat([V.iloc[: k + 1], crash]))
    assert exit_ is not None and exit_["action"] == "EXIT_LONG" and exit_["reason"] == "stop_loss"
    assert exit_["trigger_price"] == rm["stop_loss"]
    assert await session.scalar(select(ActivePosition).where(ActivePosition.symbol == SYM)) is None
    actions = (
        await session.scalars(select(TradeSignal.action).where(TradeSignal.symbol == SYM))
    ).all()
    assert sorted(actions) == ["ENTER_LONG", "EXIT_LONG"]


async def _enter(session: AsyncSession, **kw: float) -> dict:  # type: ignore[type-arg]
    k = int(STRAT.evaluate(V, lag=0).entries.to_numpy().argmax())
    c = V.iloc[: k + 1]
    price = float(c["close"].iloc[-1])
    out = await process_close(session, STRAT, RISK, SYM, TF, c, price, 10_000.0, **kw)
    assert out is not None
    return out


async def test_tick_exit_fills_at_actual_price_not_stop_level(session: AsyncSession) -> None:
    from src.signals import exit_on_price

    rm = (await _enter(session))["risk_management"]
    inside = (rm["stop_loss"] + rm["take_profit"]) / 2
    assert await exit_on_price(session, STRAT.name, SYM, TF, inside) is None
    gap = rm["stop_loss"] * 0.97  # price gapped 3% through the stop
    out = await exit_on_price(session, STRAT.name, SYM, TF, gap)
    assert out is not None and out["reason"] == "stop_loss"
    assert out["trigger_price"] == round(gap, 8)  # real slippage recorded
    assert await exit_on_price(session, STRAT.name, SYM, TF, gap) is None  # already flat


async def test_entry_capped_by_free_equity(session: AsyncSession) -> None:
    assert (await _enter(session, max_size_usd=123.0))["risk_management"][
        "position_size_usd"
    ] == 123.0


async def test_entry_skipped_when_equity_fully_used(session: AsyncSession) -> None:
    k = int(STRAT.evaluate(V, lag=0).entries.to_numpy().argmax())
    c = V.iloc[: k + 1]
    out = await process_close(session, STRAT, RISK, SYM, TF, c, 100.0, 10_000.0, max_size_usd=0)
    assert out is None
    assert await session.scalar(select(ActivePosition).where(ActivePosition.symbol == SYM)) is None


async def test_engine_recovers_book_and_exits_on_tick(session: AsyncSession) -> None:
    from contextlib import asynccontextmanager
    from typing import Any

    from src.signals import SignalEngine

    rm = (await _enter(session))["risk_management"]

    @asynccontextmanager
    async def sessions() -> Any:
        yield session

    eng = SignalEngine(sessions, STRAT, RISK, TF, 10_000.0, days=30)  # type: ignore[arg-type]
    await eng.load()  # simulated restart: book rebuilt from active_positions
    assert eng.book[SYM][:2] == (rm["stop_loss"], rm["take_profit"])
    await eng.on_tick(SYM, rm["take_profit"] + 1)
    assert SYM not in eng.book
    last = await session.scalar(
        select(TradeSignal).where(TradeSignal.symbol == SYM).order_by(TradeSignal.timestamp.desc())
    )
    assert last is not None and last.payload["reason"] == "take_profit"
