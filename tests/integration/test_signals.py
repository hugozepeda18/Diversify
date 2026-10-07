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
