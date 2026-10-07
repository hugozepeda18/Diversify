import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

import ccxt
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.risk import RiskManager, RiskPlan
from src.data.candles import load_candles
from src.models import ActivePosition, TradeSignal
from src.strategies import BaseStrategy

log = logging.getLogger(__name__)


def build_payload(
    *,
    timestamp: datetime,
    strategy: str,
    symbol: str,
    timeframe: str,
    action: str,
    price: float,
    plan: RiskPlan,
    indicators: dict[str, float],
    reason: str | None = None,
) -> dict[str, Any]:
    """The TECH_DESIGN §5 signal contract (plus `reason` on exits)."""
    payload: dict[str, Any] = {
        "event": "TRADE_SIGNAL",
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "strategy": strategy,
        "symbol": symbol,
        "timeframe": timeframe,
        "action": action,
        "trigger_price": round(price, 8),
        "risk_management": {
            "stop_loss": round(plan.stop_loss, 8),
            "take_profit": round(plan.take_profit, 8),
            "risk_reward_ratio": round(plan.risk_reward_ratio, 2),
            "position_size_usd": round(plan.position_size_usd, 2),
        },
        "indicators_snapshot": {k: round(v, 4) for k, v in indicators.items()},
    }
    if reason:
        payload["reason"] = reason
    return payload


async def get_position(
    session: AsyncSession, strategy: str, symbol: str, timeframe: str
) -> ActivePosition | None:
    return await session.scalar(
        select(ActivePosition).where(
            ActivePosition.strategy_name == strategy,
            ActivePosition.symbol == symbol,
            ActivePosition.timeframe == timeframe,
        )
    )


def _plan(p: ActivePosition) -> RiskPlan:
    entry, sl, tp = float(p.entry_price), float(p.stop_loss), float(p.take_profit)
    return RiskPlan(sl, tp, (tp - entry) / (entry - sl), float(p.position_size_usd))


async def _record(
    session: AsyncSession,
    *,
    decided_at: datetime,
    strategy: str,
    symbol: str,
    timeframe: str,
    action: str,
    fill: float,
    plan: RiskPlan,
    snapshot: dict[str, float],
    reason: str | None = None,
) -> dict[str, Any]:
    payload = build_payload(
        timestamp=decided_at,
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        action=action,
        price=fill,
        plan=plan,
        indicators=snapshot,
        reason=reason,
    )
    session.add(
        TradeSignal(
            timestamp=decided_at,
            strategy_name=strategy,
            symbol=symbol,
            action=action,
            price=fill,
            payload=payload,
        )
    )
    await session.commit()
    return payload


async def exit_on_price(
    session: AsyncSession, strategy: str, symbol: str, timeframe: str, price: float
) -> dict[str, Any] | None:
    """Live stop check: close the position at the *actual* price once it crosses SL or TP.

    Filling at the tick (not the stop level) records the real slippage of a gap through the stop.
    """
    position = await get_position(session, strategy, symbol, timeframe)
    if position is None:
        return None
    plan = _plan(position)
    if price <= plan.stop_loss:
        reason = "stop_loss"
    elif price >= plan.take_profit:
        reason = "take_profit"
    else:
        return None
    await session.delete(position)
    return await _record(
        session,
        decided_at=datetime.now(UTC),
        strategy=strategy,
        symbol=symbol,
        timeframe=timeframe,
        action="EXIT_LONG",
        fill=price,
        plan=plan,
        snapshot={},
        reason=reason,
    )


async def process_close(
    session: AsyncSession,
    strategy: BaseStrategy,
    risk: RiskManager,
    symbol: str,
    timeframe: str,
    candles: pd.DataFrame,
    price: float,
    equity: float,
    max_size_usd: float | None = None,
) -> dict[str, Any] | None:
    """Act on the newest closed candle (last row of `candles`); persist and return any signal.

    Mirrors the backtest: an open position exits on its SL/TP (checked on the candle's
    low/high, a fallback for ticks missed while offline) or the strategy's exit; otherwise an
    entry signal opens one at `price` (live price ~ next bar's open), capped at `max_size_usd`.
    State lives in the DB, so a restart resumes cleanly.
    """
    last = candles.iloc[-1]
    decided_at = candles.index[-1] + pd.Timedelta(seconds=ccxt.Exchange.parse_timeframe(timeframe))
    signals = strategy.evaluate(candles, lag=0)
    row = strategy.frame(candles).iloc[-1].drop("close", errors="ignore")
    snapshot = {str(k): float(v) for k, v in row.items()}
    position = await get_position(session, strategy.name, symbol, timeframe)
    if position is not None:
        plan = _plan(position)
        # ponytail: SL checked before TP when one candle spans both (pessimistic, like vbt).
        if last["low"] <= plan.stop_loss:
            reason, fill = "stop_loss", plan.stop_loss
        elif last["high"] >= plan.take_profit:
            reason, fill = "take_profit", plan.take_profit
        elif signals.exits.iloc[-1]:
            reason, fill = "signal", price
        else:
            return None
        action = "EXIT_LONG"
        await session.delete(position)
    elif signals.entries.iloc[-1]:
        reason, fill, action = None, price, "ENTER_LONG"
        plan = risk.plan(price, float(risk.atr(candles).iloc[-1]), equity)
        if max_size_usd is not None:
            if max_size_usd <= 0:
                log.warning("skip %s entry: open positions already use all equity", symbol)
                return None
            plan = plan._replace(position_size_usd=min(plan.position_size_usd, max_size_usd))
        session.add(
            ActivePosition(
                strategy_name=strategy.name,
                symbol=symbol,
                timeframe=timeframe,
                side="long",
                entry_price=price,
                stop_loss=plan.stop_loss,
                take_profit=plan.take_profit,
                position_size_usd=plan.position_size_usd,
            )
        )
    else:
        return None
    return await _record(
        session,
        decided_at=decided_at,
        strategy=strategy.name,
        symbol=symbol,
        timeframe=timeframe,
        action=action,
        fill=fill,
        plan=plan,
        snapshot=snapshot,
        reason=reason,
    )


class SignalEngine:
    """Live state for one strategy + timeframe across many symbols.

    Candle closes drive entries/exits; every price tick checks SL/TP. `book` mirrors open
    positions (symbol -> (stop_loss, take_profit, size_usd)) so ticks skip the DB, and caps
    total open size at equity (spot: no leverage across coins either).
    """

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        strategy: BaseStrategy,
        risk: RiskManager,
        timeframe: str,
        equity: float,
        days: int,
    ) -> None:
        self.sessions, self.strategy, self.risk = sessions, strategy, risk
        self.timeframe, self.equity, self.days = timeframe, equity, days
        self.book: dict[str, tuple[float, float, float]] = {}
        self.lock = asyncio.Lock()  # ponytail: one lock for all symbols; fine at ~1 tick/s/coin

    async def load(self) -> None:
        """Crash recovery: rebuild the book from active_positions."""
        async with self.sessions() as session:
            rows = await session.scalars(
                select(ActivePosition).where(
                    ActivePosition.strategy_name == self.strategy.name,
                    ActivePosition.timeframe == self.timeframe,
                )
            )
            for p in rows:
                plan = _plan(p)
                self.book[p.symbol] = (plan.stop_loss, plan.take_profit, plan.position_size_usd)

    def _sync(self, symbol: str, payload: dict[str, Any] | None) -> None:
        if payload is None:
            return
        log.info("SIGNAL %s", json.dumps(payload))
        # ponytail: alert dispatch (Telegram/webhook) skipped by request; plug it in here.
        if payload["action"] == "ENTER_LONG":
            rm = payload["risk_management"]
            self.book[symbol] = (rm["stop_loss"], rm["take_profit"], rm["position_size_usd"])
        else:
            self.book.pop(symbol, None)

    async def on_tick(self, symbol: str, price: float) -> None:
        stops = self.book.get(symbol)
        if stops is None or stops[0] < price < stops[1]:
            return  # hot path: no position, or price inside the SL/TP band
        async with self.lock, self.sessions() as session:
            payload = await exit_on_price(
                session, self.strategy.name, symbol, self.timeframe, price
            )
            self._sync(symbol, payload)

    async def on_close(
        self, symbol: str, timeframe: str, closed: pd.DataFrame, price: float | None
    ) -> None:
        log.info("%s %s closed @ %s", symbol, timeframe, closed["close"].iloc[-1])
        async with self.lock, self.sessions() as session:
            candles = await load_candles(session, symbol, timeframe, self.days)
            in_use = sum(size for s, (_, _, size) in self.book.items() if s != symbol)
            payload = await process_close(
                session,
                self.strategy,
                self.risk,
                symbol,
                timeframe,
                candles,
                price if price is not None else float(candles["close"].iloc[-1]),
                self.equity,
                max_size_usd=self.equity - in_use,
            )
            self._sync(symbol, payload)
