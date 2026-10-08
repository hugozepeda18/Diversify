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
from src.execution import BinanceDemoBroker
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
    execution: dict[str, Any] | None = None,
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
    if execution is not None:
        payload["execution"] = execution  # what the exchange actually did
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
        return await close_position(session, position, price, "stop_loss")
    if price >= plan.take_profit:
        return await close_position(session, position, price, "take_profit")
    return None


async def close_position(
    session: AsyncSession,
    position: ActivePosition,
    fill: float,
    reason: str,
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Delete the position and record the EXIT_LONG signal (outside a candle close)."""
    plan = _plan(position)
    await session.delete(position)
    return await _record(
        session,
        decided_at=datetime.now(UTC),
        strategy=position.strategy_name,
        symbol=position.symbol,
        timeframe=position.timeframe,
        action="EXIT_LONG",
        fill=fill,
        plan=plan,
        snapshot={},
        reason=reason,
        execution=execution,
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
    broker: BinanceDemoBroker | None = None,
) -> dict[str, Any] | None:
    """Act on the newest closed candle (last row of `candles`); persist and return any signal.

    Mirrors the backtest: an open position exits on its SL/TP (checked on the candle's
    low/high, a fallback for ticks missed while offline) or the strategy's exit; otherwise an
    entry signal opens one at `price` (live price ~ next bar's open), capped at `max_size_usd`.
    State lives in the DB, so a restart resumes cleanly.

    With a `broker`, orders go to the exchange *before* the DB commit: if an order fails the
    exception propagates and nothing is recorded. Exchange-held positions skip the candle
    SL/TP check; their OCO fills are synced by `SignalEngine.sync_exchange`.
    """
    last = candles.iloc[-1]
    decided_at = candles.index[-1] + pd.Timedelta(seconds=ccxt.Exchange.parse_timeframe(timeframe))
    signals = strategy.evaluate(candles, lag=0)
    row = strategy.frame(candles).iloc[-1].drop("close", errors="ignore")
    snapshot = {str(k): float(v) for k, v in row.items()}
    position = await get_position(session, strategy.name, symbol, timeframe)
    execution: dict[str, Any] | None = None
    if position is not None:
        plan = _plan(position)
        on_exchange = position.exchange_ref is not None
        # ponytail: SL checked before TP when one candle spans both (pessimistic, like vbt).
        if not on_exchange and last["low"] <= plan.stop_loss:
            reason, fill = "stop_loss", plan.stop_loss
        elif not on_exchange and last["high"] >= plan.take_profit:
            reason, fill = "take_profit", plan.take_profit
        elif signals.exits.iloc[-1]:
            reason, fill = "signal", price
            if on_exchange:
                if broker is None:
                    raise RuntimeError(f"{symbol} position lives on the exchange; run --execute")
                assert position.quantity is not None and position.exchange_ref is not None
                sold = await broker.exit(symbol, float(position.quantity), position.exchange_ref)
                execution = {"price": sold, "quantity": float(position.quantity)}
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
        entry_price, quantity, ref = price, None, None
        if broker is not None:
            done = await broker.enter(symbol, plan, price)
            plan = plan._replace(
                stop_loss=done.stop_loss,
                take_profit=done.take_profit,
                position_size_usd=done.cost_usd,
            )
            entry_price, quantity, ref = done.price, done.quantity, done.ref
            execution = done._asdict()
        session.add(
            ActivePosition(
                strategy_name=strategy.name,
                symbol=symbol,
                timeframe=timeframe,
                side="long",
                entry_price=entry_price,
                stop_loss=plan.stop_loss,
                take_profit=plan.take_profit,
                position_size_usd=plan.position_size_usd,
                quantity=quantity,
                exchange_ref=ref,
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
        execution=execution,
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
        broker: BinanceDemoBroker | None = None,
    ) -> None:
        self.sessions, self.strategy, self.risk = sessions, strategy, risk
        self.timeframe, self.equity, self.days = timeframe, equity, days
        self.broker = broker
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
        if self.broker is not None:
            return  # the exchange holds the stops; sync_exchange records their fills
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
            equity, free = self.equity, self.equity - in_use
            if self.broker is not None:  # real account: size off the actual balance
                cash = await self.broker.free_usdt()
                equity, free = cash + in_use, cash * 0.99  # 1% buffer for price moves/fees
            payload = await process_close(
                session,
                self.strategy,
                self.risk,
                symbol,
                timeframe,
                candles,
                price if price is not None else float(candles["close"].iloc[-1]),
                equity,
                max_size_usd=free,
                broker=self.broker,
            )
            self._sync(symbol, payload)

    async def sync_exchange(self) -> None:
        """Record exchange-side OCO fills (incl. ones that fired while the bot was offline)."""
        assert self.broker is not None
        async with self.lock, self.sessions() as session:
            rows = await session.scalars(
                select(ActivePosition).where(
                    ActivePosition.strategy_name == self.strategy.name,
                    ActivePosition.timeframe == self.timeframe,
                    ActivePosition.exchange_ref.is_not(None),
                )
            )
            for p in list(rows):
                symbol, ref, qty = p.symbol, p.exchange_ref, p.quantity  # read before commits
                assert ref is not None and qty is not None
                result = await self.broker.check(symbol, ref)
                if result is None:
                    continue
                reason, fill = result
                if reason == "cancelled":  # OCO removed by hand: never leave coins unprotected
                    plan = _plan(p)
                    p.exchange_ref = await self.broker.place_oco(
                        symbol, float(qty), plan.stop_loss, plan.take_profit
                    )
                    log.warning("%s OCO was cancelled outside the bot; re-placing it", symbol)
                    await session.commit()
                    continue
                payload = await close_position(session, p, fill, reason, {"price": fill})
                self._sync(symbol, payload)

    async def run_exchange_sync(self, every: float = 30.0) -> None:
        while True:
            try:
                await self.sync_exchange()
            except Exception:  # transient API errors: retry next round
                log.exception("exchange sync failed")
            await asyncio.sleep(every)
