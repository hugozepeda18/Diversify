from datetime import datetime
from typing import Any

import ccxt
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.risk import RiskManager, RiskPlan
from src.models import ActivePosition, TradeSignal
from src.strategies import BaseStrategy


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


async def process_close(
    session: AsyncSession,
    strategy: BaseStrategy,
    risk: RiskManager,
    symbol: str,
    timeframe: str,
    candles: pd.DataFrame,
    price: float,
    equity: float,
) -> dict[str, Any] | None:
    """Act on the newest closed candle (last row of `candles`); persist and return any signal.

    Mirrors the backtest: an open position exits on its SL/TP (checked on the candle's
    low/high) or the strategy's exit; otherwise an entry signal opens one at `price`
    (the live price ~ next bar's open). State lives in the DB, so a restart resumes cleanly.
    """
    last = candles.iloc[-1]
    decided_at = candles.index[-1] + pd.Timedelta(seconds=ccxt.Exchange.parse_timeframe(timeframe))
    signals = strategy.evaluate(candles, lag=0)
    row = strategy.frame(candles).iloc[-1].drop("close", errors="ignore")
    snapshot = {str(k): float(v) for k, v in row.items()}
    position = await session.scalar(
        select(ActivePosition).where(
            ActivePosition.strategy_name == strategy.name,
            ActivePosition.symbol == symbol,
            ActivePosition.timeframe == timeframe,
        )
    )
    if position is not None:
        entry, sl, tp = (
            float(position.entry_price),
            float(position.stop_loss),
            float(position.take_profit),
        )
        # ponytail: SL checked before TP when one candle spans both (pessimistic, like vbt).
        if last["low"] <= sl:
            reason, fill = "stop_loss", sl
        elif last["high"] >= tp:
            reason, fill = "take_profit", tp
        elif signals.exits.iloc[-1]:
            reason, fill = "signal", price
        else:
            return None
        plan = RiskPlan(sl, tp, (tp - entry) / (entry - sl), float(position.position_size_usd))
        action = "EXIT_LONG"
        await session.delete(position)
    elif signals.entries.iloc[-1]:
        reason, fill, action = None, price, "ENTER_LONG"
        plan = risk.plan(price, float(risk.atr(candles).iloc[-1]), equity)
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
    payload = build_payload(
        timestamp=decided_at,
        strategy=strategy.name,
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
            strategy_name=strategy.name,
            symbol=symbol,
            action=action,
            price=fill,
            payload=payload,
        )
    )
    await session.commit()
    return payload
