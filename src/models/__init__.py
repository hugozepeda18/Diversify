import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import DateTime, Numeric, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

Price = Numeric(18, 8)


class Base(DeclarativeBase):
    pass


class MarketCandle(Base):
    """TimescaleDB hypertable partitioned on `timestamp` (see migration)."""

    __tablename__ = "market_candles"

    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    timeframe: Mapped[str] = mapped_column(String(10), primary_key=True)
    open: Mapped[Decimal] = mapped_column(Price)
    high: Mapped[Decimal] = mapped_column(Price)
    low: Mapped[Decimal] = mapped_column(Price)
    close: Mapped[Decimal] = mapped_column(Price)
    volume: Mapped[Decimal] = mapped_column(Numeric(30, 8))  # meme coins trade >1e10 units


class TradeSignal(Base):
    __tablename__ = "trade_signals"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    strategy_name: Mapped[str] = mapped_column(String(50))
    symbol: Mapped[str] = mapped_column(String(20))
    action: Mapped[str] = mapped_column(String(10))
    price: Mapped[Decimal] = mapped_column(Price)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)


class ActivePosition(Base):
    """Open paper trades; reloaded on boot for crash recovery."""

    __tablename__ = "active_positions"
    # One open position per strategy/symbol/timeframe: the DB rejects duplicate entries.
    __table_args__ = (UniqueConstraint("strategy_name", "symbol", "timeframe"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    strategy_name: Mapped[str] = mapped_column(String(50))
    symbol: Mapped[str] = mapped_column(String(20))
    timeframe: Mapped[str] = mapped_column(String(10))
    side: Mapped[str] = mapped_column(String(10))
    entry_price: Mapped[Decimal] = mapped_column(Price)
    stop_loss: Mapped[Decimal] = mapped_column(Price)
    take_profit: Mapped[Decimal] = mapped_column(Price)
    position_size_usd: Mapped[Decimal] = mapped_column(Price)
    # Set only when a broker executed the entry: coins held and the exchange-side OCO (SL/TP).
    quantity: Mapped[Decimal | None] = mapped_column(Numeric(30, 8))
    exchange_ref: Mapped[str | None] = mapped_column(String(64))
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
