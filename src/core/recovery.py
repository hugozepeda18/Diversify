from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import ActivePosition

PositionKey = tuple[str, str, str]  # (strategy_name, symbol, timeframe)


async def load_active_positions(session: AsyncSession) -> dict[PositionKey, ActivePosition]:
    """Rebuild in-memory position state from the DB on boot."""
    rows = (await session.scalars(select(ActivePosition))).all()
    return {(p.strategy_name, p.symbol, p.timeframe): p for p in rows}
