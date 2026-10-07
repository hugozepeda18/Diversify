from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.db import engine


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    # Each test runs inside an outer transaction that is rolled back; commits become savepoints.
    async with engine.connect() as conn:
        trans = await conn.begin()
        async with AsyncSession(bind=conn, join_transaction_mode="create_savepoint") as s:
            yield s
        await trans.rollback()
