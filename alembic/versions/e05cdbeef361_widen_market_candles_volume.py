"""widen market_candles volume

Revision ID: e05cdbeef361
Revises: d50891b86ed8
Create Date: 2026-10-07 14:36:16.882245

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e05cdbeef361"
down_revision: str | Sequence[str] | None = "d50891b86ed8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # NUMERIC(18,8) caps at 1e10; DOGE daily volume exceeds that.
    op.alter_column(
        "market_candles",
        "volume",
        type_=sa.Numeric(precision=30, scale=8),
        existing_type=sa.Numeric(precision=18, scale=8),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        "market_candles",
        "volume",
        type_=sa.Numeric(precision=18, scale=8),
        existing_type=sa.Numeric(precision=30, scale=8),
    )
