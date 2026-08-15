"""extend_observation_source_enum

Revision ID: 04f85f25de05
Revises: f16eb05e3297
Create Date: 2026-08-15 17:03:18.651904

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '04f85f25de05'
down_revision: Union[str, Sequence[str], None] = 'f16eb05e3297'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to include extended ObservationSource values."""
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        new_values = [
            "SENTINEL1_SAR",
            "SMARTPHONE_GRVI",
            "IOT_SENSOR",
            "WEATHER_STATION",
            "MANUAL_SCOUT",
        ]
        for val in new_values:
            op.execute(f"ALTER TYPE observation_source_enum ADD VALUE IF NOT EXISTS '{val}'")
    else:
        # SQLite stores Enum values as VARCHAR check constraints or plain text strings
        pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
