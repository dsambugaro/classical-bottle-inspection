"""drop analyses.adhoc_json (the ad hoc features were removed from the core package)

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-13 10:00:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Applies this revision."""
    with op.batch_alter_table("analyses", schema=None) as batch_op:
        batch_op.drop_column("adhoc_json")


def downgrade() -> None:
    """Reverts this revision."""
    with op.batch_alter_table("analyses", schema=None) as batch_op:
        batch_op.add_column(sa.Column("adhoc_json", sa.JSON(), nullable=False, server_default="{}"))
