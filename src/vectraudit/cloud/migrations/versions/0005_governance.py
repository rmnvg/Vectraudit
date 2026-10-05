"""Project access and workspace export controls; inherit existing access by default."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "workspaces",
        sa.Column("exports_enabled", sa.Boolean(), server_default=sa.true(), nullable=False),
    )
    op.add_column(
        "projects",
        sa.Column(
            "access_roles", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=True
        ),
    )


def downgrade() -> None:
    op.drop_column("projects", "access_roles")
    op.drop_column("workspaces", "exports_enabled")
