"""Mark checks and ledger decisions as release checks or scheduled monitor runs."""

import sqlalchemy as sa
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Every existing check and decision came from a release check.
    for table in ("checks", "decisions"):
        op.add_column(
            table,
            sa.Column("trigger", sa.String(16), server_default="release", nullable=False),
        )


def downgrade() -> None:
    for table in ("decisions", "checks"):
        op.drop_column(table, "trigger")
