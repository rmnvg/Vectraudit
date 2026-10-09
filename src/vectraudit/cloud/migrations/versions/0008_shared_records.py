"""Project-scoped workbench revisions independent of raw-run retention."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "shared_records",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column(
            "workspace_id",
            sa.String(32),
            sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            sa.String(32),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("source_id", sa.String(32), nullable=False),
        sa.Column("revision", sa.String(32), nullable=False),
        sa.Column("query_text", sa.Boolean(), nullable=False),
        sa.Column("digest", sa.String(64), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("uploaded_by", sa.String(64), nullable=True),
        sa.Column("data", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False),
        sa.UniqueConstraint("project_id", "kind", "source_id", "revision", "query_text"),
    )
    op.create_index(
        "ix_shared_records_project_uploaded", "shared_records", ["project_id", "uploaded_at", "id"]
    )


def downgrade() -> None:
    op.drop_index("ix_shared_records_project_uploaded", table_name="shared_records")
    op.drop_table("shared_records")
