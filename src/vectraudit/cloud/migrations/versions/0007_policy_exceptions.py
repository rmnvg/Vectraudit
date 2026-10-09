"""Audited, expiring exceptions for named failing policy rules."""

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "policy_exceptions",
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
        sa.Column("rule", sa.String(1200), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("owner", sa.String(200), nullable=False),
        sa.Column("origin_check_id", sa.String(32), nullable=False),
        sa.Column("created_by", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("max_checks", sa.Integer(), nullable=True),
        sa.Column("uses", sa.Integer(), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_by", sa.String(64), nullable=True),
    )
    op.create_index(
        "ix_policy_exceptions_project_rule", "policy_exceptions", ["project_id", "rule"]
    )


def downgrade() -> None:
    op.drop_index("ix_policy_exceptions_project_rule", table_name="policy_exceptions")
    op.drop_table("policy_exceptions")
