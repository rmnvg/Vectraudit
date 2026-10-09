"""Decision ledger: release decisions that outlive their runs and checks."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "decisions",
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
        sa.Column("check_id", sa.String(32), nullable=True),
        sa.Column("baseline_name", sa.String(100), nullable=True),
        sa.Column("verdict", sa.String(16), nullable=False),
        sa.Column("policy_digest", sa.String(64), nullable=False),
        sa.Column("dataset_sha256", sa.String(64), nullable=False),
        sa.Column("baseline_run_id", sa.String(64), nullable=False),
        sa.Column("baseline_digest", sa.String(64), nullable=False),
        sa.Column("candidate_run_id", sa.String(64), nullable=False),
        sa.Column("candidate_digest", sa.String(64), nullable=False),
        sa.Column("candidate_label", sa.String(200), nullable=True),
        sa.Column("candidate_commit", sa.String(64), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_by", sa.String(64), nullable=True),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "summary", sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False
        ),
        sa.UniqueConstraint("check_id"),
    )
    op.create_index("ix_decisions_project_decided", "decisions", ["project_id", "decided_at"])


def downgrade() -> None:
    op.drop_index("ix_decisions_project_decided", table_name="decisions")
    op.drop_table("decisions")
