"""Alembic environment: the URL comes from the config built in :mod:`vectraudit.cloud.migrate`."""

from __future__ import annotations

from alembic import context
from sqlalchemy import Connection

from vectraudit.cloud.db import make_engine
from vectraudit.cloud.models import Base

config = context.config
target_metadata = Base.metadata


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=connection.dialect.name == "sqlite",
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


connection = config.attributes.get("connection")
if connection is not None:
    _run(connection)
else:
    engine = make_engine(config.get_main_option("sqlalchemy.url") or "")
    with engine.connect() as conn:
        _run(conn)
    engine.dispose()
