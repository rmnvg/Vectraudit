"""PostgreSQL + pgvector connector (``vectraudit[pgvector]``), strictly read-only.

Connection fields::

    {"backend": "pgvector", "dsn_env": "PG_DSN", "table": "public.chunks",
     "id_column": "id", "vector_column": "embedding", "distance": "cosine",
     "filter_columns": ["lang", "tenant_id"], "statement_timeout_ms": 10000,
     "params": {"hnsw.ef_search": 100}}

Every session is ``default_transaction_read_only`` with a ``statement_timeout``.
Identifiers are validated and quoted, values are bound parameters, and tuning parameters
are validated integers set with ``SET LOCAL`` inside each query's transaction. The exact
reference disables index scans for that transaction, so PostgreSQL sorts every row.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from vectraudit.connectors.base import (
    BackendInfo,
    Capabilities,
    Connector,
    ConnectorError,
    Distance,
    ErrorCode,
    FilterValue,
    SearchRequest,
)

INSTALL_HINT = 'Install the PostgreSQL driver: `pip install "vectraudit[pgvector]"`.'
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
OPERATORS = {Distance.L2: "<->", Distance.COSINE: "<=>", Distance.IP: "<#>"}
OPCLASS_SUFFIX = {Distance.L2: "l2_ops", Distance.COSINE: "cosine_ops", Distance.IP: "ip_ops"}
TUNING_PARAMS = ("hnsw.ef_search", "ivfflat.probes")
VECTOR_TYPES = ("vector", "halfvec")


def check_identifier(name: str, what: str) -> str:
    """A plain SQL identifier (letters, digits, underscore), or ConnectorError."""
    if not isinstance(name, str) or not IDENTIFIER.fullmatch(name):
        raise ConnectorError(
            ErrorCode.CONFIG,
            f"{what} {name!r} is not a plain SQL identifier.",
            "Use letters, digits and underscores (quote-free names); schema.table for tables.",
        )
    return name


def vector_literal(values: Iterable[float]) -> str:
    """pgvector's text form, e.g. ``[0.5,1.0]`` (values are finite by construction)."""
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


class PgvectorConnector(Connector):
    """Search one table's vector column with pgvector."""

    name = "pgvector"

    def __init__(
        self,
        dsn: str,
        *,
        table: str,
        id_column: str = "id",
        vector_column: str = "embedding",
        distance: str = "cosine",
        filter_columns: Sequence[str] = (),
        statement_timeout_ms: int = 10_000,
        connect_timeout_s: int = 10,
    ) -> None:
        try:
            import psycopg
        except ImportError as e:
            raise ConnectorError(
                ErrorCode.DEPENDENCY, "psycopg is not installed.", INSTALL_HINT
            ) from e
        parts = table.split(".") if isinstance(table, str) else [table]
        if len(parts) > 2:
            raise ConnectorError(
                ErrorCode.CONFIG, f"table {table!r} must be table or schema.table."
            )
        self.table_parts = [check_identifier(p, "table") for p in parts]
        self.id_column = check_identifier(id_column, "id_column")
        self.vector_column = check_identifier(vector_column, "vector_column")
        self.filter_columns = frozenset(
            check_identifier(c, "filter column") for c in filter_columns
        )
        try:
            self.distance = Distance(distance)
        except ValueError as e:
            raise ConnectorError(
                ErrorCode.CONFIG,
                f"distance {distance!r} is not supported.",
                "Use cosine, l2 or ip (the operator class your index was built with).",
            ) from e
        self.table = ".".join(self.table_parts)
        self.statement_timeout_ms = statement_timeout_ms
        self._conn: Any = None
        try:
            self._conn = psycopg.connect(
                dsn,
                connect_timeout=connect_timeout_s,
                autocommit=False,
                options=(
                    "-c default_transaction_read_only=on "
                    f"-c statement_timeout={int(statement_timeout_ms)}"
                ),
            )
            self._conn.read_only = True
            self._column_type, self._dim = self._inspect_column()
        except psycopg.Error as e:
            self.close()
            raise _pg_error(e, self.table) from e
        except BaseException:
            self.close()
            raise

    # --- SQL helpers ----------------------------------------------------------------

    def _relation(self) -> Any:
        from psycopg import sql

        return sql.SQL(".").join(sql.Identifier(p) for p in self.table_parts)

    def _fetch(self, query: Any, params: Sequence[Any] = ()) -> list[tuple[Any, ...]]:
        import psycopg

        try:
            with self._conn.transaction(), self._conn.cursor() as cur:
                cur.execute(query, params)
                rows: list[tuple[Any, ...]] = cur.fetchall()
                return rows
        except psycopg.Error as e:
            raise _pg_error(e, self.table) from e

    def _inspect_column(self) -> tuple[str, int]:
        rows = self._fetch(
            "SELECT t.typname, a.atttypmod FROM pg_attribute a "
            "JOIN pg_type t ON t.oid = a.atttypid "
            "WHERE a.attrelid = %s::regclass AND a.attname = %s AND NOT a.attisdropped",
            (self.table, self.vector_column),
        )
        if not rows:
            raise ConnectorError(
                ErrorCode.CONFIG, f"Column {self.vector_column!r} not found in {self.table}."
            )
        type_name, typmod = rows[0]
        # Catalog reads need no table privilege; check now so a run fails before searching.
        allowed = self._fetch("SELECT has_table_privilege(%s, 'SELECT')", (self.table,))
        if not allowed[0][0]:
            raise ConnectorError(
                ErrorCode.AUTH,
                f"The database role cannot SELECT from {self.table}.",
                "Grant SELECT on the table (and USAGE on its schema) to the evaluation role.",
            )
        if type_name not in VECTOR_TYPES:
            raise ConnectorError(
                ErrorCode.UNSUPPORTED,
                f"{self.table}.{self.vector_column} has type {type_name}; "
                f"supported: {', '.join(VECTOR_TYPES)}.",
            )
        dim = int(typmod)
        if dim <= 0:  # unconstrained vector column: read one row's dimension
            from psycopg import sql

            sample = self._fetch(
                sql.SQL(
                    "SELECT vector_dims({col}) FROM {rel} WHERE {col} IS NOT NULL LIMIT 1"
                ).format(col=sql.Identifier(self.vector_column), rel=self._relation())
            )
            if not sample:
                raise ConnectorError(ErrorCode.CONFIG, f"{self.table} has no vectors.")
            dim = int(sample[0][0])
        return str(type_name), dim

    # --- contract -------------------------------------------------------------------

    def capabilities(self) -> Capabilities:
        return Capabilities(
            filters=bool(self.filter_columns),
            exact_reference=True,
            config_inspection=True,
            query_plans=True,
            tuning_params=TUNING_PARAMS,
        )

    def describe(self) -> BackendInfo:
        server = self._fetch("SHOW server_version")[0][0]
        ext = self._fetch("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        indexes = self._fetch(
            "SELECT i.relname, pg_get_indexdef(i.oid), am.amname "
            "FROM pg_index x JOIN pg_class i ON i.oid = x.indexrelid "
            "JOIN pg_am am ON am.oid = i.relam "
            "JOIN pg_attribute a ON a.attrelid = x.indrelid AND a.attnum = ANY(x.indkey) "
            "WHERE x.indrelid = %s::regclass AND a.attname = %s ORDER BY i.relname",
            (self.table, self.vector_column),
        )
        estimate = self._fetch(
            "SELECT reltuples::bigint FROM pg_class WHERE oid = %s::regclass", (self.table,)
        )
        reltuples = int(estimate[0][0]) if estimate else -1
        suffix = OPCLASS_SUFFIX[self.distance]
        vector_indexes = [
            {"name": name, "method": method, "definition": definition}
            for name, definition, method in indexes
            if method in ("hnsw", "ivfflat")
        ]
        return BackendInfo(
            backend="pgvector",
            distance=self.distance,
            dim=self._dim,
            count=reltuples if reltuples >= 0 else None,
            target={
                "table": self.table,
                "id_column": self.id_column,
                "vector_column": self.vector_column,
            },
            backend_version=f"PostgreSQL {server}; pgvector {ext[0][0] if ext else 'missing'}",
            client_version=_client_version(),
            config={
                "column_type": self._column_type,
                "indexes": vector_indexes,
                "matching_index": any(suffix in ix["definition"] for ix in vector_indexes),
                "count_is_estimate": True,
                "statement_timeout_ms": self.statement_timeout_ms,
                "filter_columns": sorted(self.filter_columns),
            },
            snapshot=None,
        )

    def _query(self, request: SearchRequest, *, explain: bool = False) -> tuple[Any, list[Any]]:
        from psycopg import sql

        unknown = set(request.filter) - self.filter_columns
        if unknown:
            raise ConnectorError(
                ErrorCode.UNSUPPORTED,
                f"Filter field(s) {', '.join(sorted(unknown))} are not filter_columns of this "
                "connection.",
                "Add them to filter_columns (plain column names) in the connection file.",
            )
        conditions = [
            sql.SQL("{} = %s").format(sql.Identifier(col)) for col in sorted(request.filter)
        ]
        where = (
            sql.SQL(" WHERE ") + sql.SQL(" AND ").join(conditions) if conditions else sql.SQL("")
        )
        query = sql.SQL(
            "{explain}SELECT {id}::text FROM {rel}{where} ORDER BY {vec} {op} %s::{vtype} LIMIT %s"
        ).format(
            explain=sql.SQL("EXPLAIN (FORMAT JSON) " if explain else ""),
            id=sql.Identifier(self.id_column),
            rel=self._relation(),
            where=where,
            vec=sql.Identifier(self.vector_column),
            op=sql.SQL(OPERATORS[self.distance]),
            vtype=sql.SQL(self._column_type),
        )
        params: list[FilterValue] = [request.filter[c] for c in sorted(request.filter)]
        return query, [*params, vector_literal(request.vector), request.k]

    def _settings(self, request: SearchRequest) -> list[Any]:
        from psycopg import sql

        statements: list[Any] = []
        for name in sorted(request.params):
            value = request.params[name]
            if type(value) is not int or value < 1:
                raise ConnectorError(ErrorCode.CONFIG, f"{name} must be a positive integer.")
            # Names come from TUNING_PARAMS (checked by the base class); values are ints.
            statements.append(
                sql.SQL("SET LOCAL {} = {}").format(sql.SQL(name), sql.Literal(value))
            )
        if request.exact:
            # Index scans off and sequential scans explicitly on: a role or database default
            # of enable_seqscan = off would otherwise let the planner pick the ANN index.
            statements += [
                sql.SQL("SET LOCAL enable_seqscan = on"),
                sql.SQL("SET LOCAL enable_indexscan = off"),
                sql.SQL("SET LOCAL enable_bitmapscan = off"),
            ]
        return statements

    def _run(self, request: SearchRequest, *, explain: bool) -> list[tuple[Any, ...]]:
        import psycopg

        if len(request.vector) != self._dim:
            raise ConnectorError(
                ErrorCode.QUERY,
                f"Query has dimension {len(request.vector)}, {self.table} expects {self._dim}.",
            )
        query, params = self._query(request, explain=explain)
        settings = self._settings(request)
        try:
            with self._conn.transaction(), self._conn.cursor() as cur:
                for statement in settings:
                    cur.execute(statement)
                cur.execute(query, params)
                rows: list[tuple[Any, ...]] = cur.fetchall()
                return rows
        except psycopg.Error as e:
            raise _pg_error(e, self.table) from e

    def _search(self, request: SearchRequest) -> list[str]:
        return [str(row[0]) for row in self._run(request, explain=False)]

    def explain(self, request: SearchRequest) -> dict[str, Any]:
        """The planner's plan for a query, without executing it (EXPLAIN, no ANALYZE)."""
        self.check_request(request)
        plan = self._run(request, explain=True)[0][0][0]["Plan"]
        nodes: list[str] = []
        indexes: list[str] = []

        def walk(node: Mapping[str, Any]) -> None:
            nodes.append(str(node.get("Node Type")))
            if node.get("Index Name"):
                indexes.append(str(node["Index Name"]))
            for child in node.get("Plans", []):
                walk(child)

        walk(plan)
        return {"nodes": nodes, "indexes": indexes, "total_cost": plan.get("Total Cost")}

    def evidence(
        self, request: SearchRequest, found: list[str], exact: list[str] | None
    ) -> dict[str, Any] | None:
        """The planner's plan for this query (EXPLAIN without ANALYZE: nothing executes)."""
        return {"kind": "pg_plan", "source": "planner estimate (EXPLAIN)", **self.explain(request)}

    def close(self) -> None:
        conn, self._conn = getattr(self, "_conn", None), None
        if conn is not None:
            conn.close()


def _pg_error(e: Any, table: str) -> ConnectorError:
    """Classify a psycopg error by SQLSTATE."""
    import psycopg

    state = getattr(getattr(e, "diag", None), "sqlstate", None) or getattr(e, "sqlstate", None)
    message = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
    if state in ("28P01", "28000") or "password authentication failed" in message:
        return ConnectorError(
            ErrorCode.AUTH,
            f"PostgreSQL rejected the credentials: {message}",
            "Check the DSN in the variable named by dsn_env.",
        )
    if state == "42501":
        return ConnectorError(
            ErrorCode.AUTH,
            f"Permission denied on {table}: {message}",
            "Grant SELECT on the table to the evaluation role.",
        )
    if state == "57014":
        return ConnectorError(
            ErrorCode.TIMEOUT,
            f"The query exceeded statement_timeout: {message}",
            "Raise statement_timeout_ms or lower ef_search / probes.",
        )
    if state in ("42P01", "42703", "3F000", "42704"):
        return ConnectorError(ErrorCode.CONFIG, f"PostgreSQL: {message}")
    if state is None and isinstance(e, psycopg.OperationalError):
        return ConnectorError(
            ErrorCode.CONNECTION,
            f"Cannot reach PostgreSQL: {message}",
            "Check the host, port and network access from the runner.",
        )
    if state is not None and state.startswith("08"):
        return ConnectorError(ErrorCode.CONNECTION, f"PostgreSQL connection failed: {message}")
    return ConnectorError(ErrorCode.QUERY, f"PostgreSQL: {message}")


def _client_version() -> str | None:
    try:
        return f"psycopg {version('psycopg')}"
    except PackageNotFoundError:
        return None
