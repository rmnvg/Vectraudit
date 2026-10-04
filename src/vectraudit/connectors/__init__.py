"""Vector backend connectors behind one contract (see :mod:`vectraudit.connectors.base`).

Like :mod:`vectraudit.core`, this package is a plain library: no web framework imports.
Backend client libraries are optional extras, imported only when a connector is opened.
"""

from vectraudit.connectors.base import (
    BackendInfo,
    Capabilities,
    Connector,
    ConnectorError,
    Distance,
    ErrorCode,
    SearchRequest,
)
from vectraudit.connectors.config import ConnectionConfig, load_connection, open_connector

__all__ = [
    "BackendInfo",
    "Capabilities",
    "ConnectionConfig",
    "Connector",
    "ConnectorError",
    "Distance",
    "ErrorCode",
    "SearchRequest",
    "load_connection",
    "open_connector",
]
