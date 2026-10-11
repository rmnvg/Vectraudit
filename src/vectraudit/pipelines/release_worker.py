"""One side of a paired release run: serves its checkout's pipeline over stdin/stdout.

``vectraudit release run-paired`` starts one worker per side, because two checkouts can
define the same Python module and one process can import only one of them. Each worker
opens its own release pipeline with its own environment, then answers one JSON request per
line with one JSON response per line:

- first line: ``{"ready": true, "config": {...}}`` or ``{"ready": false, "error": "..."}``;
- per request (a ``PipelineRequest``): ``{"result": {...}}`` (a ``PipelineResult``) or
  ``{"error": {"code": ..., "retry_after": ...}}``.

Failed calls carry only their stable code, never the adapter's message, as run files do.
Anything the adapter prints goes to stderr so it cannot corrupt the protocol.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, TextIO


def _send(stream: TextIO, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
    stream.flush()


def main(argv: list[str] | None = None) -> int:
    from vectraudit.pipelines.errors import PipelineCallError
    from vectraudit.pipelines.models import PipelineRequest
    from vectraudit.pipelines.release import ReleaseError, load_release, open_release_pipeline

    args = sys.argv[1:] if argv is None else argv
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    sys.stdout = sys.stderr  # adapter prints must not reach the protocol stream
    try:
        release_path = Path(args[0])
        release = load_release(release_path)
        pipeline, config = open_release_pipeline(release.pipeline, release_path.parent)
    except (ReleaseError, IndexError, OSError, ValueError) as exc:
        message = str(exc) if isinstance(exc, ReleaseError) else "Cannot open the release."
        _send(protocol, {"ready": False, "error": message})
        return 1
    _send(protocol, {"ready": True, "config": config})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = PipelineRequest.model_validate_json(line)
            raw = pipeline(request)
            _send(protocol, {"result": raw.model_dump(mode="json")})
        except PipelineCallError as exc:
            _send(protocol, {"error": {"code": exc.code, "retry_after": exc.retry_after}})
        except Exception:
            _send(protocol, {"error": {"code": "pipeline_error", "retry_after": None}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
