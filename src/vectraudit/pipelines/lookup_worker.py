"""Execute a locally registered document resolver with a private output protocol."""

from __future__ import annotations

import importlib
import json
import os
import sys

from vectraudit.pipelines.document_lookup import LookupRequest, LookupResult


def main() -> None:
    """Only validated passage responses reach the parent; discard adapter stdout/stderr."""
    with (
        os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as output,
        open(os.devnull, "w") as sink,
    ):
        os.dup2(sink.fileno(), sys.stdout.fileno())
        os.dup2(sink.fileno(), sys.stderr.fileno())
        try:
            payload = json.loads(sys.stdin.buffer.read(1024 * 1024 + 1))
            request = LookupRequest.model_validate(payload["request"])
            sys.path.insert(0, payload["root"])
            module, name = payload["target"].split(":")
            resolver = getattr(importlib.import_module(module), name)
            raw = resolver(request.model_copy(deep=True))
            result = LookupResult.model_validate(raw)
            output.write(result.model_dump_json())
            output.flush()
        except BaseException:
            # No exception strings, paths, arbitrary output or source content in errors.
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
