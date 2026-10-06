"""Fresh-process adapter probe; its only output is a bounded diagnostic report."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from vectraudit.pipelines.diagnostic_report import DiagnosticReport, failure_report, inspect_result
from vectraudit.pipelines.errors import PipelineCallError
from vectraudit.pipelines.lexical import hybrid_pipeline
from vectraudit.pipelines.manifest import Adapter, open_adapter
from vectraudit.pipelines.models import PipelineRequest, PipelineResult
from vectraudit.pipelines.runtime import Pipeline


def probe(payload: dict[str, Any]) -> DiagnosticReport:
    """Open one trusted configuration and call it exactly once, without writing run artifacts."""
    request = PipelineRequest.model_validate(payload["request"])
    docs = payload["documents"]
    pipeline: Pipeline
    try:
        if payload["builtin"]:
            pipeline = hybrid_pipeline(docs)
        else:
            adapter = Adapter.model_validate(payload["adapter"])
            if (
                adapter.kind == "http"
                and adapter.token_env
                and not os.environ.get(adapter.token_env)
            ):
                return failure_report("missing_env")
            pipeline = open_adapter(adapter, Path(payload["root"]))
    except (ImportError, ModuleNotFoundError):
        return failure_report("dependency_missing")
    except Exception:
        return failure_report("adapter_configuration")
    started = time.perf_counter()
    try:
        result = pipeline(request.model_copy(deep=True))
        if not isinstance(result, PipelineResult):
            return failure_report("invalid_response", executed=True)
        result = PipelineResult.model_validate(result.model_dump())
        if len(result.hits) > request.k:
            return failure_report("too_many_hits", executed=True)
    except PipelineCallError as exc:
        return failure_report(exc.code, executed=True)
    except ValidationError:
        return failure_report("invalid_response", executed=True)
    except TimeoutError:
        return failure_report("timeout", executed=True)
    except ImportError:
        return failure_report("dependency_missing", executed=True)
    except Exception:
        return failure_report("adapter_exception", executed=True)
    return inspect_result(
        request,
        result,
        docs,
        latency_ms=(time.perf_counter() - started) * 1000,
        corpus_version_known=payload["corpus_version_known"],
        forbidden=payload["forbidden"],
        no_answer=payload["no_answer"],
    )


def main() -> None:
    # Redirect OS file descriptors too: native libraries and adapters may print secrets.
    # Keep a private duplicate of the protocol pipe for the final sanitized report only.
    with (
        os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as output,
        open(os.devnull, "w") as sink,
    ):
        os.dup2(sink.fileno(), sys.stdout.fileno())
        os.dup2(sink.fileno(), sys.stderr.fileno())
        try:
            payload = json.loads(sys.stdin.buffer.read(6 * 1024 * 1024))
            report = probe(payload)
        except BaseException:
            report = failure_report("worker_failed")
        output.write(report.model_dump_json())
        output.flush()


if __name__ == "__main__":
    main()
