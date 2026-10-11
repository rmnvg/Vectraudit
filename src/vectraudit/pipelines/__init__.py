"""Text-first, instrumented retrieval pipelines for local evaluation."""

from vectraudit.pipelines.models import (
    EvidenceCoverage,
    Hit,
    PipelineRequest,
    PipelineResult,
    SourceReference,
    StageTrace,
)
from vectraudit.pipelines.runner import (
    PairedSide,
    PipelineOptions,
    execute_paired,
    execute_pipeline,
)
from vectraudit.pipelines.runtime import HTTPPipeline, PythonPipeline, Stage

__all__ = [
    "EvidenceCoverage",
    "HTTPPipeline",
    "Hit",
    "PairedSide",
    "PipelineOptions",
    "PipelineRequest",
    "PipelineResult",
    "PythonPipeline",
    "SourceReference",
    "Stage",
    "StageTrace",
    "execute_paired",
    "execute_pipeline",
]
