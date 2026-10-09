"""Text-first, instrumented retrieval pipelines for local evaluation."""

from vectraudit.pipelines.models import (
    EvidenceCoverage,
    Hit,
    PipelineRequest,
    PipelineResult,
    SourceReference,
    StageTrace,
)
from vectraudit.pipelines.runner import PipelineOptions, execute_pipeline
from vectraudit.pipelines.runtime import HTTPPipeline, PythonPipeline, Stage

__all__ = [
    "EvidenceCoverage",
    "HTTPPipeline",
    "Hit",
    "PipelineOptions",
    "PipelineRequest",
    "PipelineResult",
    "PythonPipeline",
    "SourceReference",
    "Stage",
    "StageTrace",
    "execute_pipeline",
]
