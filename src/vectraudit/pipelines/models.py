"""Strict, bounded contracts shared by Python and HTTP pipelines."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vectraudit.contracts import Context, Filters, History


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SourceReference(Contract):
    """Optional source/chunk identity; ``Hit.id`` remains the relevance-label identity."""

    source_id: str = Field(min_length=1, max_length=512)
    source_revision: str | None = Field(default=None, min_length=1, max_length=512)
    chunk_id: str | None = Field(default=None, min_length=1, max_length=512)
    start: int | None = Field(default=None, ge=0, strict=True)
    end: int | None = Field(default=None, ge=1, strict=True)

    @model_validator(mode="after")
    def span(self) -> SourceReference:
        if (self.start is None) != (self.end is None):
            raise ValueError("source spans require both start and end")
        if self.start is not None and self.end is not None and self.start >= self.end:
            raise ValueError("source spans require start < end")
        return self


Observation = Literal["observed", "partial", "unrecorded", "unsupported"]


class EvidenceCoverage(Contract):
    """Version 1 declares observation completeness, not inferred application capabilities."""

    format_version: Literal[1] = 1
    ranked_output: Observation = "observed"
    stage_trace: Observation = "unrecorded"
    source_mapping: Observation = "unrecorded"


class Hit(Contract):
    id: str = Field(min_length=1, max_length=512)
    score: float | None = None
    source: SourceReference | None = None


class EvidenceObservation(Contract):
    format_version: Literal[1] = 1
    coverage: EvidenceCoverage | None = None
    hits: list[Hit] = Field(default_factory=list, max_length=10000)


class ContextItem(Contract):
    """One document as it reached the model. ``id`` is the same identity as ``Hit.id``."""

    id: str = Field(min_length=1, max_length=512)
    source: SourceReference | None = None
    truncated: bool | None = None
    """True when only part of the document reached the model; ``None`` when unknown."""


class ModelContext(Contract):
    """What the application actually gave the model, in order. Version 1.

    ``coverage`` says how completely it was recorded: ``observed`` means every document the
    model received is listed, so a delivered document that is absent was left out;
    ``partial`` lists some documents and proves nothing about absences. ``method`` says how
    the items were obtained: ``reported`` by the application, or ``text_match``, inferred by
    finding recorded document text in the recorded prompt.
    """

    format_version: Literal[1] = 1
    coverage: Observation = "observed"
    method: Literal["reported", "text_match"] = "reported"
    items: list[ContextItem] = Field(default_factory=list, max_length=10000)

    @model_validator(mode="after")
    def coherent(self) -> ModelContext:
        if self.coverage in {"unrecorded", "unsupported"} and self.items:
            raise ValueError("unrecorded model context cannot list documents")
        if len({item.id for item in self.items}) != len(self.items):
            raise ValueError("model context lists a document twice")
        return self


class PipelineRequest(Contract):
    query_id: str
    text: str | None = None
    vector: list[float] | None = None
    k: int = Field(default=10, ge=1, le=10000)
    filters: Filters = Field(default_factory=dict)
    context: Context = Field(default_factory=dict)
    history: History = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)


class StageTrace(Contract):
    name: str = Field(min_length=1, max_length=128)
    kind: Literal["keyword", "vector", "filter", "fusion", "rerank", "truncate", "custom"]
    parents: list[str] = Field(default_factory=list, max_length=64)
    hits: list[Hit] = Field(max_length=10000)
    latency_ms: float = Field(ge=0)
    source: Literal["measured", "reported"] = "reported"

    @model_validator(mode="after")
    def unique_hits(self) -> StageTrace:
        if len({h.id for h in self.hits}) != len(self.hits):
            raise ValueError("stage contains duplicate document ids")
        return self


class PipelineResult(Contract):
    hits: list[Hit] = Field(default_factory=list, max_length=10000)
    stages: list[StageTrace] = Field(default_factory=list, max_length=64)
    abstained: bool = False
    cost_usd: float | None = Field(default=None, ge=0)
    memory_bytes: int | None = Field(default=None, ge=0)
    coverage: EvidenceCoverage | None = None
    context: ModelContext | None = None
    """Optional: the documents the model received. Kept local; never uploaded by default."""

    @model_validator(mode="after")
    def coherent(self) -> PipelineResult:
        if self.coverage is not None:
            if self.coverage.ranked_output in {"unrecorded", "unsupported"} and (
                self.hits or self.abstained
            ):
                raise ValueError("unobserved ranking cannot contain hits or assert abstention")
            if self.coverage.stage_trace in {"unrecorded", "unsupported"} and self.stages:
                raise ValueError("unobserved stage trace cannot contain stages")
            if self.coverage.source_mapping == "observed" and any(
                hit.source is None for hit in self.hits
            ):
                raise ValueError("observed source mapping requires sources for every delivered hit")
        if len({h.id for h in self.hits}) != len(self.hits):
            raise ValueError("result contains duplicate document ids")
        if self.abstained and self.hits:
            raise ValueError("an abstaining pipeline must not return documents")
        seen: set[str] = set()
        for stage in self.stages:
            if stage.name in seen or any(p not in seen for p in stage.parents):
                raise ValueError("traces require unique names and earlier parents")
            seen.add(stage.name)
        if self.stages and self.stages[-1].hits != self.hits:
            raise ValueError("last stage must match the delivered results")
        return self
