"""Strict, bounded contracts shared by Python and HTTP pipelines."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Hit(Contract):
    id: str = Field(min_length=1, max_length=512)
    score: float | None = None


class PipelineRequest(Contract):
    query_id: str
    text: str | None = None
    vector: list[float] | None = None
    k: int = Field(default=10, ge=1, le=10000)
    filters: dict[str, str | int | float | bool] = Field(default_factory=dict)
    context: dict[str, str | int | float | bool] = Field(default_factory=dict)
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

    @model_validator(mode="after")
    def coherent(self) -> PipelineResult:
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
