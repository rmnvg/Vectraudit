"""Bounded request values shared by datasets, pipelines and local API inputs."""

from __future__ import annotations

import json
import math
import re
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    TypeAdapter,
)

Scalar = StrictStr | StrictInt | StrictFloat | StrictBool
ContextValue = Scalar | list[Scalar]
FilterValue = ContextValue | dict[str, StrictInt | StrictFloat]
FIELD_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _fields(value: Any, *, filters: bool) -> Any:
    if not isinstance(value, dict):
        raise ValueError("context/filter must be an object of field -> value")
    if len(value) > 64:
        raise ValueError("context/filter allows at most 64 fields")
    for key, item in value.items():
        if not isinstance(key, str) or not FIELD_NAME.fullmatch(key):
            raise ValueError("context/filter field must be a dotted identifier")
        if isinstance(item, dict):
            if not filters or not item or set(item) - {"gt", "gte", "lt", "lte"}:
                raise ValueError("filter ranges use only gt, gte, lt, lte")
            if {"gt", "gte"} <= set(item) or {"lt", "lte"} <= set(item):
                raise ValueError("a range must use one lower and one upper bound at most")
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in item.values()):
                raise ValueError("filter range bounds must be finite numbers")
            low, high = item.get("gt", item.get("gte")), item.get("lt", item.get("lte"))
            if (
                low is not None
                and high is not None
                and (low > high or (low == high and ("gt" in item or "lt" in item)))
            ):
                raise ValueError("filter range is empty")
            continue
        values = item if isinstance(item, list) else [item]
        if isinstance(item, list) and (len(item) > 256 or (filters and not item)):
            raise ValueError("lists allow at most 256 values; filter lists must not be empty")
        for scalar in values:
            if type(scalar) not in (str, int, float, bool):
                raise ValueError("context/filter values must be scalars or lists of scalars")
            if isinstance(scalar, float) and not math.isfinite(scalar):
                raise ValueError("context/filter values must be finite")
    if len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode()) > 32768:
        raise ValueError("context/filter exceeds 32 KiB")
    return value


def _context(value: Any) -> Any:
    return _fields(value, filters=False)


def _filters(value: Any) -> Any:
    return _fields(value, filters=True)


Context = Annotated[dict[str, ContextValue], BeforeValidator(_context)]
Filters = Annotated[dict[str, FilterValue], BeforeValidator(_filters)]


class ConversationTurn(BaseModel):
    """One earlier turn, ordered oldest first; the current query is separate."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=8192)


def _history(value: list[ConversationTurn]) -> list[ConversationTurn]:
    if sum(len(turn.content.encode()) for turn in value) > 65536:
        raise ValueError("conversation history exceeds 64 KiB")
    return value


History = Annotated[list[ConversationTurn], Field(max_length=64), AfterValidator(_history)]
CONTEXT = TypeAdapter(Context)
FILTERS = TypeAdapter(Filters)
HISTORY = TypeAdapter(History)


def matches_filter(actual: Any, condition: FilterValue) -> bool:
    """Portable scalar equality, list membership or numeric range (all bounds AND-ed)."""
    if isinstance(condition, list):
        return any(matches_filter(actual, value) for value in condition)
    if isinstance(condition, dict):
        if type(actual) not in (int, float) or not math.isfinite(actual):
            return False
        return all(
            {
                "gt": actual > bound,
                "gte": actual >= bound,
                "lt": actual < bound,
                "lte": actual <= bound,
            }[op]
            for op, bound in condition.items()
        )
    # Preserve existing scalar equality semantics.
    return bool(actual == condition)
