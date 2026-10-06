"""Typed adapter failures; diagnostics render allowlisted remedies, never exception text."""

from __future__ import annotations


class PipelineCallError(ValueError):
    """A known failure category, compatible with existing ValueError handlers."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
