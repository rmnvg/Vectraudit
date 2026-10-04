"""Brute-force protection: count failed authentication attempts per client.

A sliding window per key (client IP). While a key has more failures than the limit in the
window, further attempts get 429 with ``Retry-After``. In-process: with several workers an
attacker gets a multiple of the limit, still far from guessing 256-bit tokens; put a
proxy-level limit in front for production.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable


class FailureLimiter:
    def __init__(
        self, limit: int, window_s: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.limit = limit
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> deque[float]:
        hits = self._hits.get(key)
        if hits is None:
            return deque()
        while hits and hits[0] <= now - self.window_s:
            hits.popleft()
        if not hits:
            del self._hits[key]
        return hits

    def retry_after(self, key: str) -> int | None:
        """Seconds until ``key`` may try again, or ``None`` if it is not blocked."""
        with self._lock:
            now = self._clock()
            hits = self._prune(key, now)
            if len(hits) < self.limit:
                return None
            return max(1, math.ceil(hits[0] + self.window_s - now))

    def failure(self, key: str) -> None:
        with self._lock:
            now = self._clock()
            self._prune(key, now)
            self._hits.setdefault(key, deque()).append(now)
            if len(self._hits) > 100_000:  # bound memory under a distributed attack
                oldest = min(self._hits, key=lambda k: self._hits[k][-1])
                del self._hits[oldest]
