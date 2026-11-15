"""A global ceiling on calls per second, shared by every worker thread of a run."""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Hands out call slots at most ``max_rps`` per second, across threads.

    :meth:`reserve` books the next slot and returns how long the caller must wait for it;
    the caller sleeps in whatever way suits it (a cancellable wait, an event). With no limit
    every wait is zero.
    """

    def __init__(self, max_rps: float | None) -> None:
        self._interval = 1 / max_rps if max_rps else 0.0
        self._lock = threading.Lock()
        self._next_call = 0.0

    def reserve(self) -> float:
        """Seconds until this caller's slot (0 when it may go now)."""
        if not self._interval:
            return 0.0
        with self._lock:
            now = time.monotonic()
            slot = max(now, self._next_call)
            self._next_call = slot + self._interval
        return slot - now
