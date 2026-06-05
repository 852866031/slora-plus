"""DeltaServe GPU-yield contract (Section 6 of CO_SERVING_OPTIMIZATIONS.md).

`_maybe_pause` is the load-bearing co-serving primitive: the backward
yields the GPU at every layer boundary when the main (inference) process
is running a prefill step. Implemented as a multiprocessing.Event:

    SET     → backward may run (steady state)
    CLEARED → backward yields (inference prefill in flight)

The main scheduler clears it before dispatching prefill and sets it after.
The backward subprocess (or in-process Path C runner) calls
`maybe_pause()` between layers.

Fire-and-forget: no blocking sync between pause/resume. The
cross-process-visibility wait is scoped to the activation-copy event
(separate primitive in Section 8, not implemented here yet).
"""
from __future__ import annotations

import threading
from typing import Optional


class _GpuGrant:
    """Backed by a threading.Event (works across coroutines in the main
    process; for true cross-process use mp.Event, but the in-process
    Path C only needs threading)."""

    def __init__(self) -> None:
        self._evt = threading.Event()
        self._evt.set()  # default: backward may run

    def grant(self) -> None:
        """Inference has yielded — backward may run."""
        self._evt.set()

    def revoke(self) -> None:
        """Inference about to start a prefill — backward must yield."""
        self._evt.clear()

    def maybe_pause(self, timeout_s: float = 5.0) -> None:
        """Called by the backward at layer boundaries. No-op when set."""
        if not self._evt.is_set():
            self._evt.wait(timeout=timeout_s)

    def is_granted(self) -> bool:
        return self._evt.is_set()


_GRANT: Optional[_GpuGrant] = None


def gpu_grant() -> _GpuGrant:
    """Process-local singleton. Backward runner calls
    `gpu_grant().maybe_pause()` between layers; scheduler calls
    `gpu_grant().revoke()` before prefill, `grant()` after."""
    global _GRANT
    if _GRANT is None:
        _GRANT = _GpuGrant()
    return _GRANT
