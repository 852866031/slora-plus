# Ported from DeltaServe-vLLM dserve-vllm/vllm/deltaserve/estimator.py — sglang
# port (Phase 8). The upstream estimator is a 6-parameter linear model + lstsq
# refit; this port keeps only the rolling-mean core. The richer SLO predictor
# is out of scope for the CPU-only port — Phase 7's admission gate just needs
# *some* latency signal it can consult, and a per-kind rolling mean is the
# minimum shape that matches the contract.
"""StepTimeEstimator — rolling mean of recent step latencies per kind."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Iterable

DEFAULT_WINDOW = 64
SEED_KINDS = ("inference", "backward")


class StepTimeEstimator:
    """Tiny per-kind rolling-mean latency tracker.

    Keys are free-form strings ("inference", "backward", ...). The class
    seeds the two canonical kinds at construction so ``estimate`` and
    ``kinds`` are well-defined immediately, even before any sample lands.
    """

    def __init__(self, window: int = DEFAULT_WINDOW) -> None:
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        self._window = int(window)
        self._buffers: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self._window)
        )
        # Seed canonical kinds so estimate("inference") / estimate("backward")
        # return 0.0 instead of materializing an entry on first read.
        for k in SEED_KINDS:
            _ = self._buffers[k]

    @property
    def window(self) -> int:
        return self._window

    def record_step(self, kind: str, latency_ms: float) -> None:
        """Append one observation. Negative / NaN samples are dropped — a
        backward worker that fails mid-step would otherwise poison the mean.
        """
        if not isinstance(kind, str) or not kind:
            raise ValueError(f"kind must be a non-empty str, got {kind!r}")
        ms = float(latency_ms)
        if ms != ms:  # NaN
            return
        if ms < 0.0:
            return
        self._buffers[kind].append(ms)

    def estimate(self, kind: str) -> float:
        """Rolling mean of the last ``window`` samples for ``kind``.

        Returns 0.0 when no samples have been recorded yet; callers treat
        0.0 as "no signal — assume admission is safe".
        """
        buf = self._buffers.get(kind)
        if not buf:
            return 0.0
        return sum(buf) / len(buf)

    def sample_count(self, kind: str) -> int:
        buf = self._buffers.get(kind)
        return len(buf) if buf else 0

    def kinds(self) -> Iterable[str]:
        """Names of every kind that has either been seeded or recorded."""
        return tuple(self._buffers.keys())

    def reset(self, kind: str | None = None) -> None:
        """Clear one kind's history, or all of them when ``kind`` is None.
        Seeded kinds remain present (empty)."""
        if kind is None:
            for k in list(self._buffers.keys()):
                self._buffers[k].clear()
            return
        if kind in self._buffers:
            self._buffers[kind].clear()
