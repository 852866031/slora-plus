# Ported from DeltaServe-vLLM dserve-vllm/vllm/deltaserve/coordinator.py — sglang port (Phase 7).
# Only the admission-gate contract surface is ported; the full SLO predictor /
# backward orchestration / buffer-write bookkeeping land in Phase 8.
"""FT admission gate + minimal coordination state.

Single source of truth (per Scheduler process) for: how many FT tokens may
be admitted this step, whether a backward is pending, and the pause/resume
signals sent to the backward worker over an IPC channel.
"""

from __future__ import annotations

from typing import Any, Optional

from .step_time_estimator import StepTimeEstimator


class FinetuneCoordinator:
    """Admission gate + IPC stubs. Phase 7 surface only.

    Mapping to the upstream vLLM coordinator:
      - ``reserve(n) -> bool``     : admission decision (vLLM returns a write
                                     offset; here we return a bool — the
                                     buffer-offset bookkeeping is Phase 8).
      - ``note_injection(batch)``  : records that an FT batch was injected
                                     this step (vLLM: closes admission once
                                     the buffer can't grow further).
      - ``gpu_pause_backward()``   : signal child to yield the GPU.
      - ``gpu_resume_backward()``  : signal child to take the GPU back.
      - ``on_backward_done(grad)`` : hook fired when the backward acks done
                                     (wired by the scheduler in Phase 8).
    """

    def __init__(
        self,
        finetune_config: Any,
        backward_channel: Optional[str],
    ) -> None:
        self.finetune_config = finetune_config
        self.backward_channel = backward_channel

        # Capacity = per-step FT token budget. Mirrors vLLM coordinator.capacity
        # but kept minimal: Phase 7 does not write into the activation buffer.
        capacity = int(getattr(finetune_config, "max_saved_finetuning_tokens", 256))
        self.capacity = max(1, capacity)
        self.per_step_budget = self.capacity

        # Admission state. ``admission_open`` and ``pending_backward`` mirror
        # vLLM's coordinator: admission closes while a backward is in flight
        # and reopens when poll_backward observes it has finished.
        self.admission_open: bool = True
        self.pending_backward: bool = False

        # ``fill_count`` counts tokens admitted this cycle. ``reserved_fill``
        # tracks tokens admitted in steps whose forward hasn't run yet (only
        # relevant under async scheduling — kept here so Phase 8 can flip it
        # without revisiting this constructor).
        self.fill_count: int = 0
        self.reserved_fill: int = 0

        # Master switch. ``ft_started`` follows finetune_config.start_on_launch
        # so non-FT inference paths are completely unaffected when disabled.
        self.ft_started: bool = bool(
            getattr(finetune_config, "start_on_launch", True)
        )

        # Phase-8 hook: scheduler will assign a callable
        # ``(grad_dict) -> None`` here. Default no-op so test paths
        # don't have to set anything.
        self._on_backward_done_cb = None

        # Records of FT batches injected this step. Phase 8 will drain this
        # in update_from_output. For Phase 7 we just keep the latest count.
        self._last_injected_batch = None
        self._injection_count: int = 0

        # Phase-8: rolling per-kind step-latency estimator. Consulted by
        # ``can_admit`` as the SLO predictor signal.
        self.step_time_estimator = StepTimeEstimator()

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------
    def can_admit(self, *args, **kwargs) -> bool:
        """SLO admission predictor stub. Phase 8 wires the rolling-mean
        backward-latency estimator in but does not yet act on it — a real
        SLO budget comparison is out of scope. Always returns True; the
        ``estimate("backward")`` read keeps the contract honest so a future
        change can flip the comparison without touching call sites."""
        _ = self.step_time_estimator.estimate("backward")
        return True

    def record_step(self, kind: str, latency_ms: float) -> None:
        """Publish a measured per-step latency to the estimator. Called by
        the scheduler's per-step accounting and by the backward worker on
        completion (Phase 8 hookup)."""
        self.step_time_estimator.record_step(kind, latency_ms)

    def space_remaining(self) -> int:
        return max(0, self.capacity - self.fill_count - self.reserved_fill)

    def reserve(self, num_tokens: int) -> bool:
        """Admission gate.

        Returns True iff the scheduler may add ``num_tokens`` FT tokens to
        the running batch this step. Returns False when:
          - finetuning hasn't been started by the operator;
          - admission is closed (buffer full or backward in flight);
          - the predicted step time would violate an SLO (Phase 8);
          - there is no remaining buffer space for ``num_tokens``.

        On success, the reservation is accounted in ``reserved_fill``; the
        scheduler is expected to commit it via ``note_injection`` after the
        forward runs (Phase 8 wires the commit).
        """
        n = int(num_tokens)
        if n <= 0:
            return False
        if not self.ft_started:
            return False
        if not self.admission_open or self.pending_backward:
            return False
        if not self.can_admit(n):
            return False
        if self.space_remaining() < n:
            return False
        self.reserved_fill += n
        return True

    def note_injection(self, batch: Any) -> None:
        """Record that an FT batch was injected into the running step.

        Phase 7 just bookkeeps; Phase 8 will use this to (a) commit the
        reservation post-forward and (b) close admission once the buffer
        can no longer accept the next-smallest sample.
        """
        self._last_injected_batch = batch
        self._injection_count += 1

    # ------------------------------------------------------------------
    # Backward IPC — no-op when channel is None
    # ------------------------------------------------------------------
    def gpu_pause_backward(self) -> None:
        """Ask the backward child to yield the GPU at its next layer
        boundary. No-op when ``backward_channel`` is None (the Phase-7
        default — Phase 8 wires the real IPC)."""
        if self.backward_channel is None:
            return
        # Phase 8: send "pause" over self.backward_channel.

    def gpu_resume_backward(self) -> None:
        """Return the GPU to the backward child. Counterpart to
        ``gpu_pause_backward``. No-op when ``backward_channel`` is None."""
        if self.backward_channel is None:
            return
        # Phase 8: send "resume" over self.backward_channel.

    def on_backward_done(self, grad_dict: Any) -> None:
        """Callback fired by the backward worker when a backward pass has
        produced gradients. Phase 8 wires the real apply-grad / reopen-
        admission path; here we just dispatch to the registered callback
        if any so the contract is testable.
        """
        if self._on_backward_done_cb is not None:
            self._on_backward_done_cb(grad_dict)
        # The actual state transitions (clear pending_backward, reset
        # fill_count, reopen admission) live in poll_backward in the
        # upstream coordinator and will be ported in Phase 8.
