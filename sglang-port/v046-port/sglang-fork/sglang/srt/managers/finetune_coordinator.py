# Ported from DeltaServe-vLLM dserve-vllm/vllm/deltaserve/coordinator.py — sglang port (Phase 7).
# Only the admission-gate contract surface is ported; the full SLO predictor /
# backward orchestration / buffer-write bookkeeping land in Phase 8.
"""FT admission gate + minimal coordination state.

Single source of truth (per Scheduler process) for: how many FT tokens may
be admitted this step, whether a backward is pending, and the pause/resume
signals sent to the backward worker over an IPC channel.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from sglang.srt.deltaserve.estimator import (
    MergedExecutionEstimator,
    StepExecutionTracker,
    StepFeatures,
    REGIME_EAGER,
    REGIME_INF_PREFILL,
    REGIME_DECODE_ONLY,
)


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

        # Real 3-regime SLO execution-time estimator (ported from vLLM
        # DeltaServe). The scheduler/model-runner records each served step's
        # (features, duration) via note_step(); refit fires every REFIT_EVERY.
        # slo_gate_backward() consults it to decide whether firing the backward
        # now would blow the inference SLO.
        self.estimator = MergedExecutionEstimator()
        self.tracker = StepExecutionTracker()
        # SLO budgets (seconds). Read from finetune_config when present, else
        # sensible co-serving defaults (match DeltaServe-vLLM's llama3 YAML).
        self.ttft_slo = float(getattr(finetune_config, "ttft_slo", 0.35))
        self.max_tbt_slo = float(getattr(finetune_config, "max_tbt_slo", 0.15))
        self.avg_tbt_slo = float(getattr(finetune_config, "avg_tbt_slo", 0.10))
        self._steps_since_refit = 0
        self._slo_deferrals = 0
        self._slo_admits = 0

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------
    def can_admit(self, *args, **kwargs) -> bool:
        """Buffer-level precondition (kept for reserve()'s contract). The real
        SLO decision is slo_gate_backward(), consulted at backward-dispatch
        time where the batch composition features are available."""
        return True

    # ------------------------------------------------------------------
    # SLO estimator: record served steps + refit
    # ------------------------------------------------------------------
    def note_step(self, features: "StepFeatures", duration_s: float,
                  predicted_s: Optional[float] = None) -> None:
        """Record one served step's (composition features, measured duration)
        into the tracker and refit every REFIT_EVERY steps. Called by the
        model runner after each forward step (inference-only AND co-serve)."""
        self.tracker.add(features, duration_s, predicted=predicted_s)
        if self.tracker.check_refit():
            self.estimator.data_fit(self.tracker)

    def predict_step(self, features: "StepFeatures",
                     regime: Optional[str] = None,
                     apply_margin: bool = True) -> float:
        return self.estimator.predict(features, regime=regime,
                                      apply_margin=apply_margin)

    def slo_gate_backward(self, baseline: "StepFeatures",
                          ft_tokens: int,
                          earliest_arrival: Optional[float] = None) -> bool:
        """SLO-aware decision: may the backward / FT prefill ride on the
        UPCOMING step without blowing the inference SLO? Adapts vLLM's
        admit_ft_to_step headroom check to sglang's request-tagged path.

        Returns True (admit) when the estimator is cold (no fit yet) — matches
        the pre-redesign cold-start behaviour — or when the predicted EAGER-
        regime step time (baseline + this FT slice) stays under the TBT budget
        and leaves TTFT headroom. Returns False (defer) otherwise."""
        if not self.estimator.is_ready:
            self._slo_admits += 1
            return True  # cold start: admit, keep training the estimator
        # Hypothetical step WITH the FT prefill admitted (forces EAGER regime).
        hyp = StepFeatures(
            t_in=baseline.t_in + ft_tokens,
            p=baseline.p + 1,
            t_ft=baseline.t_ft + ft_tokens,
            b_d=baseline.b_d,
            k=baseline.k,
            prefill_lens=(list(baseline.prefill_lens) + [ft_tokens])
            if baseline.prefill_lens else None,
        )
        t_with_ft = self.estimator.predict(hyp, regime=REGIME_EAGER)
        # TBT headroom: if there are decode requests, the step must stay under
        # the per-token budget.
        if baseline.b_d > 0 and t_with_ft > self.max_tbt_slo:
            self._slo_deferrals += 1
            return False
        # TTFT headroom: a prefill-carrying step must finish within 0.9·TTFT
        # of the earliest waiting request's arrival.
        if baseline.t_in > 0 and earliest_arrival is not None:
            deadline = earliest_arrival + 0.9 * self.ttft_slo
            if (deadline - time.time() - t_with_ft) <= 0:
                self._slo_deferrals += 1
                return False
        self._slo_admits += 1
        return True

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
