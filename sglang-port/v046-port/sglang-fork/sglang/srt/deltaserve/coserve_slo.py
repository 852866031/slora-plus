# SPDX-License-Identifier: Apache-2.0
"""Process-local SLO singleton: makes the ported 3-regime estimator LIVE in the
sglang co-serving path (which is request-tagged → backward fires in
model_runner._forward, NOT the disconnected scheduler-mixin scaffold).

The model runner:
  - builds StepFeatures from each forward_batch (python ints, ~no GPU sync),
  - times each step with deferred CUDA events (no per-step synchronize), and
    feeds (features, duration) to note_step() → the estimator refits online;
  - consults should_fire_backward() before dispatching the LoRA backward.

The gate is OPT-IN (SGLANG_DS_SLO_GATE=1) so the default path is byte-unchanged;
estimator TRAINING is always on (cheap) so the model is warm if the gate is
flipped. Mirrors DeltaServe-vLLM admit_ft_to_step's EAGER-regime headroom check.
"""
from __future__ import annotations

import logging
import os
from collections import deque
from typing import Optional

from sglang.srt.deltaserve.estimator import (
    MergedExecutionEstimator,
    StepExecutionTracker,
    StepFeatures,
    REGIME_EAGER,
)

logger = logging.getLogger(__name__)


def build_step_features(forward_batch, ft_tokens: int = 0) -> StepFeatures:
    """Composition features for one forward step. Uses python-int batch
    metadata (extend_num_tokens / batch_size / seq_lens_sum) → no GPU sync.
    ``ft_tokens`` is passed by the caller (it already computed it for the gate),
    so non-FT batches add zero sync."""
    mode = forward_batch.forward_mode
    bs = int(getattr(forward_batch, "batch_size", 0) or 0)
    if mode.is_extend():
        t_in = getattr(forward_batch, "extend_num_tokens", None)
        if t_in is None:
            t_in = int(forward_batch.input_ids.shape[0])
        return StepFeatures(t_in=float(int(t_in)), p=bs,
                            t_ft=float(int(ft_tokens)), b_d=0, k=0.0)
    # decode (or idle)
    return StepFeatures(t_in=0.0, p=0, t_ft=0.0, b_d=bs,
                        k=float(int(getattr(forward_batch, "seq_lens_sum", 0) or 0)))


class _CoServeSLO:
    def __init__(self) -> None:
        self.estimator = MergedExecutionEstimator()
        self.tracker = StepExecutionTracker()
        self.ttft_slo = float(os.environ.get("SGLANG_DS_TTFT_SLO", "0.35"))
        self.max_tbt_slo = float(os.environ.get("SGLANG_DS_MAX_TBT_SLO", "0.15"))
        # Gate is opt-in; training is always on so the estimator is warm.
        self.gate_enabled = os.environ.get("SGLANG_DS_SLO_GATE", "0") == "1"
        self._pending = deque()      # (start_evt, end_evt, features)
        self.fires = 0
        self.defers = 0
        self._n_recorded = 0

    # ---- deferred CUDA-event timing (no per-step synchronize) ----
    def begin_step(self, features: StepFeatures):
        import torch
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return (start, end, features)

    def end_step(self, handle):
        """Record the end event for this step, then drain any prior pending
        steps whose end event has completed → note_step (off the hot path)."""
        if handle is None:
            return
        start, end, feats = handle
        end.record()
        self._pending.append((start, end, feats))
        # Drain completed timings (query() is non-blocking).
        while self._pending:
            s, e, f = self._pending[0]
            if not e.query():
                break
            self._pending.popleft()
            dur_s = s.elapsed_time(e) / 1000.0   # ms → s
            self._record(f, dur_s)

    def _record(self, features: StepFeatures, duration_s: float):
        self.tracker.add(features, duration_s)
        self._n_recorded += 1
        if self.tracker.check_refit():
            self.estimator.data_fit(self.tracker)
            logger.warning(f"[DeltaServe] SLO estimator refit @ {self._n_recorded} "
                           f"steps; ready={self.estimator.is_ready} "
                           f"rmse={self.estimator.fit_rmse}")

    # ---- the gate ----
    def should_fire_backward(self, baseline: StepFeatures, ft_tokens: int) -> bool:
        """Return True to fire the backward now. When the gate is enabled and
        the estimator is warm, defer (False) if firing the FT prefill would push
        the predicted EAGER-regime step time over the TBT budget."""
        if not self.gate_enabled or not self.estimator.is_ready:
            self.fires += 1
            return True
        hyp = StepFeatures(
            t_in=baseline.t_in + ft_tokens, p=baseline.p + 1,
            t_ft=baseline.t_ft + ft_tokens, b_d=baseline.b_d, k=baseline.k)
        t_with_ft = self.estimator.predict(hyp, regime=REGIME_EAGER)
        if baseline.b_d > 0 and t_with_ft > self.max_tbt_slo:
            self.defers += 1
            return False
        self.fires += 1
        return True


_SLO: Optional[_CoServeSLO] = None


def get_slo() -> _CoServeSLO:
    global _SLO
    if _SLO is None:
        _SLO = _CoServeSLO()
    return _SLO
