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

中文说明：
让移植过来的"三区间执行时间估计器"在 sglang 的协同服务路径上真正生效。模型运行器（model
runner）做三件事：
  - 从每个 forward_batch 构造 StepFeatures（都是 python int 批次元数据，几乎不触发 GPU 同步）；
  - 用"延迟读取的 CUDA event"给每一步计时（不在热路径上做 per-step synchronize），把
    (特征, 时长) 喂给 note_step() —— 估计器在线重新拟合；
  - 在派发 LoRA 反向之前先咨询 should_fire_backward()。
门控是"按需开启"的（SGLANG_DS_SLO_GATE=1），所以默认路径逐字节不变；而估计器的"训练"始终
开着（很便宜），这样一旦翻开门控，模型已经是热的。对应 DeltaServe-vLLM 里 admit_ft_to_step
的 EAGER 区间余量检查。
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
    so non-FT batches add zero sync.

    中文：把一个 forward step 的"构成"抽成特征。只用 python int 的批次元数据
    （extend_num_tokens / batch_size / seq_lens_sum），不触发 GPU 同步。ft_tokens 由调用
    方传入（它在门控里已经算过了），所以非微调批次不增加任何同步开销。extend（prefill）步
    记 t_in/p，decode 步记 b_d/k。"""
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
        # Fraction of the TBT budget the *recent decode* step time may reach
        # before we defer a backward (inference already SLO-stressed → don't pile
        # on). Decode load is the right signal: the backward-fire decision is
        # consulted on FT-prefill steps (b_d=0), so we must look at the
        # surrounding decode steps, not the prefill step's own (b_d=0) features.
        self.defer_frac = float(os.environ.get("SGLANG_DS_SLO_DEFER_FRAC", "0.8"))
        self._pending = deque()      # (start_evt, end_evt, features)
        self._last_decode_dur: Optional[float] = None
        self.fires = 0
        self.defers = 0
        self._n_recorded = 0

    # ---- deferred CUDA-event timing (no per-step synchronize) ----
    # 中文：用一对 CUDA event 在 GPU 流上打时间戳；不在热路径上 synchronize，而是把
    # (start, end, 特征) 入队，后续步用非阻塞的 end.query() 检查是否完成，完成了才读时长并
    # 记录。这样计时几乎零开销，又不污染推理延迟。
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
        if features.b_d > 0 and features.t_in == 0:   # a decode-only step
            self._last_decode_dur = duration_s
        self._n_recorded += 1
        if self.tracker.check_refit():
            self.estimator.data_fit(self.tracker)
            logger.warning(f"[DeltaServe] SLO estimator refit @ {self._n_recorded} "
                           f"steps; ready={self.estimator.is_ready} "
                           f"rmse={self.estimator.fit_rmse}")

    # ---- the gate ----
    def should_fire_backward(self, baseline: StepFeatures, ft_tokens: int) -> bool:
        """Return True to fire the backward now, False to defer it (drop this
        fire). When the gate is enabled and the estimator is warm, defer if the
        recent inference decode step time is already a large fraction of the TBT
        budget — adding the backward's GPU contention would risk blowing TBT.

        Signal is the recent *decode* step time, not the FT-prefill step's own
        features (which have b_d=0): the backward runs concurrently with the
        decode steps it would slow down. On the MPS-isolated path this is a soft
        throttle layered on the subprocess backpressure; it only triggers when
        decode is genuinely SLO-stressed.

        中文：返回 True 表示现在就派发反向，False 表示推迟（丢弃这一次）。当门控开启且估计器
        已就绪时，如果"最近一次推理 decode 步"的耗时已经占到 TBT 预算的很大比例，就推迟反向
        —— 因为再叠加反向带来的 GPU 争用可能把 TBT 撑爆。判断信号取的是最近的 decode 步耗时，
        而不是 FT-prefill 步自身的特征（那一步 b_d=0）：因为反向是和它会拖慢的 decode 步并发
        跑的。在 MPS 隔离路径上，这是叠加在子进程背压之上的"软"节流，只有 decode 真的被 SLO
        压力顶到时才会触发。"""
        if not self.gate_enabled or not self.estimator.is_ready:
            self.fires += 1
            return True
        recent = self._last_decode_dur
        if recent is not None and recent > self.defer_frac * self.max_tbt_slo:
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
