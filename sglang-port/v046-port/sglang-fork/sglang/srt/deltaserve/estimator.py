# SPDX-License-Identifier: Apache-2.0
"""Three-regime composition-based execution-time estimator.

Faithful port of DeltaServe-vLLM's `dserve-vllm/vllm/deltaserve/estimator.py`
to the sglang DeltaServe fork. The only adaptation is the logging shim
(`dprint`) — the model, the three regimes, the reduced design matrices, the
lstsq fit and the pessimistic safety margin are identical to the vLLM original.

Step-time formula (one flattened step that mixes chunked prefill + decode):

    T_step ≈ α·S + β·T_in + γ·T_ft + δ·B_d + ε·K + c

  S    = Σ nᵢ²       prefill self-attention work (exact if per-request lens are
                     passed, else the T_in²/P proxy)
  T_in = Σ nᵢ        total prefill tokens (FT tokens are a SUBSET: T_ft ⊆ T_in)
  T_ft               FT-prefill tokens; γ = extra activation-save overhead
  B_d                decode requests
  K                  total decode KV tokens

Three coefficient sets, partitioned by step COMPOSITION:
  * INF_PREFILL — T_ft==0 AND T_in>0 → [S, T_in, B_d, K, 1]      (γ≡0)
  * EAGER       — T_ft>0             → [S, T_in, T_ft, B_d, K, 1] (full)
  * DECODE_ONLY — T_in==0 AND T_ft==0 AND B_d>0 → [B_d, K, 1]    (α,β,γ≡0)

Admission is iterative (no closed form): the FT scheduler adds samples one by
one, calls predict(..., regime=EAGER) on each hypothetical, and stops when the
SLO would be violated.

中文说明：
"三区间、按构成"的执行时间估计器，忠实移植自 DeltaServe-vLLM 的 estimator.py，唯一改动是
日志垫片（dprint）；模型、三个区间、精简后的设计矩阵、lstsq 拟合与"悲观安全余量"都与 vLLM
原版一致。
一个步的耗时公式（把分块 prefill 与 decode 混在同一步里）：
    T_step ≈ α·S + β·T_in + γ·T_ft + δ·B_d + ε·K + c
按"步的构成"分成三组系数：
  * INF_PREFILL —— 纯推理 prefill（T_ft==0 且 T_in>0），γ 恒为 0；
  * EAGER       —— 含微调（T_ft>0），用完整设计矩阵（协同服务批次都落在这里）；
  * DECODE_ONLY —— 纯 decode（T_in==0 且 T_ft==0 且 B_d>0），α/β/γ 恒为 0。
准入是迭代式的（没有闭式解）：微调调度器一条条加样本，每加一条就用 predict(regime=EAGER)
预测，一旦会违反 SLO 就停。
"""
from __future__ import annotations

import csv
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_log = logging.getLogger(__name__)
def dprint(*a, **kw):  # noqa: E704 - logging shim matching the vLLM original
    _log.info(" ".join(str(x) for x in a))

# Minimum recorded steps per regime before a (re)fit is attempted (largest
# regime has 6 free params; small margin for a stable lstsq).
MIN_FIT_SAMPLES = 8
# Live refit cadence (in recorded steps).
REFIT_EVERY = 256

REGIME_INF_PREFILL = "inf_prefill"
REGIME_EAGER = "eager"
REGIME_DECODE_ONLY = "decode_only"
REGIMES = (REGIME_INF_PREFILL, REGIME_EAGER, REGIME_DECODE_ONLY)


@dataclass
class StepFeatures:
    """Batch-composition features for one scheduler step. Subset convention:
    ``t_ft`` ⊆ ``t_in`` (FT prefill tokens counted in both)."""

    t_in: float = 0.0          # total prefill tokens (FT subset included)
    p: int = 0                 # number of prefill samples
    t_ft: float = 0.0          # FT prefill tokens (subset of t_in)
    b_d: int = 0               # number of decode requests
    k: float = 0.0             # total decode KV tokens
    prefill_lens: Sequence[int] | None = None

    @property
    def s(self) -> float:
        """Σ nᵢ² — prefill attention work."""
        if self.prefill_lens:
            return float(sum(int(n) * int(n) for n in self.prefill_lens))
        if self.p > 0:
            return float(self.t_in) * float(self.t_in) / float(self.p)
        return 0.0

    @property
    def has_ft(self) -> bool:
        return self.t_ft > 0

    def row(self) -> list[float]:
        return [self.s, float(self.t_in), float(self.t_ft),
                float(self.b_d), float(self.k), 1.0]

    def regime(self) -> str:
        if self.t_ft > 0:
            return REGIME_EAGER
        elif self.t_in > 0:
            return REGIME_INF_PREFILL
        else:
            return REGIME_DECODE_ONLY


@dataclass
class StepParams:
    """Fitted coefficients for one regime. Unused columns are 0.0 (not None);
    cold-start (no fit yet) leaves all None."""

    alpha: float | None = None   # S
    beta: float | None = None    # T_in
    gamma: float | None = None   # T_ft
    delta: float | None = None   # B_d
    epsilon: float | None = None # K
    c: float | None = None       # constant

    @property
    def is_fitted(self) -> bool:
        return all(v is not None for v in
                   (self.alpha, self.beta, self.gamma,
                    self.delta, self.epsilon, self.c))

    def eval(self, f: StepFeatures) -> float:
        return (self.alpha * f.s + self.beta * f.t_in + self.gamma * f.t_ft
                + self.delta * f.b_d + self.epsilon * f.k + self.c)


class StepExecutionTracker:
    """Rolling record of every served step: features, measured + predicted
    duration, and the regime (was_graph) in effect at dispatch."""

    def __init__(self, max_steps: int = 10240) -> None:
        self.max_steps = max_steps
        self.features: list[StepFeatures] = []
        self.durations: list[float] = []
        self.predicted: list[float | None] = []
        self.was_graph: list[bool | None] = []
        self.timestamps: list[float] = []
        self._last_refit_size = 0

    def add(self, features: StepFeatures, duration: float,
            predicted: float | None = None,
            was_graph: bool | None = None) -> None:
        self.features.append(features)
        self.durations.append(float(duration))
        self.predicted.append(predicted)
        self.was_graph.append(was_graph)
        self.timestamps.append(time.time())
        if len(self.durations) > self.max_steps:
            self._drop(0)

    def _drop(self, i: int) -> None:
        del self.features[i]
        del self.durations[i]
        del self.predicted[i]
        del self.was_graph[i]
        del self.timestamps[i]

    def size(self) -> int:
        return len(self.durations)

    def check_refit(self) -> bool:
        n = self.size()
        if n > 0 and n % REFIT_EVERY == 0 and n > self._last_refit_size:
            self._last_refit_size = n
            return True
        return False

    def write_prediction_stats_csv(self, csv_path: str | None) -> None:
        if csv_path is None:
            dprint("[estimator] batch_prediction_stats_path is null — skipping dump.")
            return
        parent = os.path.dirname(csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        rows = []
        for i in range(self.size()):
            pred = self.predicted[i]
            if pred is None:
                continue
            f = self.features[i]
            rows.append({
                "timestamp": self.timestamps[i], "step_index": i,
                "regime": f.regime(), "was_graph": self.was_graph[i],
                "t_in": f.t_in, "p": f.p, "t_ft": f.t_ft,
                "b_d": f.b_d, "k": f.k, "s": f.s,
                "execution_duration": self.durations[i],
                "predicted_duration": pred,
            })
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "timestamp", "step_index", "regime", "was_graph",
                "t_in", "p", "t_ft", "b_d", "k", "s",
                "execution_duration", "predicted_duration"])
            writer.writeheader()
            writer.writerows(rows)
        dprint(f"[estimator] wrote {len(rows)} prediction rows → {csv_path}")


class MergedExecutionEstimator:
    """Three-regime composition-based step-time model. Each regime fits
    independently on its own subset of recorded steps via a reduced design
    matrix; selected at predict time by step composition."""

    def __init__(self) -> None:
        self._params: dict[str, StepParams] = {r: StepParams() for r in REGIMES}
        self._rmse: dict[str, float | None] = {r: None for r in REGIMES}
        self._warned_unfitted = False

    @property
    def is_ready(self) -> bool:
        return any(p.is_fitted for p in self._params.values())

    @property
    def fit_rmse(self) -> float | None:
        for r in (REGIME_EAGER, REGIME_INF_PREFILL, REGIME_DECODE_ONLY):
            if self._rmse[r] is not None:
                return self._rmse[r]
        return None

    @property
    def eager_rmse(self) -> float | None:
        return self._rmse[REGIME_EAGER]

    @property
    def graph_rmse(self) -> float | None:
        return self._rmse[REGIME_DECODE_ONLY]

    # ---------------- prediction ----------------
    def _select(self, features: StepFeatures,
                regime: str | None = None) -> tuple[str, StepParams]:
        if regime is None:
            regime = features.regime()
        p = self._params[regime]
        if p.is_fitted:
            return regime, p
        for r in (REGIME_EAGER, REGIME_INF_PREFILL, REGIME_DECODE_ONLY):
            if self._params[r].is_fitted:
                return r, self._params[r]
        return regime, p

    def predict(self, features: StepFeatures,
                regime: str | None = None,
                apply_margin: bool = True) -> float:
        """Predict step time. ``regime`` overrides composition-derived
        selection (the iterative admission loop forces ``regime="eager"``).
        ``apply_margin`` adds the pessimistic ×(1 + 1.5·RMSE) SLO safety
        margin (DeltaServe)."""
        if not self.is_ready:
            if not self._warned_unfitted:
                dprint("[estimator] predict() called before any fit; returning 0.0")
                self._warned_unfitted = True
            return 0.0
        selected_regime, p = self._select(features, regime=regime)
        pred = p.eval(features)
        rmse = self._rmse[selected_regime]
        if apply_margin and rmse:
            pred *= 1.0 + 1.5 * rmse
        return max(0.0, float(pred))

    # ---------------- fitting ----------------
    def data_fit(self, tracker: StepExecutionTracker) -> dict[str, StepParams]:
        buckets: dict[str, tuple[list[list[float]], list[float]]] = {
            r: ([], []) for r in REGIMES}
        for f, dur in zip(tracker.features, tracker.durations):
            r = f.regime()
            X, y = buckets[r]
            X.append(self._row_for_regime(f, r))
            y.append(dur)
        msg_parts: list[str] = []
        for r in REGIMES:
            X, y = buckets[r]
            self._params[r], self._rmse[r] = self._fit_regime(
                X, y, self._params[r], r)
            rmse_str = (f"{self._rmse[r]:.4f}" if self._rmse[r] is not None else "n/a")
            msg_parts.append(f"{r}={len(y)}(rmse={rmse_str})")
        dprint("[estimator] fit: " + ", ".join(msg_parts))
        return self._params

    @staticmethod
    def _row_for_regime(f: StepFeatures, regime: str) -> list[float]:
        if regime == REGIME_INF_PREFILL:
            return [f.s, f.t_in, f.b_d, f.k, 1.0]            # 5 cols
        elif regime == REGIME_EAGER:
            return [f.s, f.t_in, f.t_ft, f.b_d, f.k, 1.0]    # 6 cols
        else:  # REGIME_DECODE_ONLY
            return [f.b_d, f.k, 1.0]                          # 3 cols

    @staticmethod
    def _scatter_coefs(coef: np.ndarray, regime: str) -> StepParams:
        if regime == REGIME_INF_PREFILL:
            alpha, beta, delta, epsilon, c = (float(v) for v in coef)
            gamma = 0.0
        elif regime == REGIME_EAGER:
            alpha, beta, gamma, delta, epsilon, c = (float(v) for v in coef)
        else:  # REGIME_DECODE_ONLY
            delta, epsilon, c = (float(v) for v in coef)
            alpha = beta = gamma = 0.0
        return StepParams(alpha=alpha, beta=beta, gamma=gamma,
                          delta=delta, epsilon=epsilon, c=c)

    @classmethod
    def _fit_regime(cls, X: list[list[float]], y: list[float],
                    prev: StepParams, regime: str) -> tuple[StepParams, float | None]:
        if len(y) < MIN_FIT_SAMPLES:
            if len(y) > 0:
                dprint(f"[estimator] {regime} regime has {len(y)} samples "
                       f"(<{MIN_FIT_SAMPLES}); keeping previous params")
            return prev, None
        A = np.asarray(X, dtype=float)
        b = np.asarray(y, dtype=float)
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        params = cls._scatter_coefs(coef, regime)
        preds = A @ coef
        rmse = float(np.sqrt(np.mean((preds - b) ** 2)))
        return params, rmse
