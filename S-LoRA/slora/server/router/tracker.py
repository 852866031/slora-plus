from dataclasses import dataclass
import math
import math
from typing import Iterable, List, Optional, Sequence, Tuple
import time
import numpy as np
from slora.server.io_struct import Req
from enum import Enum

EPS = 1e-9

class BatchExecutionType(Enum):
    PREFILL = 0
    DECODE = 1


class BatchExecutionTracker():
    def __init__(self, max_batches = 1024) -> None:
        self.max_batches = max_batches
        self.inference_tokens_list = []
        self.finetuning_tokens_list = []
        self.execution_type_list = []
        self.execution_duration_list = []
        self.last_refit_count = 0
    
    def check_refit(self) -> bool:
        if self.size()%256 == 0 and self.size() > self.last_refit_count:
            self.last_refit_count = self.size()
            return True
        return False

    def _enforce_max_size(self) -> None:
        if self.max_batches is not None and len(self.execution_type_list) > self.max_batches:
            self.drop_batch_stats(0)

    def add_batch_stats(
        self,
        inference_tokens: Sequence[List[int]],        # per-request inference tokens
        finetuning_tokens: Sequence[List[int]],    # per-request FT tokens
        execution_type: BatchExecutionType,
        execution_duration: float,
    ) -> None:
        self.inference_tokens_list.append(inference_tokens)
        self.finetuning_tokens_list.append(finetuning_tokens)
        self.execution_type_list.append(execution_type)
        self.execution_duration_list.append(execution_duration)
    
    def drop_batch_stats(self, index: int) -> None:
        """Drop the batch statistics at the specified index."""
        if index < 0 or index >= len(self.execution_type_list):
            raise IndexError("Index out of range")
        
        del self.inference_tokens_list[index]
        del self.finetuning_tokens_list[index]
        del self.execution_type_list[index]
        del self.execution_duration_list[index]
    
    def size(self) -> int:
        """Return the number of recorded batches."""
        return len(self.execution_type_list)

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Optional
import numpy as np
import time
import math


@dataclass
class PrefillParams:
    """Fitted parameters for the prefill execution model."""
    alpha: Optional[float] = None  # coefficient for sum(n_i^2)
    beta: Optional[float] = None   # coefficient for T_in = sum(n_i)
    gamma: Optional[float] = None  # extra per-token cost for FT
    c: Optional[float] = None      # constant overhead per batch


class PrefillExecutionEstimator:
    """
    Execution time model (prefill):

        T_prefill ≈ α * Σ n_i² + β * T_in + γ * T_ft + c

    - Σ n_i² = quadratic attention term over *all* requests
               (inference + fine-tuning).
    - T_in   = total tokens in batch (inference + FT).
    - T_ft   = total FT tokens only (activation saving overhead).

    Representation:
      • For inference-only batch: List[int] of per-request token counts.
      • For co-serving batch: two independent flat lists:
            inference_tokens: List[int]
            finetuning_tokens: List[int]
        They are *not* index-aligned; they are two disjoint sets of requests.
    """

    def __init__(self) -> None:
        self._params = PrefillParams()
        self.fit_rmse: Optional[float] = None

    @staticmethod
    def _as_np(x: Iterable[float]) -> np.ndarray:
        return np.asarray(list(x), dtype=float)

    @staticmethod
    def _linfit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef

    # ======================================================================
    # Fitting from explicit batch stats
    # ======================================================================
    def fit(
        self,
        inference_only_tokens: Sequence[List[int]],      # per-batch: [n1, n2, ...]
        inference_only_times: Sequence[float],
        coserving_inf_tokens: Sequence[List[int]],       # per-batch: [n1, n2, ...]
        coserving_ft_tokens: Sequence[List[int]],        # per-batch: [m1, m2, ...]
        coserving_times: Sequence[float],
    ) -> PrefillParams:
        """
        Fit α, β, γ, c from:
          - inference-only batches
          - co-serving batches (inference + FT)

        All token lists are flat 1D lists of per-request total token counts.
        """

        sum_n2_list: List[float] = []
        T_in_list: List[float] = []
        T_ft_list: List[float] = []
        T_measured: List[float] = []

        # ---------- inference-only batches ----------
        for token_list, T in zip(inference_only_tokens, inference_only_times):
            n_inf = np.asarray(token_list, dtype=float)
            if n_inf.size == 0:
                continue

            n_total = n_inf
            sum_n2 = float(np.sum(n_total ** 2))
            T_in = float(np.sum(n_total))
            T_ft = 0.0

            sum_n2_list.append(sum_n2)
            T_in_list.append(T_in)
            T_ft_list.append(T_ft)
            T_measured.append(float(T))

        # ---------- co-serving batches ----------
        for inf_list, ft_list, T in zip(
            coserving_inf_tokens,
            coserving_ft_tokens,
            coserving_times,
        ):
            n_inf = np.asarray(inf_list, dtype=float)
            n_ft = np.asarray(ft_list, dtype=float)

            if n_inf.size == 0 and n_ft.size == 0:
                continue

            # All requests in the batch (inference + FT)
            n_total = np.concatenate([n_inf, n_ft])

            sum_n2 = float(np.sum(n_total ** 2))
            T_in = float(np.sum(n_total))
            T_ft = float(np.sum(n_ft))

            sum_n2_list.append(sum_n2)
            T_in_list.append(T_in)
            T_ft_list.append(T_ft)
            T_measured.append(float(T))

        if len(T_measured) < 4:
            raise ValueError("Not enough batches to fit PrefillExecutionEstimator (need ≥4).")

        S = self._as_np(sum_n2_list)
        Tin = self._as_np(T_in_list)
        Tft = self._as_np(T_ft_list)
        T = self._as_np(T_measured)

        X = np.column_stack([S, Tin, Tft, np.ones_like(T)])
        alpha, beta, gamma, c = self._linfit(X, T)

        self._params = PrefillParams(
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
            c=float(c),
        )

        preds = X @ np.array([alpha, beta, gamma, c])
        self.fit_rmse = float(np.sqrt(np.mean((preds - T) ** 2)))
        return self._params

    # ======================================================================
    # Prediction API
    # ======================================================================
    def predict_inference(self, token_list: List[int]) -> float:
        """Predict prefill time for an inference-only batch."""
        p = self._params
        if any(v is None for v in (p.alpha, p.beta, p.c)):
            raise ValueError("Model not fitted yet")

        n = np.asarray(token_list, dtype=float)
        if n.size == 0:
            return 0.0

        S = np.sum(n ** 2)
        Tin = np.sum(n)

        pred = p.alpha * S + p.beta * Tin + p.c

        if self.fit_rmse:
            pred *= 1 + 2 * self.fit_rmse
        return float(pred)

    def predict_coserving(
        self,
        inference_tokens: List[int],
        finetuning_tokens: List[int],
    ) -> float:
        """
        Predict prefill time for a co-serving batch.

        inference_tokens: per-request inference token counts
        finetuning_tokens: per-request FT token counts
        """

        p = self._params
        if any(v is None for v in (p.alpha, p.beta, p.gamma, p.c)):
            raise ValueError("Model not fitted yet")

        n_inf = np.asarray(inference_tokens, dtype=float)
        n_ft = np.asarray(finetuning_tokens, dtype=float)

        if n_inf.size == 0 and n_ft.size == 0:
            return 0.0

        n_total = np.concatenate([n_inf, n_ft])

        S = float(np.sum(n_total ** 2))
        Tin = float(np.sum(n_total))
        Tft = float(np.sum(n_ft))

        pred = p.alpha * S + p.beta * Tin + p.gamma * Tft + p.c
        return float(pred)

    # ======================================================================
    # Verification helpers
    # ======================================================================
    def verify_inference(self, token_list: List[int], actual_time: float) -> float:
        pred = self.predict_inference(token_list)
        err = abs(pred - actual_time) / max(actual_time, 1e-9)
        print(f"[verify_inference] pred {pred:.3f}s vs actual {actual_time:.3f}s (err {err:.2%})")
        self.fit_rmse = max(self.fit_rmse or 0.0, err)
        return err

    def verify_coserving(
        self,
        inference_tokens: List[int],
        finetuning_tokens: List[int],
        actual_time: float,
    ) -> float:
        pred = self.predict_coserving(inference_tokens, finetuning_tokens)
        err = abs(pred - actual_time) / max(actual_time, 1e-9)
        print(f"[verify_coserving] pred {pred:.3f}s vs actual {actual_time:.3f}s (err {err:.2%})")
        self.fit_rmse = max(self.fit_rmse or 0.0, err)
        return err

    # ======================================================================
    # SLO-based FT admission
    # ======================================================================
    def max_next_ft_tokens(
        self,
        inf_tokens: List[int],     # current inference requests
        ft_tokens: List[int],      # current FT requests
        earliest_req_time: Optional[float],
        ttft: float,
        *,
        ttft_unit: str = "s",
        now: Optional[float] = None,
    ) -> int:
        """
        Compute the maximum FT tokens x allowed for the *next FT request*
        so that TTFT SLO is not violated.

        Current batch:
          - inference requests: inf_tokens
          - FT requests:        ft_tokens
        New FT request: adds x tokens (as its own request).
        """

        p = self._params
        if any(v is None for v in (p.alpha, p.beta, p.gamma, p.c)):
            raise ValueError("PrefillExecutionEstimator not fitted yet.")

        # No SLO → effectively unlimited
        if earliest_req_time is None:
            return 10**12

        if now is None:
            now = time.time()

        # Normalize TTFT
        if ttft_unit == "s":
            ttft_s = float(ttft)
        elif ttft_unit == "ms":
            ttft_s = float(ttft) / 1000.0
        else:
            raise ValueError("ttft_unit must be 's' or 'ms'.")

        deadline = float(earliest_req_time) + ttft_s
        rem_time = deadline - now
        if rem_time <= 0:
            return 0

        # Safety margin (can incorporate RMSE if you want)
        safety = 1.0
        time_budget = rem_time / safety

        # Current batch stats
        n_inf = np.asarray(inf_tokens, dtype=float)
        n_ft = np.asarray(ft_tokens, dtype=float)
        n_total = np.concatenate([n_inf, n_ft]) if (n_inf.size or n_ft.size) else np.zeros(0, dtype=float)

        S_curr = float(np.sum(n_total ** 2))
        T_in_curr = float(np.sum(n_total))
        T_ft_curr = float(np.sum(n_ft))

        const_term = (
            p.alpha * S_curr +
            p.beta * T_in_curr +
            p.gamma * T_ft_curr +
            p.c
        )

        rhs = time_budget - const_term
        if rhs <= 0:
            return 0

        # New FT request with x tokens:
        #   - contributes n_total = x
        #   - contributes T_ft    = x
        #
        # Incremental delay:
        #   ΔT(x) = α * x² + (β + γ) * x
        #
        # Solve: α x² + (β + γ) x - rhs ≤ 0
        a2 = float(p.alpha)
        a1 = float(p.beta + p.gamma)

        if a2 <= 0:
            # Degenerate → linear bound
            x_cont = rhs / max(a1, 1e-12)
            return max(0, int(math.floor(x_cont)))

        disc = a1 * a1 + 4.0 * a2 * rhs
        if disc < 0:
            return 0

        x_root = (-a1 + math.sqrt(disc)) / (2.0 * a2)
        x_max = max(0, math.floor(x_root))
        return int(x_max)

    def data_fit(self, tracker: "BatchExecutionTracker") -> PrefillParams:
        """
        Fit the prefill estimator from tracked PREFILL batches.

        A PREFILL batch may be:
            1) inference-only:  n_inf != [], n_ft == []
            2) FT-only:         n_inf == [], n_ft != []
            3) co-serving:      n_inf != [], n_ft != []
        All are valid.
        """

        sum_n2_list = []
        T_in_list = []
        T_ft_list = []
        times = []

        for inf_tokens_per_batch, ft_tokens_per_batch, exec_type, duration in zip(
            tracker.inference_tokens_list,
            tracker.finetuning_tokens_list,
            tracker.execution_type_list,
            tracker.execution_duration_list,
        ):
            if exec_type != BatchExecutionType.PREFILL:
                continue

            # Convert to numpy arrays (flat integer lists)
            n_inf = np.asarray(inf_tokens_per_batch, dtype=float)
            n_ft  = np.asarray(ft_tokens_per_batch, dtype=float)

            # Skip totally empty batches (should not happen but safe)
            if n_inf.size == 0 and n_ft.size == 0:
                continue

            # Combine inference + FT requests
            n_total = (
                np.concatenate([n_inf, n_ft])
                if (n_inf.size and n_ft.size)
                else (n_inf if n_inf.size else n_ft)
            )

            # Compute model terms
            sum_n2 = float(np.sum(n_total ** 2))   # Σ n_i^2
            T_in = float(np.sum(n_total))          # Σ n_i
            T_ft = float(np.sum(n_ft))             # Σ (FT tokens only)

            sum_n2_list.append(sum_n2)
            T_in_list.append(T_in)
            T_ft_list.append(T_ft)
            times.append(float(duration))

        # Require enough samples
        if len(times) < 4:
            raise ValueError(
                "Not enough PREFILL batches to fit PrefillExecutionEstimator (need ≥4)."
            )

        S   = np.asarray(sum_n2_list)
        Tin = np.asarray(T_in_list)
        Tft = np.asarray(T_ft_list)
        T   = np.asarray(times)

        # Solve linear regression
        X = np.column_stack([S, Tin, Tft, np.ones_like(T)])
        alpha, beta, gamma, c = self._linfit(X, T)

        self._params = PrefillParams(
            alpha=float(alpha),
            beta=float(beta),
            gamma=float(gamma),
            c=float(c),
        )

        preds = X @ np.array([alpha, beta, gamma, c])
        self.fit_rmse = float(np.sqrt(np.mean((preds - T) ** 2)))

        return self._params
        

@dataclass
class DecodeParams:
    delta: float = 0.0   # per-request term
    epsilon: float = 0.0 # per-KV token term
    d: float = 0.0       # constant overhead

class DecodeExecutionEstimator:
    """
    Execution time model (decode per step):

        T_decode ≈ δ * B_t + ε * K_t + d

    where:
        B_t = number of active inference requests
        K_t = total KV-cache tokens across requests
    """

    def __init__(self) -> None:
        self._params = DecodeParams()
        self.fit_rmse = None  # same as PrefillExecutionEstimator

    @staticmethod
    def _as_np_1d(x: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(x), dtype=float)
        if arr.ndim != 1:
            raise ValueError("Expected 1D array-like")
        return arr

    @staticmethod
    def _linfit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef

    # ----------------------------------------------------------------------
    # Fitting from raw arrays
    # ----------------------------------------------------------------------
    def fit(
        self,
        total_tokens: Sequence[float],  # K_t
        batch_sizes: Sequence[float],   # B_t
        times: Sequence[float],         # measured decode times
    ) -> DecodeParams:
        K = self._as_np_1d(total_tokens)
        B = self._as_np_1d(batch_sizes)
        T = self._as_np_1d(times)

        X = np.column_stack([B, K, np.ones_like(B)])
        delta, epsilon, d = self._linfit(X, T)

        self._params = DecodeParams(delta=float(delta), epsilon=float(epsilon), d=float(d))

        preds = X @ np.array([delta, epsilon, d])
        self.fit_rmse = float(np.sqrt(np.mean((preds - T) ** 2)))
        return self._params

    # ----------------------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------------------
    def predict(self, total_tokens: float, batch_size: float) -> float:
        p = self._params
        pred = p.delta * batch_size + p.epsilon * total_tokens + p.d
        if self.fit_rmse:
            pred += 1.5 * self.fit_rmse
        return float(pred)

    def verify(self, total_tokens: float, batch_size: float, actual_time: float) -> float:
        pred = self.predict(total_tokens, batch_size)
        err = abs(pred - actual_time) / max(actual_time, 1e-9)
        print(f"[verify_decode] pred {pred:.6f}s vs actual {actual_time:.6f}s (err={err:.2%})")
        self.fit_rmse = max(self.fit_rmse or 0.0, err)
        return err

    # ----------------------------------------------------------------------
    # Fitting directly from the tracker
    # ----------------------------------------------------------------------
    def data_fit(self, tracker: BatchExecutionTracker) -> DecodeParams:
        """
        Fit using DECODE batches from tracker.

        For each DECODE batch:
            B_t = number of inference requests in batch
            K_t = total inference tokens (KV cache size)
        """

        B_list = []
        K_list = []
        T_list = []

        for inf_tokens_per_batch, ft_tokens_per_batch, exec_type, duration in zip(
            tracker.inference_tokens_list,
            tracker.finetuning_tokens_list,
            tracker.execution_type_list,
            tracker.execution_duration_list,
        ):
            if exec_type != BatchExecutionType.DECODE:
                continue

            # Decode only applies to inference requests
            # inf_tokens_per_batch = List[List[int]] OR List[int]
            try:
                # Sum per-request input lengths if nested
                n_inf = np.asarray([sum(toks) for toks in inf_tokens_per_batch], dtype=float)
            except Exception:
                # If already flat: [12, 18, 22]
                n_inf = np.asarray(inf_tokens_per_batch, dtype=float)

            if len(n_inf) == 0:
                continue

            B = len(n_inf)          # number of active requests
            K = float(np.sum(n_inf))  # total KV cache size

            B_list.append(B)
            K_list.append(K)
            T_list.append(duration)

        # Need at least 3 decode samples to fit Δ, ε, d
        if len(T_list) < 3:
            raise ValueError("Not enough DECODE batches to fit DecodeExecutionEstimator (need ≥3).")

        B_arr = np.asarray(B_list, dtype=float)
        K_arr = np.asarray(K_list, dtype=float)
        T_arr = np.asarray(T_list, dtype=float)

        X = np.column_stack([B_arr, K_arr, np.ones_like(B_arr)])
        delta, epsilon, d = self._linfit(X, T_arr)

        self._params = DecodeParams(delta=float(delta), epsilon=float(epsilon), d=float(d))

        preds = X @ np.array([delta, epsilon, d])
        self.fit_rmse = float(np.sqrt(np.mean((preds - T_arr) ** 2)))
        return self._params