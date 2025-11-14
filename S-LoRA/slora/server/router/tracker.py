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
        self.num_ft_reqs_list = []
        self.num_inf_reqs_list = []
        self.num_ft_tokens_list = []
        self.num_inf_tokens_list = []
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
        num_ft_reqs: int,
        num_inf_reqs: int,
        num_ft_tokens: int,
        num_inf_tokens: int,
        execution_type: BatchExecutionType,
        execution_duration: float,
    ) -> None:
        self.num_ft_reqs_list.append(num_ft_reqs)
        self.num_inf_reqs_list.append(num_inf_reqs)
        self.num_ft_tokens_list.append(num_ft_tokens)
        self.num_inf_tokens_list.append(num_inf_tokens)
        self.execution_type_list.append(execution_type)
        self.execution_duration_list.append(execution_duration)
    
    def drop_batch_stats(self, index: int) -> None:
        """Drop the batch statistics at the specified index."""
        if index < 0 or index >= len(self.execution_type_list):
            raise IndexError("Index out of range")
        
        del self.num_ft_reqs_list[index]
        del self.num_inf_reqs_list[index]
        del self.num_ft_tokens_list[index]
        del self.num_inf_tokens_list[index]
        del self.execution_type_list[index]
        del self.execution_duration_list[index]
    
    def size(self) -> int:
        """Return the number of recorded batches."""
        return len(self.execution_type_list)

    def print_batch_stats(self) -> None:
        """Print all recorded batch statistics in a readable table."""
        if not self.execution_type_list:
            print("[BatchExecutionTracker] No batch records to display.")
            return

        print("\n=== Batch Execution Statistics ===")
        header = (
            f"{'Idx':>3} | {'Type':<8} | {'FT_Reqs':>7} | {'INF_Reqs':>8} | "
            f"{'FT_Toks':>8} | {'INF_Toks':>9} | {'Duration (s)':>13}"
        )
        print(header)
        print("-" * len(header))

        for i, (ft_r, inf_r, ft_t, inf_t, typ, dur) in enumerate(
            zip(
                self.num_ft_reqs_list,
                self.num_inf_reqs_list,
                self.num_ft_tokens_list,
                self.num_inf_tokens_list,
                self.execution_type_list,
                self.execution_duration_list,
            )
        ):
            type_str = (
                "PREFILL" if typ == BatchExecutionType.PREFILL
                else "DECODE" if typ == BatchExecutionType.DECODE
                else str(typ)
            )

            print(
                f"{i:>3} | {type_str:<8} | {ft_r:>7} | {inf_r:>8} | "
                f"{ft_t:>8} | {inf_t:>9} | {dur:>13.6f}"
            )

        print("-" * len(header))
        print(
            f"Total batches: {len(self.execution_type_list)} | "
            f"Total time: {sum(self.execution_duration_list):.3f}s"
        )

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

    α, β, γ, c are fitted parameters.
    - Σ n_i² captures quadratic self-attention cost.
    - T_in = total inference + FT tokens in batch.
    - T_ft = fine-tuning token count (activation saving overhead).
    """

    def __init__(self) -> None:
        self._params = PrefillParams()
        self.fit_rmse = None

    @staticmethod
    def _as_np_1d(x: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(x), dtype=float)
        if arr.ndim != 1:
            raise ValueError("Expected 1D iterable of floats")
        return arr

    @staticmethod
    def _linfit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef

    def fit(
        self,
        inference_tokens: Sequence[float],
        inference_reqs: Sequence[float],
        inference_times: Sequence[float],
        coserving_tokens: Sequence[Tuple[float, float]],  # [(N_inf, N_ft), ...]
        coserving_reqs: Sequence[float],
        coserving_times: Sequence[float],
        enforce_ratio: bool = True,
    ) -> PrefillParams:
        """
        Fit α, β, γ, c using both inference-only and co-serving batches.
        For inference-only: T_ft = 0.
        """
        Ni_inf = self._as_np_1d(inference_tokens)
        Bi_inf = self._as_np_1d(inference_reqs)
        Ti_inf = self._as_np_1d(inference_times)

        co_pairs = list(coserving_tokens)
        Bi_co = self._as_np_1d(coserving_reqs)
        Ti_co = self._as_np_1d(coserving_times)

        # Flatten into joint regression dataset
        sum_n2_list = []
        T_in_list = []
        T_ft_list = []
        T_measured = []

        # --- inference-only points ---
        for Ni, Ti in zip(Ni_inf, Ti_inf):
            sum_n2_list.append(Ni ** 2 * Bi_inf[0] if Bi_inf.size > 0 else Ni ** 2)
            T_in_list.append(Ni)
            T_ft_list.append(0.0)
            T_measured.append(Ti)

        # --- co-serving points ---
        for (N_inf, N_ft), Ti in zip(co_pairs, Ti_co):
            sum_n2_list.append((N_inf + N_ft) ** 2)
            T_in_list.append(N_inf + N_ft)
            T_ft_list.append(N_ft)
            T_measured.append(Ti)

        # Fit linear model: T = α * Σ n_i² + β * T_in + γ * T_ft + c
        S = self._as_np_1d(sum_n2_list)
        Tin = self._as_np_1d(T_in_list)
        Tft = self._as_np_1d(T_ft_list)
        T = self._as_np_1d(T_measured)

        X = np.column_stack([S, Tin, Tft, np.ones_like(T)])
        alpha, beta, gamma, c = self._linfit(X, T)

        self._params = PrefillParams(alpha=float(alpha), beta=float(beta), gamma=float(gamma), c=float(c))

        preds = X @ np.array([alpha, beta, gamma, c])
        self.fit_rmse = float(np.sqrt(np.mean((preds - T) ** 2)))
        return self._params

    def predict_inference(self, N_inf: float, B: float) -> float:
        """Predict prefill time for inference-only batch."""
        p = self._params
        if any(v is None for v in (p.alpha, p.beta, p.gamma, p.c)):
            raise ValueError("Model not fitted yet")
        S = B * (N_inf ** 2)
        Tin = B * N_inf
        pred = p.alpha * S + p.beta * Tin + p.c
        if self.fit_rmse:
            pred *= 1 + self.fit_rmse
        return float(pred)

    def predict_coserving(self, N_inf: float, N_ft: float, B: float) -> float:
        """Predict prefill time for co-serving batch."""
        p = self._params
        if any(v is None for v in (p.alpha, p.beta, p.gamma, p.c)):
            raise ValueError("Model not fitted yet")
        S = B * ((N_inf + N_ft) ** 2)
        Tin = B * (N_inf + N_ft)
        Tft = B * N_ft
        pred = p.alpha * S + p.beta * Tin + p.gamma * Tft + p.c
        return float(pred)

    def verify_inference(self, N_inf: float, B: float, actual_time: float) -> float:
        pred = self.predict_inference(N_inf, B)
        err = abs(pred - actual_time) / actual_time
        print(f"[verify_inference] pred {pred:.3f}s vs actual {actual_time:.3f}s (err {err:.2%})")
        self.fit_rmse = max(self.fit_rmse or 0.0, err)
        return err

    def verify_coserving(self, N_inf: float, N_ft: float, B: float, actual_time: float) -> float:
        pred = self.predict_coserving(N_inf, N_ft, B)
        err = abs(pred - actual_time) / actual_time
        print(f"[verify_coserving] pred {pred:.3f}s vs actual {actual_time:.3f}s (err {err:.2%})")
        self.fit_rmse = max(self.fit_rmse or 0.0, err)
        return err
    
    def can_add_ft(
        self,
        inf_token_num: float,
        ft_token_num: float,
        current_batch_size: int,
        earliest_req_time: float,
        next_ft_token_num: float,
        ttft: float,
        *,
        ttft_unit: str = "s",
        now: Optional[float] = None,
    ) -> bool:
        # Ensure model is fitted for co-serving prediction
        if (self._params.alpha is None or
            self._params.beta is None or
            self._params.gamma is None or
            self._params.c is None):
            raise ValueError("PrefillExecutionEstimator not fitted (need alpha,beta,gamma,c).")

        if now is None:
            now = time.time()

        # Normalize TTFT to seconds
        if ttft_unit == "s":
            ttft_s = float(ttft)
        elif ttft_unit == "ms":
            ttft_s = float(ttft) / 1000.0
        else:
            raise ValueError(f"Unsupported ttft_unit '{ttft_unit}', use 's' or 'ms'.")
        if earliest_req_time is None:
            return True  # no SLO to meet

        # Predict prefill time if we add this FT sample (co-serving)
        N_inf = float(inf_token_num)
        N_ft  = float(next_ft_token_num + ft_token_num)
        Bp    = float(current_batch_size + 1)  # adding the FT request

        predicted_prefill = self.predict_coserving(N_inf=N_inf, N_ft=N_ft, B=Bp)

        # Check against the SLO deadline
        predicted_finish = now + predicted_prefill
        deadline = float(earliest_req_time) + ttft_s
        return predicted_finish <= deadline

    def max_next_ft_tokens(
        self,
        inf_token_num: float,
        ft_token_num: float,
        current_batch_size: int,
        earliest_req_time: Optional[float],
        ttft: float,
        *,
        ttft_unit: str = "s",
        now: Optional[float] = None,
    ) -> int:
        p = self._params
        if p.alpha is None or p.beta is None or p.gamma is None or p.c is None:
            raise ValueError("PrefillExecutionEstimator not fitted (need alpha,beta,gamma,c).")

        # Handle no SLO case
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
            raise ValueError(f"Unsupported ttft_unit '{ttft_unit}', use 's' or 'ms'.")

        deadline = float(earliest_req_time) + ttft_s
        rem_time = deadline - now
        if rem_time <= 0:
            return 0

        # Safety margin based on RMSE
        fit_rmse = float(self.fit_rmse) if self.fit_rmse is not None else 0.0
        safety = 1.0
        adjusted_budget = rem_time / safety

        # Current totals
        N_inf = float(inf_token_num)
        N_ft_curr = float(ft_token_num)
        Bp = float(current_batch_size + 1)  # batch size after adding new FT req

        # Approximate current Σn_i² and T_in
        # Σn_i² ≈ (T_in)² / B  → used as aggregate estimate
        Tin_curr = N_inf + N_ft_curr
        S_curr = (Tin_curr ** 2) / max(Bp, 1.0)

        # Constant part before adding new FT tokens
        const_term = p.alpha * S_curr + p.beta * Tin_curr + p.gamma * N_ft_curr + p.c

        # New request adds x tokens:
        # ΔT = α x² + (β + γ) x
        a2 = float(p.alpha)
        a1 = float(p.beta + p.gamma)
        rhs = adjusted_budget - const_term

        # If no remaining budget
        if rhs <= 0:
            return 0

        # Solve quadratic inequality: a2*x² + a1*x - rhs <= 0
        if a2 <= 0:
            # Degenerate to linear
            x_cont = rhs / max(a1, 1e-12)
            return max(0, int(math.floor(x_cont)))

        disc = a1 * a1 + 4.0 * a2 * rhs
        if disc < 0:
            return 0

        x_root = (-a1 + math.sqrt(disc)) / (2.0 * a2)
        x_max = max(0, math.floor(x_root))
        return int(x_max)
    
    def data_fit(self, tracker: BatchExecutionTracker) -> PrefillParams:
        """
        Fit the prefill estimator from tracked batches.

        For each PREFILL batch:
          - sum_n2 = Σ n_i²  (approximated from total tokens and number of requests)
          - T_in   = total input tokens (inf + ft)
          - T_ft   = total fine-tuning tokens

        Notes:
          If per-request lengths are not logged, we approximate Σ n_i² ≈
          (T_in² / B), which holds if all requests are roughly equal length.
        """
        sum_n2_list = []
        T_in_list = []
        T_ft_list = []
        times = []

        for ft_reqs, inf_reqs, ft_toks, inf_toks, exec_type, duration in zip(
            tracker.num_ft_reqs_list,
            tracker.num_inf_reqs_list,
            tracker.num_ft_tokens_list,
            tracker.num_inf_tokens_list,
            tracker.execution_type_list,
            tracker.execution_duration_list,
        ):
            if exec_type == BatchExecutionType.PREFILL:
                total_reqs = ft_reqs + inf_reqs
                total_toks = ft_toks + inf_toks
                if total_reqs <= 0 or total_toks <= 0:
                    continue

                # Approximate sum(n_i²) = (Σ n_i)² / B
                sum_n2 = (total_toks ** 2) / total_reqs
                sum_n2_list.append(sum_n2)
                T_in_list.append(total_toks)
                T_ft_list.append(ft_toks)
                times.append(duration)

        if len(times) < 3:
            raise ValueError("Not enough PREFILL batches to fit PrefillExecutionEstimator (need ≥3).")

        S = np.array(sum_n2_list)
        Tin = np.array(T_in_list)
        Tft = np.array(T_ft_list)
        T = np.array(times)

        X = np.column_stack([S, Tin, Tft, np.ones_like(T)])
        alpha, beta, gamma, c = self._linfit(X, T)
        self._params = PrefillParams(alpha=float(alpha), beta=float(beta), gamma=float(gamma), c=float(c))
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
    """

    def __init__(self) -> None:
        self._params = DecodeParams()
        self.fit_rmse = None  # <-- same field name as in PrefillExecutionEstimator

    @staticmethod
    def _as_np_1d(x: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(x), dtype=float)
        if arr.ndim != 1:
            raise ValueError("Expected 1D iterable of floats")
        return arr

    @staticmethod
    def _linfit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return coef

    def fit(
        self,
        total_tokens: Sequence[float],  # KV cache size (K_t)
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
        self.fit_rmse = float(np.sqrt(np.mean((preds - T) ** 2)))  # <-- consistent naming
        return self._params

    def predict(self, total_tokens: float, batch_size: float) -> float:
        p = self._params
        T_pred = p.delta * batch_size + p.epsilon * total_tokens + p.d
        if self.fit_rmse:
            T_pred += 1.5 * self.fit_rmse
        return float(T_pred)

    def verify(self, total_tokens: float, batch_size: float, actual_time: float) -> float:
        pred = self.predict(total_tokens, batch_size)
        err = abs(pred - actual_time) / max(actual_time, 1e-9)
        print(f"[verify_decode] pred {pred:.6f}s vs actual {actual_time:.6f}s (err={err:.2%})")
        self.fit_rmse = max(self.fit_rmse or 0.0, err)
        return err

    def data_fit(self, tracker: BatchExecutionTracker) -> DecodeParams:
        """
        Fit using decode batches from tracker.
        For each DECODE batch:
          - B_t : active requests in batch
          - K_t : total KV cache size (approximated as total_tokens)
        """
        B_list, K_list, T_list = [], [], []

        for ft_reqs, inf_reqs, ft_toks, inf_toks, exec_type, duration in zip(
            tracker.num_ft_reqs_list,
            tracker.num_inf_reqs_list,
            tracker.num_ft_tokens_list,
            tracker.num_inf_tokens_list,
            tracker.execution_type_list,
            tracker.execution_duration_list,
        ):
            if exec_type == BatchExecutionType.DECODE:
                B = inf_reqs
                K = inf_toks
                if B > 0 and K > 0:
                    B_list.append(B)
                    K_list.append(K)
                    T_list.append(duration)

        if len(T_list) < 3:
            raise ValueError("Not enough DECODE batches to fit DecodeExecutionEstimator (need ≥3).")

        X = np.column_stack([B_list, K_list, np.ones_like(T_list)])
        delta, epsilon, d = self._linfit(X, T_list)

        self._params = DecodeParams(delta=float(delta), epsilon=float(epsilon), d=float(d))
        preds = X @ np.array([delta, epsilon, d])
        self.fit_rmse = float(np.sqrt(np.mean((preds - np.array(T_list)) ** 2)))
        return self._params