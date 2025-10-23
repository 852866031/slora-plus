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
        if self.size()%32 == 0 and self.size() > self.last_refit_count:
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
    a: Optional[float] = None  # per-token cost
    b: Optional[float] = None  # per-request cost
    d: Optional[float] = None  # fixed overhead per batch
    c: Optional[float] = None  # extra per-token cost for FT


class PrefillExecutionEstimator:
    """
    Execution time model (prefill):
        1) Inference-only:
            T_inf = a*N_inf + b*B + d
        2) Co-serving:
            T_co  = a*(N_inf+N_ft) + b*B + d + c*N_ft
        a, b, c, d are fitted parameters.
        N_inf: total inference tokens in batch
        N_ft: total finetuning tokens in batch
        B: batch size (number of requests)
    """

    def __init__(self) -> None:
        self._params = PrefillParams()
        self.inf_err = None  # fitting error (RMSE)
        self.co_err = None  # fitting error (RMSE)

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
        coserving_reqs: Sequence[float],                 # total reqs in co-serving batches
        coserving_times: Sequence[float],
        enforce_ratio: bool = True,
    ) -> PrefillParams:
        """
        Fit (a,b,d) using all inference-only points and then fit c using all co-serving points.
        """
        Ni_inf = self._as_np_1d(inference_tokens)
        Bi_inf = self._as_np_1d(inference_reqs)
        Ti_inf = self._as_np_1d(inference_times)
        co_pairs = list(coserving_tokens)
        Bi_co = self._as_np_1d(coserving_reqs)
        Ti_co = self._as_np_1d(coserving_times)

        if len(Ni_inf) != len(Ti_inf) or len(Ni_inf) != len(Bi_inf):
            raise ValueError("inference_tokens/inference_reqs/inference_times lengths mismatch")
        if len(co_pairs) != len(Ti_co) or len(co_pairs) != len(Bi_co):
            raise ValueError("coserving_tokens/requnums/times lengths mismatch")

        # Optional safety: N_inf > 5*N_ft
        if enforce_ratio and len(co_pairs) > 0:
            Ni = np.array([p[0] for p in co_pairs], dtype=float)
            Nf = np.array([p[1] for p in co_pairs], dtype=float)

        # --- Stage 1: fit (a,b,d) from inference-only ---
        a = b = d = c = None
        if len(Ni_inf) >= 3:  # at least 3 points for rank
            X_inf = np.column_stack([Ni_inf, Bi_inf, np.ones_like(Ni_inf)])
            a_b_d = self._linfit(X_inf, Ti_inf)
            a, b, d = float(a_b_d[0]), float(a_b_d[1]), float(a_b_d[2])

        # --- Stage 2: fit c using co-serving, or joint fit if needed ---
        if len(co_pairs) == 0:
            self._params = PrefillParams(a=a, b=b, d=d, c=None)
            return self._params

        Ni = np.array([p[0] for p in co_pairs], dtype=float)
        Nf = np.array([p[1] for p in co_pairs], dtype=float)
        Bc = Bi_co
        Tc = Ti_co

        if a is not None and b is not None and d is not None:
            # Two-stage: Y' = T - (a*(Ni+Nf) + b*B + d) = c * Nf
            y_c = Tc - (a * (Ni + Nf) + b * Bc + d)
            (c_,) = self._linfit(Nf.reshape(-1, 1), y_c)
            c = float(c_)
        else:
            # Fallback: jointly fit (a,b,d,c) from co-serving only
            if len(co_pairs) < 4:
                raise ValueError("Need >=3 inference points or >=4 co-serving points for full fit")
            S = (Ni + Nf).reshape(-1, 1)
            X_joint = np.column_stack([S, Bc, np.ones_like(S), Nf])
            if np.linalg.matrix_rank(X_joint) < 4:
                raise ValueError("Rank-deficient: vary N_inf+N_ft, B, N_ft")
            a_, b_, d_, c_ = self._linfit(X_joint, Tc)
            a, b, d, c = float(a_), float(b_), float(d_), float(c_)

        self._params = PrefillParams(a=a, b=b, d=d, c=c)
         # --- Error Tracking: max relative error ---
        if len(Ni_inf) > 0:
            preds_inf = a * Ni_inf + b * Bi_inf + d
            rel_errs_inf = (preds_inf - Ti_inf) / Ti_inf
            self.inf_err = float(np.max(rel_errs_inf))
        else:
            self.inf_err = None

        if len(co_pairs) > 0:
            preds_co = a * (Ni + Nf) + b * Bc + d + c * Nf
            rel_errs_co = (preds_co - Tc) / Tc
            self.co_err = float(np.max(rel_errs_co))
        else:
            self.co_err = None
        return self._params
    
    def data_fit(self, tracker: BatchExecutionTracker) -> PrefillParams:
        inference_tokens = []
        inference_reqs = []
        inference_times = []

        coserving_tokens = []  # list of (N_inf, N_ft)
        coserving_reqs = []
        coserving_times = []

        for ft_reqs, inf_reqs, ft_toks, inf_toks, exec_type, duration in zip(
            tracker.num_ft_reqs_list,
            tracker.num_inf_reqs_list,
            tracker.num_ft_tokens_list,
            tracker.num_inf_tokens_list,
            tracker.execution_type_list,
            tracker.execution_duration_list,
        ):
            total_reqs = ft_reqs + inf_reqs

            if exec_type == BatchExecutionType.PREFILL:
                if ft_reqs == 0:
                    # Inference-only batch
                    inference_tokens.append(inf_toks)
                    inference_reqs.append(inf_reqs)
                    inference_times.append(duration)
                else:
                    # Co-serving batch
                    coserving_tokens.append((inf_toks, ft_toks))
                    coserving_reqs.append(total_reqs)
                    coserving_times.append(duration)

        return self.fit(
            inference_tokens=inference_tokens,
            inference_reqs=inference_reqs,
            inference_times=inference_times,
            coserving_tokens=coserving_tokens,
            coserving_reqs=coserving_reqs,
            coserving_times=coserving_times,
        )

    def predict_inference(self, N_inf: float, B: float) -> float:
        if self._params.a is None or self._params.b is None or self._params.d is None:
            raise ValueError("Model not fitted yet")
        result = float(self._params.a * N_inf + self._params.b * B + self._params.d)
        if self.inf_err is not None:
            result *= (1.0 + 1.5 * self.inf_err)
        return result

    def predict_coserving(self, N_inf: float, N_ft: float, B: float) -> float:
        if self._params.a is None or self._params.b is None or self._params.d is None or self._params.c is None:
            raise ValueError("Model not fitted yet")
        result = float(self._params.a * (N_inf + N_ft) + self._params.b * B + self._params.d + self._params.c * N_ft)
        if self.co_err is not None:
            result *= (1.0 + 1.5 * self.co_err)
        return result

    def verify_inference(self, N_inf: float, B: float, actual_time: float) -> float:
        pred_time = self.predict_inference(N_inf, B)
        rel_err = abs(pred_time - actual_time) / actual_time
        print(f"[verify_inference] N_inf={N_inf}, B={B} => pred {pred_time:.3f}s vs actual {actual_time:.3f}s (err {rel_err:.2%})")
        if self.inf_err is not None:
            self.inf_err = max(self.inf_err, rel_err)
        else:
            self.inf_err = rel_err
        return rel_err

    def verify_coserving(self, N_inf: float, N_ft: float, B: float, actual_time: float) -> float:
        pred_time = self.predict_coserving(N_inf, N_ft, B)
        rel_err = abs(pred_time - actual_time) / actual_time
        print(f"[verify_coserving] N_inf={N_inf}, N_ft={N_ft}, B={B} => pred {pred_time:.3f}s vs actual {actual_time:.3f}s (err {rel_err:.2%})")
        if self.co_err is not None:
            self.co_err = max(self.co_err, rel_err)
        else:
            self.co_err = rel_err
        return rel_err
    
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
        if (self._params.a is None or
            self._params.b is None or
            self._params.c is None or
            self._params.d is None):
            raise ValueError("PrefillExecutionEstimator not fitted (need a,b,c,d).")

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
        if p.a is None or p.b is None or p.c is None or p.d is None:
            raise ValueError("PrefillExecutionEstimator not fitted (need a,b,c,d).")
        if earliest_req_time is None:
            return 10**12
        if now is None:
            now = time.time()
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
        co_err = float(self.co_err) if self.co_err is not None else 0.0
        safety = 1.0 + 2.0 * max(0.0, co_err)
        adjusted_budget = rem_time / safety
        N_inf = float(inf_token_num)
        N_ft_curr = float(ft_token_num)
        Bp = float(current_batch_size + 1)
        const_term = p.a * (N_inf + N_ft_curr) + p.c * N_ft_curr + p.b * Bp + p.d
        denom = p.a + p.c
        if denom <= 0:
            return 0
        x_cont = (adjusted_budget - const_term) / denom
        x_max = math.floor(x_cont)
        if x_max < 0:
            x_max = 0
        return int(x_max)

@dataclass
class DecodeParams:
    a: float = 0.0   # per (total_tokens * batch_size) cost
    b: float = 0.0   # per-request cost
    c: float = 0.0   # fixed overhead

class DecodeExecutionEstimator:
    """
    Execution time model (decode per step):

        T = a * (T_total) + b * B + c

        Where:
          - T_total = total tokens in batch (KV cache size)
          - B       = number of requests (new tokens to generate)
    """

    def __init__(self) -> None:
        self._params = DecodeParams()
        self.decode_err = None

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
        total_tokens: Sequence[float],  # T_total for each measurement
        batch_sizes: Sequence[float],   # B for each measurement
        times: Sequence[float],         # measured decode times
    ) -> DecodeParams:
        Ttot = self._as_np_1d(total_tokens)
        B = self._as_np_1d(batch_sizes)
        T = self._as_np_1d(times)

        # Model: T = a * (Ttot*B) + b * B + c
        X = np.column_stack([Ttot * B, B, np.ones_like(B)])
        a, b, c = self._linfit(X, T)

        self._params = DecodeParams(a=float(a), b=float(b), c=float(c))
        # --- compute max relative error ---
        preds = a * (Ttot * B) + b * B + c
        rel_errs = (preds - T) / np.maximum(T, 1e-9)
        self.decode_err = float(np.max(rel_errs))
        return self._params

    def predict(self, total_tokens: float, batch_size: float) -> float:
        a, b, c = self._params.a, self._params.b, self._params.c
        result = a * (total_tokens * batch_size) + b * batch_size + c
        if self.decode_err is not None:
            result *= (1.0 + 1.5 * self.decode_err)
        return result

    def verify(self, total_tokens: float, batch_size: float, actual_time: float) -> float:
        """Compare prediction against actual measurement. Returns relative error."""
        pred_time = self.predict(total_tokens, batch_size)
        rel_err = abs(pred_time - actual_time) / max(actual_time, 1e-9)
        print(f"[verify_decode] total_tokens={total_tokens}, B={batch_size} "
              f"=> pred {pred_time:.6f}s vs actual {actual_time:.6f}s "
              f"(error={rel_err:.2%})")
        if self.decode_err is not None:
            self.decode_err = max(self.decode_err, rel_err)
        else:
            self.decode_err = rel_err
        return rel_err
    
    def data_fit(self, tracker: BatchExecutionTracker) -> DecodeParams:
        """
        Fit the DecodeExecutionEstimator using data from a BatchExecutionTracker.

        Only batches with execution_type == BatchExecutionType.DECODE are used.
        The model uses (total_tokens_in_batch, num_requests, execution_time).
        """
        total_tokens = []
        batch_sizes = []
        times = []

        for num_ft_reqs, num_inf_reqs, num_ft_toks, num_inf_toks, exec_type, duration in zip(
            tracker.num_ft_reqs_list,
            tracker.num_inf_reqs_list,
            tracker.num_ft_tokens_list,
            tracker.num_inf_tokens_list,
            tracker.execution_type_list,
            tracker.execution_duration_list,
        ):
            if exec_type == BatchExecutionType.DECODE:
                # In decode, only inference tokens/reqs matter
                total_tokens.append(num_inf_toks)
                batch_sizes.append(num_inf_reqs)
                times.append(duration)

        if len(total_tokens) < 3:
            raise ValueError("Not enough decode batches to fit DecodeExecutionEstimator (need ≥3).")

        return self.fit(
            total_tokens=total_tokens,
            batch_sizes=batch_sizes,
            times=times,
        )