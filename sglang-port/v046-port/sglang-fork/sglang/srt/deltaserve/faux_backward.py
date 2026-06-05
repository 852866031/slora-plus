"""Faux backward (DeltaServe Path C placeholder).

DeltaServe accumulates FT activations into a buffer of size
`max_saved_finetuning_tokens` (typ. 256), then fires ONE backward over
the full s_max batch. Per-token decode hooks just append to the buffer;
backward only fires when full.

For Path C MVP we approximate that: each FT-bearing forward step adds
its actual token count to a counter; when the counter crosses
`_BACKWARD_TOKEN_THRESHOLD` (default 256), run backward-shaped work
sized for s_max=256 and reset.

The cost is roughly:
  - L layers × (FFN bwd + attn bwd + LoRA grad) at [s_max, D] shape
  - 1 LM-head bwd at [s_max, D] @ [D, vocab]
  - bf16 throughout (LM head fp32)
  - Per-layer _maybe_pause yield to inference

For Llama-3.2-1B (D=2048, inter=8192, L=16, vocab=128k) at s_max=256
this should take ~5-15ms on H200 (much lower than the ~50ms on A100
the doc cites, because H200 is ~3× faster).

For Llama-3-8B (D=4096, L=32) at s_max=256 → ~30-80ms on H200.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import torch

from sglang.srt.deltaserve.gpu_grant import gpu_grant

logger = logging.getLogger(__name__)

# DeltaServe FT buffer policy: only fire backward when the running token
# count reaches the threshold (matches `max_saved_finetuning_tokens=256`).
_BACKWARD_TOKEN_THRESHOLD = 256

# DeltaServe Section 4: SLO-aware admission throttling. After every faux
# backward fires, the next fire is gated by a minimum interval. This is a
# crude proxy for the real 6-param SLO estimator from the doc — it bounds
# FT throughput to keep inference TTFT/latency under budget. Configurable
# via SGLANG_DS_BACKWARD_MIN_INTERVAL_MS (default 0 = unthrottled).
import os as _os
_BACKWARD_MIN_INTERVAL_S = float(_os.environ.get("SGLANG_DS_BACKWARD_MIN_INTERVAL_MS", "0")) / 1000.0
_last_backward_fire_t = 0.0
_backward_skipped_for_slo = 0

_accumulated_tokens = 0
_backward_calls = 0
_backward_total_ms = 0.0

# Static buffers reused across calls (avoids alloc per backward).
_buffers: Dict[str, torch.Tensor] = {}


def _persist_stats(elapsed_ms: float, cuda_graph: bool = False, real: bool = False) -> None:
    """Write stats JSON to the gates dir so the http_server (separate proc)
    can read /finetune_stats."""
    import os, tempfile, json
    stats_path = os.path.join(tempfile.gettempdir(), "sglang_ds_gates", "faux_stats.json")
    try:
        with open(stats_path, "w") as f:
            json.dump({
                "backward_calls": _backward_calls,
                "backward_total_ms": _backward_total_ms,
                "last_call_ms": elapsed_ms,
                "skipped_for_slo": _backward_skipped_for_slo,
                "slo_min_interval_ms": _BACKWARD_MIN_INTERVAL_S * 1000,
                "cuda_graph": cuda_graph,
                "real": real,
            }, f)
    except OSError:
        pass


def get_stats() -> Dict[str, float]:
    """Telemetry: how many backwards fired + total wall time."""
    return {
        "backward_calls": _backward_calls,
        "backward_total_ms": _backward_total_ms,
        "pending_tokens": _accumulated_tokens,
        "skipped_for_slo": _backward_skipped_for_slo,
        "slo_min_interval_ms": _BACKWARD_MIN_INTERVAL_S * 1000,
    }


def reset_stats() -> None:
    global _backward_calls, _backward_total_ms, _accumulated_tokens
    _backward_calls = 0
    _backward_total_ms = 0.0
    _accumulated_tokens = 0


def _get_buf(name: str, shape: tuple, dtype, device) -> torch.Tensor:
    key = f"{name}:{shape}:{dtype}:{device}"
    buf = _buffers.get(key)
    if buf is None:
        buf = torch.zeros(shape, dtype=dtype, device=device)
        _buffers[key] = buf
    return buf


# Section 2: CUDA graph cache. First fire captures the matmul sequence
# into a CUDA graph; subsequent fires replay (no kernel launch overhead).
_BWD_GRAPH = None
_BWD_GRAPH_X = None  # static input buffer (we just zero-fill it; replay only
                     # needs shapes/addresses stable, not values)
_USE_BWD_GRAPH = _os.environ.get("SGLANG_DS_BWD_CUDA_GRAPH", "1") != "0"


def precapture_graph(model: torch.nn.Module) -> bool:
    """Capture the faux backward CUDA graph at startup. Returns True on
    success. Called from model_runner.load_model end so the one-time
    capture cost doesn't land during a live FT request."""
    global _BWD_GRAPH, _BWD_GRAPH_X
    if not _USE_BWD_GRAPH or _BWD_GRAPH is not None:
        return False
    try:
        layers = list(model.model.layers)
        L = len(layers)
        q_w = layers[0].self_attn.qkv_proj.weight
        D = q_w.shape[1]
        device = q_w.device
        dtype = q_w.dtype
    except Exception:
        return False
    inter_dim = 4 * D
    s = _BACKWARD_TOKEN_THRESHOLD
    r = 16

    try:
        _BWD_GRAPH_X = _get_buf("x_resid", (s, D), dtype, device)
        w_down = _get_buf("w_down", (D, inter_dim), dtype, device)
        w_attn = _get_buf("w_attn", (D, D), dtype, device)
        w_loraA = _get_buf("loraA", (D, r), dtype, device)
        w_loraB = _get_buf("loraB", (r, D), dtype, device)
        # Warmup pass
        with torch.no_grad():
            xw = _BWD_GRAPH_X
            for _ in range(L):
                g_y = xw @ w_down
                xw = g_y @ w_down.t()
                _ = xw @ w_down
                for _ in range(6):
                    xw = xw @ w_attn
                for _ in range(8):
                    z = xw @ w_loraA
                    _ = z @ w_loraB
        torch.cuda.synchronize()
        # Capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            with torch.no_grad():
                xw = _BWD_GRAPH_X
                for _ in range(L):
                    g_y = xw @ w_down
                    xw = g_y @ w_down.t()
                    _ = xw @ w_down
                    for _ in range(6):
                        xw = xw @ w_attn
                    for _ in range(8):
                        z = xw @ w_loraA
                        _ = z @ w_loraB
        _BWD_GRAPH = g
        logger.warning(f"[DeltaServe] backward CUDA graph PRE-CAPTURED at startup (s={s}, D={D}, L={L})")
        return True
    except Exception as e:
        logger.warning(f"[DeltaServe] precapture failed: {e}")
        return False


def _run_full_backward(model: torch.nn.Module) -> float:
    """Run a backward-shaped compute over s_max=256 tokens against full
    model dims. Heavy enough to be visible in benchmark numbers.

    Section 2: when SGLANG_DS_BWD_CUDA_GRAPH=1 (default), first call captures
    a CUDA graph over the matmul sequence; subsequent calls replay it.
    """
    global _backward_calls, _backward_total_ms, _BWD_GRAPH, _BWD_GRAPH_X
    grant = gpu_grant()
    try:
        layers = list(model.model.layers)
        L = len(layers)
        q_w = layers[0].self_attn.qkv_proj.weight
        D = q_w.shape[1]
        device = q_w.device
        dtype = q_w.dtype
    except Exception:
        return 0.0
    inter_dim = 4 * D  # SwiGLU intermediate ratio for Llama
    s = _BACKWARD_TOKEN_THRESHOLD
    r = 16

    # Section 2: CUDA graph replay path. Capture once; replay forever.
    if _USE_BWD_GRAPH and _BWD_GRAPH is not None:
        t0 = time.monotonic()
        _BWD_GRAPH.replay()
        torch.cuda.synchronize()
        elapsed_ms = (time.monotonic() - t0) * 1000
        _backward_calls += 1
        _backward_total_ms += elapsed_ms
        import os, tempfile, json
        try:
            stats_path = os.path.join(tempfile.gettempdir(), "sglang_ds_gates", "faux_stats.json")
            with open(stats_path, "w") as f:
                json.dump({
                    "backward_calls": _backward_calls,
                    "backward_total_ms": _backward_total_ms,
                    "last_call_ms": elapsed_ms,
                    "skipped_for_slo": _backward_skipped_for_slo,
                    "slo_min_interval_ms": _BACKWARD_MIN_INTERVAL_S * 1000,
                    "cuda_graph": True,
                }, f)
        except OSError:
            pass
        logger.warning(f"[DeltaServe] faux backward #{_backward_calls} GRAPH replay: {elapsed_ms:.1f}ms")
        return elapsed_ms

    t0 = time.monotonic()
    with torch.no_grad():
        x = _get_buf("x_resid", (s, D), dtype, device)
        # FFN bwd-shaped + attn bwd-shaped + LoRA per layer. The work is
        # sized to match a REAL DeltaServe LoRA-bwd at s_max=256 tokens on
        # this model — for Llama-3.2-1B that's ~30-50ms warm. Real per-layer
        # backward does: rmsnorm-bwd + 3 ffn matmuls + 4 attn matmuls +
        # 8 LoRA grad ops + rope-bwd + LM head bwd. Roughly 10 matmuls/layer
        # at FFN scale; we approximate with 3 FFN-scale + 6 attn-scale here.
        w_down = _get_buf("w_down", (D, inter_dim), dtype, device)
        w_attn = _get_buf("w_attn", (D, D), dtype, device)
        w_loraA = _get_buf("loraA", (D, r), dtype, device)
        w_loraB = _get_buf("loraB", (r, D), dtype, device)
        for _ in range(L):
            grant.maybe_pause()  # yield if inference is in prefill
            # FFN bwd: 3 matmuls at FFN scale (heaviest per-layer ops)
            grad_y = x @ w_down            # [s, inter]
            x = grad_y @ w_down.t()        # [s, D]
            _ = x @ w_down                 # 3rd
            # Attn bwd: 6 matmuls at hidden scale (q/k/v/o + scores)
            for _ in range(6):
                x = x @ w_attn
            # LoRA grads: 8 small matmuls per layer (q/k/v/o × A,B)
            for _ in range(8):
                z = x @ w_loraA
                _ = z @ w_loraB
        # LM head bwd (heaviest single op): [s, D] @ [D, vocab] in fp32
        # Plus the per-vocab-chunk fp32 cast — this is the precision-load-bearing
        # piece. Match DeltaServe's _logits_chunked at vocab=128k chunked /16k.
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            vocab_w = lm_head.weight  # [vocab, D]
            _ = x.float() @ vocab_w.float().t()
            # softmax + CE-like reduction (small but real)
            _ = _.softmax(-1).log()

    torch.cuda.synchronize()
    elapsed_ms = (time.monotonic() - t0) * 1000
    _backward_calls += 1
    _backward_total_ms += elapsed_ms

    # Section 2: capture the matmul sequence into a CUDA graph on the first
    # fire so subsequent fires replay (no kernel launch overhead). The cost
    # is one warmup pass + one capture pass + one replay = 3× this first
    # invocation; amortized over hundreds of subsequent fires.
    if _USE_BWD_GRAPH and _BWD_GRAPH is None:
        try:
            # Pre-allocate static IO outside the graph pool.
            _BWD_GRAPH_X = _get_buf("x_resid", (s, D), dtype, device)
            # Warmup: one full run to populate the caching allocator.
            with torch.no_grad():
                xw = _BWD_GRAPH_X
                w_down = _get_buf("w_down", (D, inter_dim), dtype, device)
                w_attn = _get_buf("w_attn", (D, D), dtype, device)
                w_loraA = _get_buf("loraA", (D, r), dtype, device)
                w_loraB = _get_buf("loraB", (r, D), dtype, device)
                for _ in range(L):
                    g_y = xw @ w_down
                    xw = g_y @ w_down.t()
                    _ = xw @ w_down
                    for _ in range(6):
                        xw = xw @ w_attn
                    for _ in range(8):
                        z = xw @ w_loraA
                        _ = z @ w_loraB
            torch.cuda.synchronize()
            # Capture.
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                with torch.no_grad():
                    xw = _BWD_GRAPH_X
                    w_down = _buffers[f"w_down:({D}, {inter_dim}):{dtype}:{device}"]
                    w_attn = _buffers[f"w_attn:({D}, {D}):{dtype}:{device}"]
                    w_loraA = _buffers[f"loraA:({D}, {r}):{dtype}:{device}"]
                    w_loraB = _buffers[f"loraB:({r}, {D}):{dtype}:{device}"]
                    for _ in range(L):
                        g_y = xw @ w_down
                        xw = g_y @ w_down.t()
                        _ = xw @ w_down
                        for _ in range(6):
                            xw = xw @ w_attn
                        for _ in range(8):
                            z = xw @ w_loraA
                            _ = z @ w_loraB
            _BWD_GRAPH = g
            logger.warning(f"[DeltaServe] faux backward CUDA graph CAPTURED (next fires will replay)")
        except Exception as e:
            logger.warning(f"[DeltaServe] cuda graph capture failed: {e} — falling back to eager")

    # Persist stats to a file so the HTTP server (separate process) can read them.
    import os, tempfile, json
    stats_path = os.path.join(tempfile.gettempdir(), "sglang_ds_gates", "faux_stats.json")
    try:
        with open(stats_path, "w") as f:
            json.dump({
                "backward_calls": _backward_calls,
                "backward_total_ms": _backward_total_ms,
                "last_call_ms": elapsed_ms,
                "skipped_for_slo": _backward_skipped_for_slo,
                "slo_min_interval_ms": _BACKWARD_MIN_INTERVAL_S * 1000,
                "cuda_graph": _BWD_GRAPH is not None,
            }, f)
    except OSError:
        pass
    logger.warning(
        f"[DeltaServe] faux backward #{_backward_calls} fired (eager): {elapsed_ms:.1f}ms"
    )
    return elapsed_ms


def run_faux_backward(snapshot: Dict[str, Any], model: torch.nn.Module) -> Optional[float]:
    """Accumulate captured FT tokens; fire backward when threshold reached.
    Returns the elapsed ms when a backward fires, None otherwise.

    Task A: when SGLANG_DS_REAL_BACKWARD=1, dispatch to real_backward instead
    of the synthetic matmul loop. Same buffering + same threshold."""
    global _accumulated_tokens
    layer_in = snapshot.get("layer_in") or {}
    if not layer_in:
        return None

    # Task A dispatch — replace faux with real LoRA backward when configured.
    try:
        from sglang.srt.deltaserve.real_backward import is_enabled, get_real_backward
        if is_enabled():
            rb = get_real_backward()
            if rb is not None:
                # Real backward uses real activations; threshold buffering is moot
                # because every fire is real (no need to amortize over 256 tokens
                # of synthetic work).
                t = rb.process(snapshot, sample_lens=[])
                if t > 0:
                    elapsed_ms = t * 1000
                    global _backward_calls, _backward_total_ms
                    _backward_calls += 1
                    _backward_total_ms += elapsed_ms
                    _persist_stats(elapsed_ms, cuda_graph=False, real=True)
                    return elapsed_ms
            return None
    except Exception as e:
        logger.warning(f"[DeltaServe] real_backward dispatch failed: {e} — falling back to faux")

    # Each layer_in entry has shape [n_ft_tokens, D]. They should all have
    # the same n; take it from the first entry.
    n_tokens = next(iter(layer_in.values())).shape[0]
    _accumulated_tokens += n_tokens

    if _accumulated_tokens >= _BACKWARD_TOKEN_THRESHOLD:
        # Section 4: SLO-aware throttling. If the previous backward was too
        # recent, defer this one — the buffer keeps growing but the inference
        # path gets its slice of the GPU back.
        global _last_backward_fire_t, _backward_skipped_for_slo
        if _BACKWARD_MIN_INTERVAL_S > 0:
            since = time.monotonic() - _last_backward_fire_t
            if since < _BACKWARD_MIN_INTERVAL_S:
                _backward_skipped_for_slo += 1
                return None  # defer; tokens stay accumulated
        _accumulated_tokens = 0
        _last_backward_fire_t = time.monotonic()
        return _run_full_backward(model)
    return None
