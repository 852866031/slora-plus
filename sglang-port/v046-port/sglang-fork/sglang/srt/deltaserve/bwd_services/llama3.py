# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Llama-3 backward service — manual LoRA SFT backward (Phase 3).

Trains the FT LoRA adapter (attention-only q/k/v/o) in the backward subprocess
from the captured residual-stream activations, while inference keeps serving.
Mirrors DeltaServe `models/{llama,llama3}/SFT_service.py` but adapted to vLLM:

- PEFT-separate LoRA (`lora_A [r,in]`, `lora_B [out,r]`, delta = scaling·(x@Aᵀ)@Bᵀ)
  rather than DeltaServe's packed `[2,4r,H,Hd]`; grads derived directly in PEFT layout.
- vLLM's **fused** base weights (`qkv_proj`/`gate_up_proj`) — sliced once at setup.
- vLLM RoPE (NeoX rotate-half; cos/sin rebuilt from `inv_freq=1/theta^(2i/d)`).
- No score clamp (vLLM doesn't clamp).

We capture only the per-layer residual-stream **input** (`layer_in[i]`) + `final_in`,
so each layer's forward is **rematerialized** from `layer_in[i]` to recover the
intermediates the manual backward needs, then the gradients are computed by hand.

Precision (see `deltaserve-backward-precision` memory): scores/softmax/RMSNorm/LM-head
in fp32; bulk projections in the weights' dtype; fp32 LoRA master / optimizer. The
manual backward runs in fp32. MLP/embeddings/norms are frozen — only the 8 LoRA
tensors per layer get gradients.
"""

import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging as _ds_log; dprint = lambda *a, **kw: _ds_log.getLogger(__name__).info(" ".join(str(x) for x in a))
from sglang.srt.deltaserve.bwd_services.base import BackwardService

_TOL = 5e-2  # bf16 round-trip tolerance for the capture/forward-fidelity checks
_VOCAB_CHUNK = 16384


# --------------------------------------------------------------------------- #
# Math helpers (module-level so the gradcheck test can call them directly).
# Everything here operates in the dtype of its inputs, except scores/softmax/
# RMSNorm internals which upcast to fp32. The gradcheck passes fp32 throughout.
# --------------------------------------------------------------------------- #

def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float):
    """NeoX cos/sin for the given positions. Returns (cos, sin) [n, head_dim//2] fp32."""
    inv_freq = 1.0 / (theta ** (
        torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device)
        / head_dim))
    freqs = torch.outer(positions.float(), inv_freq)   # [n, head_dim//2]
    return freqs.cos(), freqs.sin()


def apply_rope(xh: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """NeoX rotate-half on [n, H, head_dim]; cos/sin [n, head_dim//2]."""
    h = xh.shape[-1] // 2
    x1, x2 = xh[..., :h], xh[..., h:]
    c, s = cos[:, None, :].to(xh.dtype), sin[:, None, :].to(xh.dtype)
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def rope_backward(g: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Inverse of apply_rope (transpose of the rotation)."""
    h = g.shape[-1] // 2
    g1, g2 = g[..., :h], g[..., h:]
    c, s = cos[:, None, :].to(g.dtype), sin[:, None, :].to(g.dtype)
    return torch.cat([g1 * c + g2 * s, -g1 * s + g2 * c], dim=-1)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float):
    """RMSNorm forward (fp32 internal), output cast back to x.dtype."""
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return ((xf * inv) * w.float()).to(x.dtype)


def rmsnorm_backward(x: torch.Tensor, grad_y: torch.Tensor, w: torch.Tensor,
                     eps: float):
    """Exact gradient of y = rmsnorm(x)·w w.r.t. x (fp32)."""
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)   # 1/rms
    xhat = xf * inv
    g_xhat = grad_y.float() * w.float()
    d = xf.shape[-1]
    dot = (g_xhat * xhat).sum(-1, keepdim=True)
    return (g_xhat - xhat * dot / d) * inv


def _logits_chunked(h: torch.Tensor, lm_w: torch.Tensor, vocab: int):
    """fp32 logits = h @ lm_w[:vocab].T, computed in vocab chunks (bounds the fp32
    LM-head temporary). h [m, D], lm_w [vocab_pad, D]. Returns [m, vocab] fp32."""
    hf = h.float()
    out = hf.new_empty((hf.shape[0], vocab))
    for c in range(0, vocab, _VOCAB_CHUNK):
        e = min(c + _VOCAB_CHUNK, vocab)
        out[:, c:e] = hf @ lm_w[c:e].float().t()
    return out


def _proj(xin, w_base, A, B, scaling):
    """y = xin @ w_base.T + scaling·(xin @ A.T) @ B.T. w_base [out,in], A [r,in], B [out,r].

    The LoRA master is fp32; cast its compute copy to xin's dtype (bf16 in prod,
    fp32 in the gradcheck) so the matmul dtypes match (DeltaServe fp32-master /
    low-precision-compute rule)."""
    y = F.linear(xin, w_base)
    if A is not None:
        y = y + scaling * F.linear(F.linear(xin, A.to(xin.dtype)), B.to(xin.dtype))
    return y


def _proj_backward(xin, gy, w_base, A, B, scaling, cdt=torch.float32):
    """Grad of _proj, computed in the bulk compute dtype ``cdt`` (bf16 in prod,
    fp32 in the gradcheck). Returns (grad_xin, grad_A, grad_B).

    Casts the weights inside the function on every call. This is intentional
    and matches DeltaServe's precision contract (bf16 compute for projection
    backward, fp32 LoRA master).

    Why we don't amortize the casts: DeltaServe packs all 8 LoRA tensors per
    layer into a single ``[2, 4r, H, Hd]`` buffer cast once at the layer
    boundary (SFT_service.py:424). We use separate qA/qB/… tensors so the
    cast count is the cardinality of distinct tensors (7 base + 8 LoRA = 15),
    and that's the same whether we cast inline or pre-cast. The real-cast
    count is also mode-symmetric:
      - bf16 cdt (default): 7 base ``.to(bf16)`` are no-ops, 8 LoRA fp32→bf16
        are real casts. Total: 8 real casts/layer.
      - fp32 cdt (``backward_fp32=True``, rare): 7 base bf16→fp32 are real
        casts (heavier — base tensors are big, e.g. ``down`` is [D, inter]),
        8 LoRA ``.to(fp32)`` are no-ops. Total: 7 real casts/layer.
    Amortizing wouldn't change either count. Keep this layout."""
    xin = xin.to(cdt)
    gy = gy.to(cdt)
    grad_xin = gy @ w_base.to(cdt)
    grad_A = grad_B = None
    if A is not None:
        Ac, Bc = A.to(cdt), B.to(cdt)
        Z = xin @ Ac.t()                       # [n, r]
        grad_Z = scaling * (gy @ Bc)           # [n, r]
        grad_A = grad_Z.t() @ xin              # [r, in]
        grad_B = scaling * (gy.t() @ Z)        # [out, r]
        grad_xin = grad_xin + grad_Z @ Ac
    return grad_xin, grad_A, grad_B


def layer_forward(x, lw, scaling, cos, sin, seq_lens, b_start, dims, eps,
                  saved_gate_up=None, saved_qh=None, saved_kh=None, saved_vh=None):
    """Rematerialize one Llama decoder layer forward, returning the ``cache`` the
    manual backward needs (attention internals + MLP pre-activations). The frozen
    `down` matmul / layer output are NOT computed — the backward only needs the
    incoming gradient, not the layer output (we already saved the next layer's input).

    When ``saved_gate_up`` ([n, 2*intermediate] = gate||up, captured in the forward)
    is given, the MLP `gate_up` matmul is skipped entirely — the biggest recompute in
    the layer. Otherwise gate/up are recomputed (the gradcheck path).

    When ``saved_qh/kh/vh`` are given (each [n, q_size or kv_size] flat,
    captured by ``self_attn.attn`` forward_pre_hook), we ALSO skip Q/K/V proj
    + RoPE — the second-biggest recompute. ``x_norm1`` is still computed from
    ``x`` (cheap; the Q/K/V LoRA-A grad needs it as ``grad_A = grad_Z.t() @
    x_norm1``). ``cos/sin`` are unused on this fast path. If any of the three
    saved tensors is absent, the recompute path runs.

    ``lw`` is a dict of base weights (q,k,v,o,gate,up,in_ln,post_ln) + LoRA
    (qA,qB,kA,kB,vA,vB,oA,oB). Functional (no in-place) so autograd can differentiate
    it for the gradcheck. ``dims`` = (Hq, Hkv, Hd, kv_size)."""
    Hq, Hkv, Hd, kv_size = dims
    n = x.shape[0]
    D = Hq * Hd
    scale = 1.0 / math.sqrt(Hd)
    kv_repeat = Hq // Hkv

    x_norm1 = rmsnorm(x, lw["in_ln"], eps)
    if (saved_qh is not None
            and saved_kh is not None
            and saved_vh is not None):
        # Skip Q/K/V proj + RoPE — use the saved post-RoPE flat tensors.
        qh = saved_qh.view(n, Hq, Hd)
        kh = saved_kh.view(n, Hkv, Hd)
        vh = saved_vh.view(n, Hkv, Hd)
    else:
        q = _proj(x_norm1, lw["q"], lw["qA"], lw["qB"], scaling)        # [n, D]
        k = _proj(x_norm1, lw["k"], lw["kA"], lw["kB"], scaling)        # [n, kv_size]
        v = _proj(x_norm1, lw["v"], lw["vA"], lw["vB"], scaling)
        qh = apply_rope(q.view(n, Hq, Hd), cos, sin)
        kh = apply_rope(k.view(n, Hkv, Hd), cos, sin)
        vh = v.view(n, Hkv, Hd)

    ctx_blocks = []
    for st, ln in zip(b_start, seq_lens):
        q_blk = qh[st:st + ln].transpose(0, 1)      # [Hq, L, Hd]
        k_blk = kh[st:st + ln].transpose(0, 1)      # [Hkv, L, Hd]
        v_blk = vh[st:st + ln].transpose(0, 1)
        if kv_repeat != 1:
            k_rep = k_blk.repeat_interleave(kv_repeat, 0)
            v_rep = v_blk.repeat_interleave(kv_repeat, 0)
        else:
            k_rep, v_rep = k_blk, v_blk
        scores = (q_blk.float() @ k_rep.float().transpose(-1, -2)) * scale
        mask = torch.triu(torch.ones(ln, ln, dtype=torch.bool, device=x.device), 1)
        scores = scores.masked_fill(mask, -1e9)
        att = torch.softmax(scores, dim=-1)         # [Hq, L, L] fp32
        ctx_blk = (att @ v_rep.float()).to(x.dtype).transpose(0, 1)  # [L, Hq, Hd]
        ctx_blocks.append(ctx_blk)
    ctx_flat = torch.cat(ctx_blocks, 0).reshape(n, D)

    o = _proj(ctx_flat, lw["o"], lw["oA"], lw["oB"], scaling)       # [n, D]
    resid_mid = x + o

    if saved_gate_up is not None:
        # Skip the gate_up matmul — use the saved forward pre-activations.
        inter = saved_gate_up.shape[-1] // 2
        gate = saved_gate_up[:, :inter]
        up = saved_gate_up[:, inter:]
    else:
        x_norm2 = rmsnorm(resid_mid, lw["post_ln"], eps)
        gate = F.linear(x_norm2, lw["gate"])
        up = F.linear(x_norm2, lw["up"])

    return {"x": x, "x_norm1": x_norm1, "qh": qh, "kh": kh, "vh": vh,
            "ctx_flat": ctx_flat, "resid_mid": resid_mid,
            "gate": gate, "up": up}


def ffn_backward_core(grad_out, cache, lw, eps, cdt=torch.float32):
    """FFN-block backward (frozen MLP, with residual): returns grad_resid_mid.

    Captures the shape-stable bulk of the per-layer backward: cast-in, FFN
    silu/sigmoid + 3 GEMMs against frozen down/gate/up, rmsnorm-post_ln backward,
    plus the residual add. Inputs are all sized by the chained ``grad_out``
    (``[n, D]``) and ``cache`` slices captured at the same n. Pure function over
    its inputs so it can be wrapped 1:1 by a CUDA graph at fixed n=s_max."""
    gout = grad_out.to(cdt)
    gate = cache["gate"].to(cdt)
    up = cache["up"].to(cdt)
    sig = torch.sigmoid(gate)
    silu = gate * sig
    silu_grad = sig * (1.0 + gate * (1.0 - sig))
    grad_h_mid = gout @ lw["down"].to(cdt)           # [n, inter]
    grad_up = grad_h_mid * silu
    grad_gate = grad_h_mid * up * silu_grad
    grad_x_norm2 = grad_gate @ lw["gate"].to(cdt) + grad_up @ lw["up"].to(cdt)
    return rmsnorm_backward(cache["resid_mid"], grad_x_norm2,
                            lw["post_ln"], eps).to(cdt) + gout


def attn_backward_core(qh, kh, vh, grad_ctx, seq_lens, b_start, dims, cdt=torch.float32,
                       *, grad_qh_buf=None, grad_kh_buf=None, grad_vh_buf=None):
    """Per-sample GQA attention backward (eager). Returns (grad_qh, grad_kh,
    grad_vh) in flat ``[n, H_*, Hd]`` layout, in ``cdt``. Per-sample shapes
    (variable seq_lens) prevent CUDA-graph capture; the runner uses a padded
    variant for the graphed fast path and silently falls back to this when a
    batch overflows the padded bounds. Scores/softmax/dQ/dK/dV always run in
    fp32 (the load-bearing GQA precision rule).

    If ``grad_qh_buf/kh_buf/vh_buf`` are provided (persistent buffers sized at
    s_max), the function zeroes their first n rows in-place and returns views
    into them — saves L=32 fresh-allocation zero-fills per backward. Falls
    back to fresh allocation when not provided (preserves the gradcheck path,
    which has no service to own persistent buffers)."""
    Hq, Hkv, Hd, _ = dims
    scale = 1.0 / math.sqrt(Hd)
    kv_repeat = Hq // Hkv
    n = grad_ctx.shape[0]
    device = grad_ctx.device
    # Use the persistent buffers only when they're large enough for n. They're
    # sized at s_max for the production path (n ≤ s_max always); the runner's
    # eager-fallback inside Llama3GraphedBackward.attn_backward also passes
    # its own static_grad_* (also s_max). The undersized branch fires only in
    # contrived parity tests that exceed s_max — we silently alloc fresh.
    if grad_qh_buf is not None and grad_qh_buf.shape[0] >= n:
        grad_qh_buf[:n].zero_()
        grad_kh_buf[:n].zero_()
        grad_vh_buf[:n].zero_()
        grad_qh = grad_qh_buf[:n]
        grad_kh = grad_kh_buf[:n]
        grad_vh = grad_vh_buf[:n]
    else:
        grad_qh = torch.zeros((n, Hq, Hd), dtype=cdt, device=device)
        grad_kh = torch.zeros((n, Hkv, Hd), dtype=cdt, device=device)
        grad_vh = torch.zeros((n, Hkv, Hd), dtype=cdt, device=device)
    qh_f, kh_f, vh_f = qh.float(), kh.float(), vh.float()
    for st, ln in zip(b_start, seq_lens):
        q_blk = qh_f[st:st + ln].transpose(0, 1)     # [Hq, L, Hd] fp32
        k_blk = kh_f[st:st + ln].transpose(0, 1)     # [Hkv, L, Hd] fp32
        v_blk = vh_f[st:st + ln].transpose(0, 1)
        if kv_repeat != 1:
            k_rep = k_blk.repeat_interleave(kv_repeat, 0)
            v_rep = v_blk.repeat_interleave(kv_repeat, 0)
        else:
            k_rep, v_rep = k_blk, v_blk
        mask = torch.triu(torch.ones(ln, ln, dtype=torch.bool, device=device), 1)
        scores = (q_blk @ k_rep.transpose(-1, -2)) * scale   # fp32
        scores = scores.masked_fill(mask, -1e9)
        att = torch.softmax(scores, dim=-1)          # [Hq, L, L] fp32
        g = grad_ctx[st:st + ln].transpose(0, 1).float()     # fp32
        grad_att = g @ v_rep.transpose(-1, -2)       # [Hq, L, L]
        grad_v_rep = att.transpose(-1, -2) @ g       # [Hq, L, Hd]
        sm = (grad_att * att).sum(-1, keepdim=True)
        grad_scores = (att * (grad_att - sm)).masked_fill(mask, 0.0)
        grad_q_blk = (grad_scores @ k_rep) * scale
        grad_k_rep = (grad_scores.transpose(-1, -2) @ q_blk) * scale
        if kv_repeat != 1:
            grad_k_kv = grad_k_rep.view(Hkv, kv_repeat, ln, Hd).sum(1)
            grad_v_kv = grad_v_rep.view(Hkv, kv_repeat, ln, Hd).sum(1)
        else:
            grad_k_kv, grad_v_kv = grad_k_rep, grad_v_rep
        grad_qh[st:st + ln] = grad_q_blk.transpose(0, 1).to(cdt)
        grad_kh[st:st + ln] = grad_k_kv.transpose(0, 1).to(cdt)
        grad_vh[st:st + ln] = grad_v_kv.transpose(0, 1).to(cdt)
    return grad_qh, grad_kh, grad_vh


def layer_backward(grad_out, cache, lw, scaling, cos, sin, seq_lens, b_start,
                   dims, eps, cdt=torch.float32,
                   *, grad_qh_buf=None, grad_kh_buf=None, grad_vh_buf=None):
    """Manual gradient of one layer. Bulk matmuls (FFN-bwd, LoRA-grad, rope/proj-bwd)
    run in ``cdt`` (bf16 in prod, fp32 for the gradcheck); the **attention core**
    (scores/softmax-bwd/dQ/dK/dV) and **RMSNorm backward** always run in fp32 (the
    load-bearing precision rules — DeltaServe). Returns (grad_x, grads in cdt).
    MLP/norms frozen. Composes ``ffn_backward_core`` + ``attn_backward_core`` so
    the eager and graphed paths share one definition of the math.

    ``grad_{qh,kh,vh}_buf`` are optional persistent buffers passed through to
    ``attn_backward_core`` — used by the service to avoid 96 zero-fill kernel
    launches per backward. Not provided by the gradcheck test."""
    Hq, Hkv, Hd, kv_size = dims
    n = grad_out.shape[0]
    D = Hq * Hd

    # --- FFN backward (frozen MLP): grad w.r.t. resid_mid (incl. residual) ---
    grad_resid_mid = ffn_backward_core(grad_out, cache, lw, eps, cdt)

    # --- O backward (cdt) ---
    grad_ctx_flat, grad_oA, grad_oB = _proj_backward(
        cache["ctx_flat"], grad_resid_mid, lw["o"], lw["oA"], lw["oB"], scaling, cdt)

    # --- attention backward (per-sample, GQA): core in fp32, results back to cdt ---
    grad_ctx = grad_ctx_flat.view(n, Hq, Hd)
    grad_qh, grad_kh, grad_vh = attn_backward_core(
        cache["qh"], cache["kh"], cache["vh"], grad_ctx,
        seq_lens, b_start, dims, cdt,
        grad_qh_buf=grad_qh_buf, grad_kh_buf=grad_kh_buf, grad_vh_buf=grad_vh_buf)

    grad_q = rope_backward(grad_qh, cos, sin).reshape(n, D)
    grad_k = rope_backward(grad_kh, cos, sin).reshape(n, kv_size)
    grad_v = grad_vh.reshape(n, kv_size)

    # --- q/k/v projection backward (input = x_norm1), cdt ---
    xn1 = cache["x_norm1"]
    gx_q, grad_qA, grad_qB = _proj_backward(xn1, grad_q, lw["q"], lw["qA"], lw["qB"], scaling, cdt)
    gx_k, grad_kA, grad_kB = _proj_backward(xn1, grad_k, lw["k"], lw["kA"], lw["kB"], scaling, cdt)
    gx_v, grad_vA, grad_vB = _proj_backward(xn1, grad_v, lw["v"], lw["vA"], lw["vB"], scaling, cdt)
    grad_x_norm1 = gx_q + gx_k + gx_v

    # --- RMSNorm backward to x (fp32) + residual path ---
    grad_x = rmsnorm_backward(cache["x"], grad_x_norm1, lw["in_ln"], eps).to(cdt) + grad_resid_mid

    grads = {"qA": grad_qA, "qB": grad_qB, "kA": grad_kA, "kB": grad_kB,
             "vA": grad_vA, "vB": grad_vB, "oA": grad_oA, "oB": grad_oB}
    return grad_x, grads


def head_backward(final_in, lm_w, norm_w, eps, ids, seq_lens, b_start, vocab):
    """LM-head + final-norm: per-sample shift CE loss + grad w.r.t. final_in.
    Returns (loss: float, n_valid: int, grad_final_in [n,D] fp32)."""
    normed = rmsnorm(final_in, norm_w, eps)
    n = final_in.shape[0]
    logit_grad = normed.new_zeros((n, vocab), dtype=torch.float32)
    total_loss = normed.new_zeros((), dtype=torch.float32)
    n_valid = 0
    for st, ln in zip(b_start, seq_lens):
        if ln < 2:
            continue
        h = normed[st:st + ln - 1]                   # positions predict next
        lg = _logits_chunked(h, lm_w, vocab)         # [ln-1, vocab] fp32
        tgt = ids[st + 1:st + ln].long()
        total_loss = total_loss + F.cross_entropy(lg, tgt, reduction="sum")
        p = torch.softmax(lg, dim=-1)
        p[torch.arange(ln - 1, device=p.device), tgt] -= 1.0
        logit_grad[st:st + ln - 1] = p
        n_valid += int(ln - 1)
    if n_valid == 0:
        return 0.0, 0, final_in.new_zeros((n, final_in.shape[-1]), dtype=torch.float32)
    logit_grad /= n_valid
    # grad w.r.t. the post-final-norm hidden = logit_grad @ lm_w, chunked over vocab.
    grad_normed = logit_grad.new_zeros((n, final_in.shape[-1]))
    for c in range(0, vocab, _VOCAB_CHUNK):
        e = min(c + _VOCAB_CHUNK, vocab)
        grad_normed += logit_grad[:, c:e] @ lm_w[c:e].float()
    grad_final_in = rmsnorm_backward(final_in, grad_normed, norm_w, eps)
    return float(total_loss.item() / n_valid), n_valid, grad_final_in


# --------------------------------------------------------------------------- #
# Service
# --------------------------------------------------------------------------- #

_PROJ = ("q", "k", "v", "o")




# -- Subprocess stub (Phase 6) ----------------------------------------------
# The IPC subprocess path still uses this class; for Path C (in-process)
# we call the module-level math functions directly. Eventually this class
# should host the real recv loop (CUDA-IPC weight share + process_backward
# = the vLLM port), but for now it's just an echo stub so subprocess boots.

class Llama3BackwardService(BackwardService):
    """Phase 6 IPC stub. Math lives in the module-level functions; this
    class only exists to satisfy the subprocess entry point until the
    real recv loop is ported."""

    def step(self, activations):
        # Echo with act * 0.01 as a synthetic grad — same as the original stub.
        return {k: (v * 0.01) for k, v in activations.items()}

    def apply_grads(self, grad_dict):
        return None
