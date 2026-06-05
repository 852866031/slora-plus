"""Real LoRA-SFT backward for the sglang DeltaServe port.

Adapted from DSV-vLLM's `Llama3BackwardService.process_backward`. Runs
in-process (Path C) — sources base weights directly from the live
model_runner.model. The math layer (layer_forward / layer_backward /
attn_backward_core / head_backward) is unchanged from the vLLM port.

Per backward:
  1. Pull base weight views from model.layers[i] (q/k/v/o/gate/up/down/norms)
  2. Use fp32 LoRA master tensors for q/k/v/o (rank from FT adapter config)
  3. Compute head_backward → grad w.r.t. final residual
  4. Walk layers in reverse: layer_forward (populate cache) → layer_backward
  5. Accumulate grads into LoRA fp32 master; fused AdamW.step()
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from sglang.srt.deltaserve.bwd_services.llama3 import (
    head_backward, layer_forward, layer_backward, rope_cos_sin,
)
from sglang.srt.deltaserve.gpu_grant import gpu_grant

logger = logging.getLogger(__name__)


class _RealBackward:
    """Holds extracted base weights + fp32 LoRA master tensors + optimizer
    for the duration of the process. Constructed once at model_runner.load_model
    end; called from faux_backward.run_faux_backward when real-bwd is enabled."""

    def __init__(self, model: Optional[nn.Module] = None, lora_rank: int = 16,
                 lora_alpha: float = 32.0, lr: float = 5e-6,
                 weight_decay: float = 0.01, backward_fp32: bool = False,
                 state: Optional[Dict[str, Any]] = None,
                 device: Optional[str] = None) -> None:
        # Two construction paths share the same tail (LoRA masters + optimizer):
        #   - in-process (Path C): `model` is the live sglang model; base weights
        #     are zero-copy VIEWS into it.
        #   - subprocess (S12a): `state` is a portable dict produced by
        #     extract_base_state() (config + base weights as tensors); we move
        #     them to `device`. Same layout, so the verified backward is byte-
        #     identical — no HF re-load / RoPE-permutation risk.
        assert (model is None) ^ (state is None), "pass exactly one of model/state"
        self.model = model
        self.lora_rank = int(lora_rank)
        self.scaling = float(lora_alpha) / float(lora_rank)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.save_attn_qkv = False  # we DON'T capture qh/kh/vh, recompute via layer_forward

        if model is not None:
            self._init_from_model(model)
        else:
            self._init_from_state(state, device or "cuda")
        self.dims = (self.Hq, self.Hkv, self.Hd, self.kv_size)
        self.bwd_dtype = torch.float32 if backward_fp32 else self.base_dtype

        self._build_lora_and_optim()

    def _set_config(self, c: Dict[str, Any]) -> None:
        self.D = int(c["hidden_size"]); self.L = int(c["num_hidden_layers"])
        self.Hq = int(c["num_attention_heads"])
        self.Hkv = int(c.get("num_key_value_heads") or self.Hq)
        self.Hd = int(c.get("head_dim") or (self.D // self.Hq))
        self.kv_size = self.Hkv * self.Hd
        self.q_size = self.Hq * self.Hd
        self.inter = int(c["intermediate_size"])
        self.theta = float(c.get("rope_theta") or 10000.0)
        self.eps = float(c.get("rms_norm_eps") or 1e-5)
        self.vocab = int(c["vocab_size"])

    def _init_from_model(self, model: nn.Module) -> None:
        cfg = model.config if hasattr(model, "config") else getattr(model, "model").config
        def _cfg(name, default=None):
            v = getattr(cfg, name, None)
            if v is None and hasattr(cfg, "to_dict"):
                v = cfg.to_dict().get(name, default)
            return v if v is not None else default
        self._set_config({
            "hidden_size": _cfg("hidden_size"), "num_hidden_layers": _cfg("num_hidden_layers"),
            "num_attention_heads": _cfg("num_attention_heads"),
            "num_key_value_heads": _cfg("num_key_value_heads", _cfg("num_attention_heads")),
            "head_dim": _cfg("head_dim", None), "intermediate_size": _cfg("intermediate_size"),
            "rope_theta": _cfg("rope_theta", 10000.0), "rms_norm_eps": _cfg("rms_norm_eps", 1e-5),
            "vocab_size": _cfg("vocab_size"),
        })
        inner = model.model if hasattr(model, "model") and hasattr(model.model, "layers") else model
        self.layers = inner.layers
        self.norm_w = inner.norm.weight
        self.lm_w = model.lm_head.weight
        self.base_dtype = self.lm_w.dtype
        # Slice fused weights once into per-layer views (no copy).
        self.base = [self._slice_layer(self.layers[i]) for i in range(self.L)]

    def _slice_layer(self, lyr) -> Dict[str, torch.Tensor]:
        attn = lyr.self_attn; mlp = lyr.mlp
        qkv_w = getattr(attn.qkv_proj, "base_layer", attn.qkv_proj).weight
        gate_up_w = getattr(mlp.gate_up_proj, "base_layer", mlp.gate_up_proj).weight
        o_w = getattr(attn.o_proj, "base_layer", attn.o_proj).weight
        down_w = getattr(mlp.down_proj, "base_layer", mlp.down_proj).weight
        return {
            "q": qkv_w[:self.q_size], "k": qkv_w[self.q_size:self.q_size + self.kv_size],
            "v": qkv_w[self.q_size + self.kv_size:], "o": o_w,
            "gate": gate_up_w[:self.inter], "up": gate_up_w[self.inter:], "down": down_w,
            "in_ln": lyr.input_layernorm.weight, "post_ln": lyr.post_attention_layernorm.weight,
        }

    def _init_from_state(self, state: Dict[str, Any], device: str) -> None:
        self._set_config(state["config"])
        self.layers = None
        self.base_dtype = getattr(torch, str(state.get("base_dtype", "bfloat16")).split(".")[-1])
        self.norm_w = state["norm_w"].to(device)
        self.lm_w = state["lm_w"].to(device)
        self.base = [{k: v.to(device) for k, v in layer.items()} for layer in state["base"]]

    def _build_lora_and_optim(self) -> None:
        # Build fp32 LoRA master tensors per layer × {q,k,v,o} × {A,B}.
        # LoRA A: [rank, D]; B: [out_dim, rank] (out_dim = q_size for q/o, kv_size for k/v).
        device = self.lm_w.device
        params: List[nn.Parameter] = []
        self.lora: List[Dict[str, Dict[str, nn.Parameter]]] = []
        for i in range(self.L):
            ld: Dict[str, Dict[str, nn.Parameter]] = {}
            for proj, out_dim in [("q", self.q_size), ("o", self.q_size),
                                  ("k", self.kv_size), ("v", self.kv_size)]:
                A = nn.Parameter(torch.randn(self.lora_rank, self.D, device=device,
                                              dtype=torch.float32) * 0.01)
                B = nn.Parameter(torch.zeros(out_dim, self.lora_rank, device=device,
                                              dtype=torch.float32))
                ld[proj] = {"A": A, "B": B}
                params.append(A); params.append(B)
            self.lora.append(ld)

        self.optimizer = torch.optim.AdamW(params, lr=self.lr,
                                            weight_decay=self.weight_decay, fused=True)
        # Dedicated CUDA stream so backward kernels can OVERLAP with the
        # next forward batch (concurrent execution on the GPU, single Python
        # thread). Cheaper alternative to MPS subprocess isolation; gets
        # ~50% of the win for ~5% of the integration cost.
        self.bwd_stream = torch.cuda.Stream(device=device)
        self._call_count = 0
        self._cum_loss = 0.0
        self._cum_tokens = 0
        self.publish_enabled = False   # §13: apply trained LoRA in inference forward
        self._infer_hooks = []
        logger.warning(
            f"[DeltaServe] real_backward state built: D={self.D} L={self.L} "
            f"Hq={self.Hq} Hkv={self.Hkv} Hd={self.Hd} inter={self.inter} "
            f"vocab={self.vocab} lora_rank={self.lora_rank} "
            f"params={sum(p.numel() for p in params)/1e6:.2f}M"
        )

    # --- §13 served-LoRA publish ------------------------------------------- #
    def attach_inference_hooks(self) -> None:
        """Apply the trained LoRA delta in the MODEL's inference forward via
        forward-hooks on each layer's qkv_proj + o_proj, reading the live fp32
        masters every call so training continuously affects serving. Base
        weights stay frozen — the backward's rematerialization uses `self.base`
        views and the math layer's explicit weights, NOT these modules' outputs,
        so the two paths don't interfere. In-place param updates from
        optimizer.step propagate even through captured CUDA graphs (the matmul
        kernels reference the A/B tensor memory)."""
        if self.layers is None:
            logger.warning("[DeltaServe] §13 publish: no model handle (state-built) — hooks skipped")
            return
        if self._infer_hooks:
            return
        for i in range(self.L):
            attn = self.layers[i].self_attn
            self._infer_hooks.append(attn.qkv_proj.register_forward_hook(self._make_qkv_hook(i)))
            self._infer_hooks.append(attn.o_proj.register_forward_hook(self._make_o_hook(i)))
        self.publish_enabled = True
        logger.warning(f"[DeltaServe] §13 publish ON: {len(self._infer_hooks)} LoRA "
                       f"inference hooks attached ({self.L} layers × qkv+o)")

    def _delta(self, x, A, B):
        # scaling·(x @ Aᵀ) @ Bᵀ ; x any dtype, A[r,D]/B[out,r] fp32 → x.dtype
        d = (x.float() @ A.t()) @ B.t()
        return (self.scaling * d).to(x.dtype)

    def _make_qkv_hook(self, i: int):
        ld = self.lora[i]; qs, ks = self.q_size, self.kv_size
        def hook(module, inp, out):
            if not self.publish_enabled:
                return None
            x = inp[0]
            t = out[0] if isinstance(out, tuple) else out
            t[..., :qs] += self._delta(x, ld["q"]["A"], ld["q"]["B"])
            t[..., qs:qs + ks] += self._delta(x, ld["k"]["A"], ld["k"]["B"])
            t[..., qs + ks:qs + 2 * ks] += self._delta(x, ld["v"]["A"], ld["v"]["B"])
            return out
        return hook

    def _make_o_hook(self, i: int):
        ld = self.lora[i]
        def hook(module, inp, out):
            if not self.publish_enabled:
                return None
            x = inp[0]
            t = out[0] if isinstance(out, tuple) else out
            t += self._delta(x, ld["o"]["A"], ld["o"]["B"])
            return out
        return hook

    def _layer_weights(self, i: int) -> dict:
        lw = dict(self.base[i])
        ld = self.lora[i]
        for proj in ("q", "k", "v", "o"):
            lw[proj + "A"] = ld.get(proj, {}).get("A")
            lw[proj + "B"] = ld.get(proj, {}).get("B")
        return lw

    def process(self, snapshot: Dict[str, Any], sample_lens: List[int]) -> float:
        """Run one real backward over the captured activations. Returns
        elapsed seconds. snapshot is FinetuneAccumulator.pop_step() output."""
        if not snapshot.get("layer_in"):
            return 0.0
        final_in = snapshot.get("final_in")
        if final_in is None:
            return 0.0
        n = final_in.shape[0]
        # Skip decode-step fires that have no shift-by-1 targets (n_valid would
        # be 0). head_backward needs at least 2 tokens per sample to compute
        # one CE term. For 1-token decode steps the backward is pure waste —
        # ~37ms × hundreds of decode steps per FT request adds up.
        if n < 2:
            return 0.0

        # If sample_lens not provided, treat as single sample of length n.
        if not sample_lens or sum(sample_lens) != n:
            sample_lens = [n]
        b_start, acc = [], 0
        for s in sample_lens:
            b_start.append(acc); acc += s

        device = final_in.device
        positions = torch.cat([torch.arange(s, device=device) for s in sample_lens])
        cos, sin = rope_cos_sin(positions, self.Hd, self.theta)
        ids = snapshot.get("concat_input_ids")
        if ids is None:
            return 0.0
        ids = ids[:n].to(device).long()

        grant = gpu_grant()
        # Wait for any IN-FLIGHT backward to finish before enqueuing the next
        # one (so we don't pile up CUDA memory). The bwd stream is serial
        # internally; this prevents two backwards' tensors from coexisting.
        self.bwd_stream.synchronize()
        self.optimizer.zero_grad(set_to_none=True)

        t_kick = time.monotonic()
        # Enqueue entire backward on the bwd stream. The Python call returns
        # immediately after enqueue; GPU runs concurrently with the next
        # forward on the default stream.
        with torch.cuda.stream(self.bwd_stream):
            loss, n_valid, g = head_backward(
                final_in[:n], self.lm_w, self.norm_w, self.eps,
                ids, sample_lens, b_start, self.vocab,
            )
            for i in range(self.L - 1, -1, -1):
                grant.maybe_pause()
                x_in = snapshot["layer_in"].get(i)
                if x_in is None:
                    continue
                x_in = x_in[:n]
                lw = self._layer_weights(i)
                with torch.no_grad():
                    cache = layer_forward(
                        x_in, lw, self.scaling, cos, sin, sample_lens, b_start,
                        self.dims, self.eps,
                    )
                grad_x, grads = layer_backward(
                    g, cache, lw, self.scaling, cos, sin, sample_lens, b_start,
                    self.dims, self.eps, cdt=self.bwd_dtype,
                )
                ld = self.lora[i]
                for proj in ("q", "k", "v", "o"):
                    gA = grads.get(proj + "A"); gB = grads.get(proj + "B")
                    if gA is not None and ld[proj]["A"] is not None:
                        pA = ld[proj]["A"]
                        pA.grad = (pA.grad + gA.float()) if pA.grad is not None else gA.float()
                    if gB is not None and ld[proj]["B"] is not None:
                        pB = ld[proj]["B"]
                        pB.grad = (pB.grad + gB.float()) if pB.grad is not None else gB.float()
                g = grad_x

            self.optimizer.step()

        # Don't sync — return immediately. The next forward batch on default
        # stream proceeds and overlaps with this backward on bwd_stream.
        # Elapsed measured here is enqueue cost only (~ms), not real backward time.
        elapsed = time.monotonic() - t_kick
        self._call_count += 1
        self._cum_loss += float(loss)
        self._cum_tokens += int(n_valid)
        logger.warning(
            f"[DeltaServe] real_backward #{self._call_count}: {elapsed*1000:.1f}ms "
            f"loss={loss:.4f} n_valid={n_valid} cum_tokens={self._cum_tokens}"
        )
        return elapsed


def extract_base_state(model: nn.Module) -> Dict[str, Any]:
    """Produce a portable, process-independent snapshot of the frozen base
    weights + config, for the S12a backward subprocess. Weights are cloned to
    CPU (so they survive `torch.save` to /dev/shm and a load in the child).
    Layout is identical to the in-process VIEW path, so the child's backward is
    byte-for-byte the same math — no HF re-load / RoPE-permutation risk.

    Builds a throwaway _RealBackward(model) only to reuse its exact slicing;
    we copy out `base/norm_w/lm_w` and drop everything else."""
    rb = _RealBackward(model=model)
    config = {
        "hidden_size": rb.D, "num_hidden_layers": rb.L,
        "num_attention_heads": rb.Hq, "num_key_value_heads": rb.Hkv,
        "head_dim": rb.Hd, "intermediate_size": rb.inter,
        "rope_theta": rb.theta, "rms_norm_eps": rb.eps, "vocab_size": rb.vocab,
    }
    base = [{k: v.detach().to("cpu", copy=True) for k, v in layer.items()}
            for layer in rb.base]
    return {
        "config": config,
        "base": base,
        "norm_w": rb.norm_w.detach().to("cpu", copy=True),
        "lm_w": rb.lm_w.detach().to("cpu", copy=True),
        "base_dtype": str(rb.base_dtype),
    }


_INSTANCE: Optional[_RealBackward] = None


def build_real_backward(model: nn.Module, **kwargs) -> _RealBackward:
    """Module-level singleton accessor (in-process Path C)."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = _RealBackward(model=model, **kwargs)
    return _INSTANCE


def build_real_backward_from_state(state: Dict[str, Any], device: str = "cuda",
                                   **kwargs) -> _RealBackward:
    """Construct the backward in a subprocess from extract_base_state() output."""
    return _RealBackward(model=None, state=state, device=device, **kwargs)


def get_real_backward() -> Optional[_RealBackward]:
    return _INSTANCE


def is_enabled() -> bool:
    """Env var SGLANG_DS_REAL_BACKWARD=1 switches faux → real."""
    return os.environ.get("SGLANG_DS_REAL_BACKWARD", "0") == "1"
