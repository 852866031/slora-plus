# SPDX-License-Identifier: Apache-2.0
"""Per-sample activation capture for finetuning-tagged requests.

When a forward batch contains FT-tagged samples mixed with inference samples,
this accumulator captures the FT rows of the residual stream entering each
decoder layer (``layer_in``) and the fused ``gate_up_proj`` output inside each
MLP (``mlp_gate_up``). Capture is gated on a per-row boolean ``finetune_mask``
set via ``begin_step``; ``pop_step`` snapshots and resets state for the next
forward.

Registration uses PyTorch's canonical ``module.register_forward_pre_hook`` /
``register_forward_hook`` directly — additive, not monkey-patching of
``forward``.

Section 7 fast path: when the mask True rows are contiguous, use a slice view
(zero-copy) instead of boolean mask indexing (index_select kernel + alloc).
Silent fallback to mask path when interleaved.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_LAYER_IN_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.input_layernorm$")
_GATE_UP_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate_up_proj$")
_FINAL_NORM_RE = re.compile(r"(?:^|\.)model\.norm$|^norm$")


def _contig_slice_from_mask(mask: torch.Tensor) -> Optional[slice]:
    """If mask True rows are one contiguous run, return a slice. Else None.
    CPU sync — but only fires once per begin_step, not per hook firing."""
    if mask is None:
        return None
    nz = mask.nonzero(as_tuple=False).flatten()
    if nz.numel() == 0:
        return slice(0, 0)
    lo, hi = int(nz[0].item()), int(nz[-1].item()) + 1
    if (hi - lo) == nz.numel():
        return slice(lo, hi)
    return None


class FinetuneAccumulator:
    def __init__(self) -> None:
        # Section 1: layer-input residual stream, fused gate_up output,
        # final-norm input + output, and the input ids for CE targets.
        self.layer_in: Dict[int, torch.Tensor] = {}
        self.mlp_gate_up: Dict[int, torch.Tensor] = {}
        self.final_in: Optional[torch.Tensor] = None
        self.final_hidden: Optional[torch.Tensor] = None
        self.concat_input_ids: Optional[torch.Tensor] = None
        self._finetune_mask: Optional[torch.Tensor] = None
        self._contig_slice: Optional[slice] = None
        self._req_ids: List[str] = []
        self._handles: List = []

    def register_hooks(
        self,
        model: nn.Module,
        module_map: Optional[Dict[nn.Module, int]] = None,
    ) -> None:
        if module_map is not None:
            for mod, idx in module_map.items():
                self._handles.append(
                    mod.register_forward_pre_hook(self._make_pre_hook(idx))
                )
                self._handles.append(
                    mod.register_forward_hook(self._make_post_hook(idx))
                )
            return

        for name, mod in model.named_modules():
            m_in = _LAYER_IN_RE.search(name)
            if m_in is not None:
                idx = int(m_in.group(1))
                self._handles.append(
                    mod.register_forward_pre_hook(self._make_pre_hook(idx))
                )
                continue
            m_gu = _GATE_UP_RE.search(name)
            if m_gu is not None:
                idx = int(m_gu.group(1))
                self._handles.append(
                    mod.register_forward_hook(self._make_post_hook(idx))
                )
                continue
            # Final norm: capture both input (`final_in`) and output (`final_hidden`)
            # so the backward can do head_backward without recomputing the norm.
            if _FINAL_NORM_RE.search(name):
                self._handles.append(
                    mod.register_forward_pre_hook(self._make_final_pre_hook())
                )
                self._handles.append(
                    mod.register_forward_hook(self._make_final_post_hook())
                )

    def _capture(self, val: torch.Tensor) -> torch.Tensor:
        sl = self._contig_slice
        if sl is not None:
            return val[sl].detach().clone()
        return val[self._finetune_mask].detach().clone()

    def _make_pre_hook(self, idx: int):
        def pre_hook(module, args):
            if self._finetune_mask is None:
                return
            if len(args) >= 2 and args[1] is not None:
                val = args[0] + args[1]
            else:
                val = args[0]
            self.layer_in[idx] = self._capture(val)

        return pre_hook

    def _make_post_hook(self, idx: int):
        def post_hook(module, inputs, output):
            if self._finetune_mask is None:
                return
            out = output[0] if isinstance(output, tuple) else output
            self.mlp_gate_up[idx] = self._capture(out)

        return post_hook

    def _make_final_pre_hook(self):
        def pre_hook(module, args):
            if self._finetune_mask is None:
                return
            val = args[0] if len(args) >= 1 else None
            if val is None:
                return
            if len(args) >= 2 and args[1] is not None:
                val = val + args[1]
            self.final_in = self._capture(val)

        return pre_hook

    def _make_final_post_hook(self):
        def post_hook(module, inputs, output):
            if self._finetune_mask is None:
                return
            out = output[0] if isinstance(output, tuple) else output
            self.final_hidden = self._capture(out)

        return post_hook

    def begin_step(
        self,
        req_ids: List[str],
        finetune_mask: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
    ) -> None:
        self._req_ids = list(req_ids)
        self._finetune_mask = finetune_mask
        self._contig_slice = _contig_slice_from_mask(finetune_mask) if finetune_mask is not None else None
        self.layer_in = {}
        self.mlp_gate_up = {}
        self.final_in = None
        self.final_hidden = None
        if input_ids is not None:
            self.concat_input_ids = (
                input_ids[self._contig_slice].detach().clone()
                if self._contig_slice is not None
                else input_ids[finetune_mask].detach().clone()
            )
        else:
            self.concat_input_ids = None

    def pop_step(self) -> dict:
        snapshot = {
            "layer_in": self.layer_in,
            "mlp_gate_up": self.mlp_gate_up,
            "final_in": self.final_in,
            "final_hidden": self.final_hidden,
            "concat_input_ids": self.concat_input_ids,
            "req_ids": list(self._req_ids),
        }
        self.layer_in = {}
        self.mlp_gate_up = {}
        self.final_in = None
        self.final_hidden = None
        self.concat_input_ids = None
        self._finetune_mask = None
        self._contig_slice = None
        self._req_ids = []
        return snapshot

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
