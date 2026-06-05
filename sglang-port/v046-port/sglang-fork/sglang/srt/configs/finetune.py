# Copied from DeltaServe-vLLM dserve-vllm/vllm/config/finetune.py:17 — sglang port
# SPDX-License-Identifier: Apache-2.0
"""Configuration for DeltaServe co-serving (inference + LoRA finetuning).

sglang port of the upstream FinetuneConfig dataclass. The original applies a
vLLM-specific ``@config`` decorator (``vllm.config.utils.config``); for the
phase-1 scaffold we use a plain ``@dataclass`` so this module has no external
dependencies. Later phases may swap in sglang's equivalent registration
decorator when wiring through ``server_args.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class FinetuneConfig:
    """Knobs for the DeltaServe co-serving finetuning layer."""

    enable_finetuning: bool = False
    """Master switch for co-serving. When True, the scheduler spawns the
    backward (SFT) subprocess and sets up the shared activation buffers. When
    False (default), sglang behaves exactly like upstream — no extra process,
    no overhead."""

    backward_mps_percentage: int = 10
    """CUDA MPS active-thread percentage granted to the backward (SFT) process.
    Applied as CUDA_MPS_ACTIVE_THREAD_PERCENTAGE only while spawning the child,
    so it inherits a constrained MPS partition and inference keeps the rest.
    Mirrors DeltaServe's model_rpc.py (=10). Requires the MPS daemon to be
    running to take effect."""

    finetuning_lora_path: Optional[str] = None
    """Path to the dedicated finetuning LoRA adapter (PEFT format dir). This is
    the adapter the backward process trains. Its weights are loaded fp32 and
    shared with the backward process."""

    data_path: Optional[str] = None
    """Path to the finetuning corpus: one tokenizable sample per non-empty line.
    Loaded + tokenized at startup by the FinetuningStore."""

    num_epochs: int = 1
    """Number of passes over the finetuning corpus."""

    learning_rate: float = 1e-4
    """AdamW learning rate for the LoRA SFT backward."""

    weight_decay: float = 0.0
    """AdamW weight decay for the LoRA SFT backward."""

    gamma: float = 1.0
    """StepLR multiplicative decay applied once per finetuning epoch."""

    backward_fp32: bool = False
    """Run the backward's bulk matmuls in fp32. Default False = the model dtype."""

    backward_cuda_graph: bool = False
    """Capture per-layer backward sub-regions as CUDA graphs."""

    backward_cuda_graph_attn_bn_max: int = 8
    """[backward_cuda_graph] Padded-attention max distinct samples per backward."""

    backward_cuda_graph_attn_l_max: int = 64
    """[backward_cuda_graph] Padded-attention max per-sample sequence length."""

    max_prepare: Optional[int] = None
    """Cap on how many corpus lines to load (None = all)."""

    max_saved_finetuning_tokens: int = 256
    """Per-backward FT token budget. Also sizes the shared activation buffers."""

    save_activations: bool = True
    """Whether to actually accumulate FT activations into the shared buffers."""

    backward_sleep_seconds: float = 2.0
    """Stub backward sleep duration to simulate a backward pass."""

    # --- SLO-aware admission + execution-time estimator ---
    ttft_slo: float = 1.0
    """Time-to-first-token SLO (seconds)."""

    avg_tbt_slo: float = 0.1
    """Average time-between-tokens SLO (seconds)."""

    max_tbt_slo: float = 0.2
    """Max time-between-tokens SLO (seconds)."""

    ft_tokens_admission_constrain_factor: float = -1.0
    """Cap FT tokens admitted per step relative to that step's INFERENCE PREFILL."""

    profile_on_launch: bool = True
    """Run the offline execution-time profiling pass at launch."""

    start_on_launch: bool = True
    """Whether FT admission is open as soon as serving begins."""

    profile_num_repeats: int = 2
    """Recorded passes per profiling shape."""

    batch_prediction_stats_path: Optional[str] = None
    """If set, dump per-step predicted-vs-actual execution times to this CSV."""

    bwd_log_path: Optional[str] = None
    """If set, append one row per completed backward to this CSV."""

    # --- Inference pre-emption of FT-only stepping ---
    forward_interruptible: bool = False
    """Master switch for the inference-preemption pipeline."""

    ft_only_admission_grace_ms: float = 2.0
    """[forward_interruptible] Tier-A grace window (ms)."""

    # --- debug / observability ---
    print_weight_hash: bool = False
    print_activation_hash: bool = False
    print_step_mode: bool = False
    print_scheduler_add: bool = False
    print_engine_batch_exec: bool = False
    print_engine_batch_done: bool = False
    print_engine_req_recv: bool = False
