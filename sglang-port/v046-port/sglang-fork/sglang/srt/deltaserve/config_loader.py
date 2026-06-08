# SPDX-License-Identifier: Apache-2.0
"""YAML config loader for sglang DeltaServe co-serving (Phase G).

Port of DeltaServe-vLLM's deltaserve/config_loader.py, retargeted from vLLM
EngineArgs to sglang ServerArgs. Reads a DeltaServe-style sectioned YAML and
produces (server_args_kwargs, FinetuneConfig, extras):

  finetune / slo / debug  -> folded into FinetuneConfig kwargs (same vocab as vLLM)
  model / engine / parallel / lora -> merged + vocab-translated to ServerArgs
  server / adapters       -> returned verbatim as extras

The finetune/slo/debug sections are vocab-identical to vLLM (both are
FinetuneConfig fields), so the original vLLM yaml's FT/SLO knobs load unchanged.
The engine-vocab sections are written in vLLM EngineArgs names and are translated
to sglang ServerArgs names via _ENGINE_ALIASES (model->model_path, etc).

中文：把 vLLM 的分节 YAML 直接加载进 sglang —— finetune/slo/debug 折进 FinetuneConfig
（与 vLLM 同词表），engine 词表（model/engine/parallel/lora）按别名表翻译成 sglang
ServerArgs。这样 vLLM 那份 _both_A100.yaml 的 SLO/微调旋钮能逐字加载，配置"一模一样"。
"""
from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_FINETUNE_SECTION = "finetune"
_DEBUG_SECTION = "debug"
_SLO_SECTION = "slo"
_PASSTHROUGH_SECTIONS = ("server", "adapters")
_SPECIAL_SECTIONS = (_FINETUNE_SECTION, _DEBUG_SECTION, _SLO_SECTION,
                     *_PASSTHROUGH_SECTIONS)

# vLLM EngineArgs vocab -> sglang ServerArgs field names (§G translation table).
_ENGINE_ALIASES = {
    "model": "model_path",
    "max_model_len": "context_length",
    "gpu_memory_utilization": "mem_fraction_static",
    "tensor_parallel_size": "tp_size",
    "max_loras": "max_loras_per_batch",
    # enforce_eager / lora_path_* / enable_lora handled as special cases below.
}
# vLLM-frontend-only keys with no sglang analogue → warn & drop.
_ENGINE_DROP = ("api_server_count", "disable_log_stats", "enable_lora")


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Read + validate the sectioned YAML. Resolve relative ``*path*`` values
    under finetune/adapters against the YAML's own directory."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"DeltaServe config not found: {path}")
    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping of sections")
    for section, body in raw.items():
        if body is not None and not isinstance(body, dict):
            raise ValueError(f"{path}: section '{section}' must be a mapping")
    base_dir = path.resolve().parent
    for section in (_FINETUNE_SECTION, "adapters"):
        body = raw.get(section) or {}
        for key, val in list(body.items()):
            if "path" in key.lower() and isinstance(val, str) \
                    and not Path(val).is_absolute():
                body[key] = str((base_dir / val).resolve())
    logger.warning(f"[DeltaServe] loaded config {path}: sections={list(raw)}")
    return raw


def split_config(config: dict[str, Any]):
    """-> (server_args_kwargs, FinetuneConfig, extras)."""
    from sglang.srt.configs.finetune import FinetuneConfig

    # finetune + slo + debug -> FinetuneConfig (filter to known dataclass fields).
    finetune_body = {**(config.get(_FINETUNE_SECTION) or {}),
                     **(config.get(_DEBUG_SECTION) or {}),
                     **(config.get(_SLO_SECTION) or {})}
    known = {f.name for f in dataclasses.fields(FinetuneConfig)}
    ft_kwargs, dropped = {}, []
    for k, v in finetune_body.items():
        if k in known:
            ft_kwargs[k] = v
        else:
            dropped.append(k)
    if dropped:
        logger.warning(f"[DeltaServe] config: ignoring unknown finetune/slo/debug "
                       f"keys (no sglang field): {dropped}")
    finetune_config = FinetuneConfig(**ft_kwargs)

    extras = {name: dict(config.get(name) or {}) for name in _PASSTHROUGH_SECTIONS}

    # engine-vocab sections -> one ServerArgs kwargs bag (translated).
    sa_kwargs: dict[str, Any] = {}
    for section, body in config.items():
        if section in _SPECIAL_SECTIONS:
            continue
        for key, value in (body or {}).items():
            if key in _ENGINE_DROP:
                logger.warning(f"[DeltaServe] config: dropping vLLM-only engine key "
                               f"'{key}' (no sglang analogue)")
                continue
            if key == "enforce_eager":
                # vLLM enforce_eager:true == sglang disable_cuda_graph:true (same polarity)
                sa_kwargs["disable_cuda_graph"] = bool(value)
                continue
            if key.startswith("lora_path"):
                sa_kwargs.setdefault("lora_paths", []).append(value)
                continue
            tgt = _ENGINE_ALIASES.get(key, key)
            sa_kwargs[tgt] = value

    # adapters.lora_path_* -> lora_paths too (inference served adapters)
    for key, value in (config.get("adapters") or {}).items():
        if key.startswith("lora_path"):
            sa_kwargs.setdefault("lora_paths", []).append(value)

    return sa_kwargs, finetune_config, extras


def finetune_config_from_yaml(path: str | Path):
    """Convenience for the scheduler subprocess: YAML path -> FinetuneConfig."""
    _, ft, _ = split_config(load_yaml_config(path))
    return ft
