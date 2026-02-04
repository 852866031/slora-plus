#!/usr/bin/env python3
"""
Extract only Q/K/V/O LoRA weights from an existing adapter,
optionally cast them to a specific dtype, and save as a new adapter.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

QKVO_SUBSTRINGS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "query_proj", "key_proj", "value_proj", "out_proj",
)

LORA_KEY_SUBSTRINGS = (
    "lora_A", "lora_B",
    "lora_embedding_A", "lora_embedding_B",
)

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def is_qkvo_lora_key(k: str) -> bool:
    if not any(s in k for s in LORA_KEY_SUBSTRINGS):
        return False
    return any(s in k for s in QKVO_SUBSTRINGS)


def load_adapter_state(adapter_dir: Path):
    st_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"

    if st_path.exists():
        from safetensors.torch import load_file
        state = load_file(str(st_path))
        src_path = st_path
    elif bin_path.exists():
        state = torch.load(str(bin_path), map_location="cpu")
        src_path = bin_path
    else:
        raise FileNotFoundError(
            f"Could not find adapter_model.safetensors or adapter_model.bin in {adapter_dir}"
        )
    return state, src_path


def save_adapter_state(state, out_dir: Path, prefer_safetensors: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    if prefer_safetensors:
        from safetensors.torch import save_file
        save_file(state, str(out_dir / "adapter_model.safetensors"))
    else:
        torch.save(state, str(out_dir / "adapter_model.bin"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="HF repo id or local adapter directory")
    ap.add_argument("--out", required=True, help="Output directory to write the new adapter")
    ap.add_argument(
        "--dtype",
        default=None,
        help="Optional target dtype: float16, bfloat16, or float32 (e.g., --dtype float16)",
    )
    ap.add_argument(
        "--no_safetensors",
        action="store_true",
        help="Save as .bin instead of .safetensors",
    )
    args = ap.parse_args()

    # Validate dtype if provided
    target_dtype = None
    if args.dtype is not None:
        if args.dtype not in DTYPE_MAP:
            raise ValueError(f"Unsupported dtype: {args.dtype}. Choose from {list(DTYPE_MAP)}")
        target_dtype = DTYPE_MAP[args.dtype]

    out_dir = Path(args.out)

    # Resolve adapter directory
    if os.path.isdir(args.src):
        adapter_dir = Path(args.src)
    else:
        adapter_dir = Path(
            snapshot_download(
                repo_id=args.src,
                allow_patterns=[
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "adapter_model.bin",
                ],
            )
        )

    # Copy config (preserves r, alpha, dropout, etc.)
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {adapter_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, out_dir / "adapter_config.json")

    # Optionally narrow target_modules in config
    cfg = json.loads(cfg_path.read_text())
    if "target_modules" in cfg and isinstance(cfg["target_modules"], list):
        cfg["target_modules"] = [
            m for m in cfg["target_modules"]
            if any(s in m for s in QKVO_SUBSTRINGS)
        ]
    (out_dir / "adapter_config.json").write_text(
        json.dumps(cfg, indent=2) + "\n"
    )

    # === LOAD ADAPTER ===
    state, _ = load_adapter_state(adapter_dir)

    # Print original dtypes
    dtypes = {v.dtype for v in state.values() if hasattr(v, "dtype")}
    print(f"Loaded adapter tensor dtypes (unique): {dtypes}")

    # Filter to keep only Q/K/V/O LoRA weights
    new_state = {k: v for k, v in state.items() if is_qkvo_lora_key(k)}

    if len(new_state) == 0:
        sample = list(state.keys())[:30]
        raise RuntimeError(
            "No QKVO LoRA keys matched. Your adapter may use different projection names.\n"
            "First 30 keys:\n" + "\n".join(sample)
        )

    # === CAST TO TARGET DTYPE IF REQUESTED ===
    if target_dtype is not None:
        print(f"Casting all kept tensors to {target_dtype}")
        new_state = {
            k: v.to(dtype=target_dtype)
            for k, v in new_state.items()
        }

        # Sanity check
        final_dtypes = {v.dtype for v in new_state.values()}
        print(f"Final saved dtypes (unique): {final_dtypes}")

    save_adapter_state(
        new_state,
        out_dir,
        prefer_safetensors=not args.no_safetensors,
    )

    print(f"Source adapter: {adapter_dir}")
    print(f"Saved new adapter to: {out_dir}")
    print(f"Kept tensors: {len(new_state)} / {len(state)}")


if __name__ == "__main__":
    main()