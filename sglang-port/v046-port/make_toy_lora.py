#!/usr/bin/env python3
"""Create a toy PEFT LoRA adapter for a base model (no training, no download).

The adapter weights (*.safetensors) are gitignored, so regenerate the adapter
locally with this script after pulling. Zero-init B → the LoRA delta is 0
(identity until trained); it's a valid, servable PEFT adapter.

Usage:
  CUDA_VISIBLE_DEVICES="" python make_toy_lora.py \
      --model <hf-path-or-local-snapshot> \
      --out adapters/llama32-1b-toy-lora
"""
import argparse
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="base model path / HF id")
    ap.add_argument("--out", default="adapters/toy-lora")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    a = ap.parse_args()

    print(f"loading base on CPU: {a.model}")
    m = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=a.rank, lora_alpha=a.alpha,
        lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    pm = get_peft_model(m, cfg)
    pm.print_trainable_parameters()
    pm.save_pretrained(a.out)
    print(f"saved adapter -> {a.out}")


if __name__ == "__main__":
    main()
