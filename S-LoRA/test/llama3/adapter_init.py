# train_llama3_qkvo_lora_ready.py
# Modified: initializes a tiny LoRA adapter (Q/K/V/O only),
# prints LoRA parameter shapes, and saves the adapter immediately (NO TRAINING).
#
# Run (2 GPUs still fine, but unnecessary since there's no training):
#   pip install -U transformers datasets accelerate peft
#   huggingface-cli login   # required for Meta-Llama-3 weights
#   accelerate launch --multi_gpu train_llama3_qkvo_lora_ready.py

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

MODEL_ID = "meta-llama/Meta-Llama-3-8B"
OUT_DIR = "adapters/llama3-toy-lora-init-only"


def print_lora_shapes(model):
    print("\n=== LoRA parameter shapes (only these should be trainable) ===")
    found = 0
    for n, p in model.named_parameters():
        if "lora_" in n:
            print(f"{n}: {tuple(p.shape)}  dtype={p.dtype}  device={p.device}")
            found += 1
    if found == 0:
        print("WARNING: No LoRA parameters found. Check target_modules / PEFT config.")
    print("============================================================\n")


def main():
    # Tokenizer (saved alongside adapter for convenience)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Base model
    use_bf16 = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    )

    # LoRA ONLY on attention projections: q/k/v/o
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    # Trainable parameters summary + LoRA shapes (right after init)
    model.print_trainable_parameters()
    print_lora_shapes(model)

    # Save adapter immediately (PEFT)
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"Saved *untrained* LoRA adapter to: {OUT_DIR}")


if __name__ == "__main__":
    main()