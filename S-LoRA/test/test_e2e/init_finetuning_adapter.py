import os
import math
import json
import torch
import torch.nn.init as init
from safetensors.torch import save_file

# === Configuration ===
num_layers = 32
hidden_size = 4096
r = 16
lora_alpha = 16
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
lora_dir = "/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/test_e2e/finetuning_adapter"
os.makedirs(lora_dir, exist_ok=True)

# === Save config file ===
lora_config = {
    "base_model_name_or_path": "decapoda-research/llama-7b-hf",
    "bias": "none",
    "enable_lora": None,
    "fan_in_fan_out": False,
    "inference_mode": True,
    "lora_alpha": lora_alpha,
    "lora_dropout": lora_dropout,
    "merge_weights": False,
    "modules_to_save": None,
    "peft_type": "LORA",
    "r": r,
    "target_modules": target_modules,
    "task_type": "CAUSAL_LM"
}
with open(os.path.join(lora_dir, "adapter_config.json"), "w") as f:
    json.dump(lora_config, f, indent=2)

# === Helper: initialize lora_A with Kaiming uniform, lora_B with zeros ===
def make_lora_weights(module_name, layer_idx):
    prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{module_name}"
    A_name = f"{prefix}.lora_A.weight"
    B_name = f"{prefix}.lora_B.weight"

    A = torch.empty((r, hidden_size), dtype=torch.float16)
    B = torch.empty((hidden_size, r), dtype=torch.float16)

    # Kaiming uniform initialization for lora_A
    init.kaiming_uniform_(A, a=math.sqrt(5))
    init.kaiming_uniform_(B, a=math.sqrt(5))

    return {A_name: A, B_name: B}

# === Create all LoRA weights ===
all_weights = {}
for layer_idx in range(num_layers):
    for module_name in target_modules:
        all_weights.update(make_lora_weights(module_name, layer_idx))

# === Save as safetensors ===
save_file(all_weights, os.path.join(lora_dir, "adapter_model.safetensors"))

print(f"LoRA adapter initialized and saved to: {lora_dir}")