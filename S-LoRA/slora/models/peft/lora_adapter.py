import re
import torch
import os
import json
from slora.mprophet.lora_config import get_lora_config_json
from slora.models.peft.layer_weights.hf_load_utils import load_hf_weights
from slora.models.peft.layer_weights.lora_layer_weight import LoraLayerWeight
from slora.utils.model_load import hf_load_config
from slora.server.router.mixed_req_queue import rprint

def create_lora_adapter(
    adapter_dir: str,
    base_model: str,
    network_config: dict,
    rank: int = 8,
    lora_alpha: int = 8,
    target_modules: list = None,
    use_safetensors: bool = False,
    tp_rank: int = 0,
    world_size: int = 1,
    no_lora_swap: bool = False,
    prefetch_stream=None
):
    if os.path.exists(adapter_dir):
        print(f"LoRA adapter directory '{adapter_dir}' already exists. Skipping creation.")
        return

    os.makedirs(adapter_dir, exist_ok=True)

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # 1) Create the adapter config dictionary
    adapter_config = {
        "base_model_name_or_path": base_model,
        "r": rank,
        "lora_alpha": lora_alpha,
        "target_modules": target_modules,
        "bias": "none",
        "enable_lora": None,
        "fan_in_fan_out": False,
        "inference_mode": True,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "merge_weights": False,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": 16,
        "task_type": "CAUSAL_LM"
    }

    config_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)
    print(f"Created {config_path}")

    layers = [
        LoraLayerWeight(i, tp_rank, world_size, adapter_config, network_config, torch.float16,
                        no_lora_swap=no_lora_swap, prefetch_stream=prefetch_stream).init_dummy(not no_lora_swap)
        for i in range(network_config["num_hidden_layers"])
        ]
    #TODO: finish this code so that we create a new adapter for finetuning


def get_lora_config(lora_dir, dummy):
    if dummy:
        return get_lora_config_json(lora_dir), lora_dir
    else:
        lora_dir = re.sub(r'-(\d+)$', '', lora_dir)
        return hf_load_config(lora_dir)

        

class LoraTpPartAdapter:

    def __init__(self, tp_rank, world_size, lora_dir, network_config,
                 swap=False, dummy=False, no_lora_swap=False, prefetch_stream=None):
        assert world_size == 1
        self.tp_rank_ = tp_rank
        self.world_size_ = world_size
        self.lora_dir = lora_dir
        self.network_config = network_config
        self.lora_config, lora_dir = get_lora_config(lora_dir, dummy)
        self.r = self.lora_config["r"]
        self.lora_alpha = self.lora_config["lora_alpha"]
        self.scaling = self.lora_alpha / self.r
        
 
        self.layers = [
            LoraLayerWeight(i, tp_rank, world_size, self.lora_config, network_config, torch.float16,
                            no_lora_swap=no_lora_swap, prefetch_stream=prefetch_stream)
            for i in range(network_config["num_hidden_layers"])
        ]

        self.prefetch_stream = prefetch_stream

        load_hf_weights("fp16", lora_dir, transformer_layer_list=self.layers,
                        swap=swap, dummy=dummy)

    def is_on_gpu(self,):
        return (self.layers[0].w_combined is not None)


    def load_to_gpu(self, prefetch=False, bmm=False):
        if prefetch:
            with self.prefetch_stream:
                for layer_weight in self.layers:
                    layer_weight.load_to_gpu(bmm=bmm)
        else:
            for layer_weight in self.layers:
                layer_weight.load_to_gpu(bmm=bmm)


    def offload_from_gpu(self,):
        for layer_weight in self.layers:
            layer_weight.offload_from_gpu()
