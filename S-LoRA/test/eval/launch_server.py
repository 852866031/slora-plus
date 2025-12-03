import argparse
import os
import socket
import sys
import os, subprocess, time, shutil
import socket
import requests

modules_loaded = "\
        module load GCCcore/13.2.0 \
        module load binutils/2.40-GCCcore-13.2.0 \
        module load CUDA/12.4 \
    "
build_cmd= " \
        PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_BUILD_ISOLATION=1 pip install -e .  \
        --no-build-isolation     --no-deps     --config-settings=--build-option=--no-isolation     -vvv \
    "

CONFIG = {
    "offline": {
        "path": "/projects/I20240005/jchen/slora-plus/S-LoRA/test/eval/",
        "base_model": "/projects/I20240005/jchen/hf_cache/models--huggyllama--llama-7b/snapshots/llama-7b",
        "adapter_dirs": [
            "/projects/I20240005/jchen/hf_cache/hub/models--tloen--alpaca-lora-7b/snapshots/12103d6baae1b320aa60631b38acb6ea094a0539",
            "/projects/I20240005/jchen/hf_cache/hub/models--MBZUAI--bactrian-x-llama-7b-lora/snapshots/73e293a50ce88d19581f76502aa7baef42bc228b"
        ],
        "finetuning_config_path": "/projects/I20240005/jchen/slora-plus/S-LoRA/test/eval/config/finetuning_config_d.json",
        "no_finetuning_config_path": "/projects/I20240005/jchen/slora-plus/S-LoRA/test/eval/config/no_finetuning_config_d.json",
    },

    "online": {
        "base_model": "huggyllama/llama-7b",
        "adapter_dirs": [
            "tloen/alpaca-lora-7b",
            "MBZUAI/bactrian-x-llama-7b-lora"
        ],
        "finetuning_config_path": "/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/eval/config/finetuning_config.json",
        "no_finetuning_config_path": "/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/eval/config/no_finetuning_config.json",
    },

    "defaults": {
        "half_model": False,
        "enable_unified_mem_manager": True,
        "enable_gpu_profile": False,
        "unified_mem_manager_max_size": 16,
        "num_adapter": 1,
        "num_token": 25000,
        "pool_size_lora": 0,
    }
}

def internet_available(timeout=2):
    """Check internet by pinging HuggingFace DNS & HTTPS."""
    try:
        socket.gethostbyname("huggingface.co")
        requests.head("https://huggingface.co", timeout=timeout)
        return True
    except Exception:
        return False

def enable_offline_mode():
    HF_CACHE_DIR = "/projects/I20240005/jchen/hf_cache"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR

    print("🔌 No internet detected. Running in OFFLINE mode.")
    print("   → HF_HUB_OFFLINE=1")
    print("   → TRANSFORMERS_OFFLINE=1")
    print(f"   → HF_HOME={HF_CACHE_DIR}")
    print(f"   → TRANSFORMERS_CACHE={HF_CACHE_DIR}\n")


def enable_online_mode():
    for var in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]:
        if var in os.environ:
            del os.environ[var]
    print("🌐 Internet detected. Running in ONLINE mode.")
    print("   → HF_HUB_OFFLINE unset")
    print("   → TRANSFORMERS_OFFLINE unset\n")


def is_mps_running():
    exe = shutil.which("nvidia-cuda-mps-control")
    if not exe:
        return False
    try:
        p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate("get_server_list\nquit\n", timeout=2.0)
        return p.returncode == 0
    except Exception:
        return False

if __name__ == "__main__":
    online = internet_available()

    if online:
        enable_online_mode()
        BASE = CONFIG["online"]
    else:
        os.system("nvidia-cuda-mps-control -d")
        enable_offline_mode()
        BASE = CONFIG["offline"]

    if not is_mps_running():
        print("MPS control daemon is not running. Please start it with:")
        print("  sudo nvidia-cuda-mps-control -d")
        sys.exit(1)

    # -----------------------------------
    # 👇 Only expose 3 arguments to user
    # -----------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--enable-finetuning", action="store_true")
    parser.add_argument("--rank_id", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--bwd_log_index", type=int, default=0)
    args = parser.parse_args()

    # Load defaults
    D = CONFIG["defaults"]

    # -----------------------------------
    # construct CMD (no behavior changed)
    # -----------------------------------
    cmd = f"python -m slora.server.api_server --max_total_token_num {D['num_token']}"
    cmd += f" --model {BASE['base_model']}"
    cmd += f" --tokenizer_mode auto"
    cmd += f" --pool-size-lora {D['pool_size_lora']}"
    cmd += f" --port {args.port}"
    cmd += f" --rank_id {args.rank_id}"
    cmd += f" --bwd_log_index {args.bwd_log_index}"

    if args.enable_finetuning:
        cmd += f" --finetuning_config_path {BASE['finetuning_config_path']}"
    else:
        cmd += f" --finetuning_config_path {BASE['no_finetuning_config_path']}"

    # adapter dirs
    for adapter_dir in BASE["adapter_dirs"]:
        cmd += f" --lora {adapter_dir}"
    cmd += " --swap"

    # unified mem manager etc.
    if D["half_model"]:
        cmd += " --half_model"
    if D["enable_unified_mem_manager"]:
        cmd += " --enable_unified_mem_manager"
        cmd += f" --unified_mem_manager_max_size {D['unified_mem_manager_max_size']}"
    if D["enable_gpu_profile"]:
        profile_cmd = f"nsys profile --cuda-memory-usage=true --trace-fork-before-exec=true --force-overwrite true -o trace "
        cmd = profile_cmd + cmd

    print(cmd)
    os.system(cmd)