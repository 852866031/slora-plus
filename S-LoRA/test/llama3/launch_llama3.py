import argparse
import os
import socket
import sys
import os, subprocess, time, shutil
import socket
import requests

CONFIG = {
    "online": {
        "base_model": "meta-llama/Meta-Llama-3-8B",
        "adapter_dirs": [
            "/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/llama3/adapters/llama3-toy-lora",
        ],
        "finetuning_config_path": "/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/llama3/config/finetuning_config.json",
        "no_finetuning_config_path": "/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/llama3/config/no_finetuning_config.json",
    },

    "defaults": {
        "half_model": False,
        "enable_unified_mem_manager": True,
        "enable_gpu_profile": False,
        "unified_mem_manager_max_size": 6,
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
        BASE = CONFIG["online"]
    else:
        print("⚠️  WARNING: Internet is not available. Exiting.")
        sys.exit(1)

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
    parser.add_argument("--ft_log_path", type=str, default="/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/llama3/bwd_log.csv")

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
    cmd += f" --ft_log_path {args.ft_log_path}"

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