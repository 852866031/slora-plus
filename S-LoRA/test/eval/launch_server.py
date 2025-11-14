import argparse
import os
import socket
import sys
import os, subprocess, time, shutil

def internet_available(timeout=2):
    """Check internet by pinging HuggingFace DNS & HTTPS."""
    try:
        # DNS check
        socket.gethostbyname("huggingface.co")
        # HTTPS check
        requests.head("https://huggingface.co", timeout=timeout)
        return True
    except Exception:
        return False


if not internet_available():
    path = "/projects/I20240005/jchen/slora-plus/S-LoRA/test/eval/"
    base_model = "/projects/I20240005/jchen/hf_cache/models--huggyllama--llama-7b/snapshots/llama-7b"
    adapter_dirs = [
        "/projects/I20240005/jchen/hf_cache/hub/models--tloen--alpaca-lora-7b/snapshots/12103d6baae1b320aa60631b38acb6ea094a0539",
        "/projects/I20240005/jchen/hf_cache/hub/models--MBZUAI--bactrian-x-llama-7b-lora/snapshots/73e293a50ce88d19581f76502aa7baef42bc228b"
    ]
    finetuning_config_path = path + "config/finetuning_config.json"
    no_finetuning_config_path = path + "config/no_finetuning_config.json"
    #
    modules_loaded = "\
        module load GCCcore/13.2.0 \
        module load binutils/2.40-GCCcore-13.2.0 \
        module load CUDA/12.4 \
    "
    build_cmd= " \
        PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_BUILD_ISOLATION=1 pip install -e .  \
        --no-build-isolation     --no-deps     --config-settings=--build-option=--no-isolation     -vvv \
    "

else:
    base_model = "huggyllama/llama-7b"
    adapter_dirs = ["tloen/alpaca-lora-7b"]
    adapter_dirs = ["tloen/alpaca-lora-7b", "MBZUAI/bactrian-x-llama-7b-lora"]

half_model = False
enable_unified_mem_manager = True
enable_gpu_profile = False
unified_mem_manager_max_size = 8
#  sudo echo quit | sudo nvidia-cuda-mps-control


def enable_offline_mode():
    HF_CACHE_DIR = "/projects/I20240005/jchen/hf_cache"
    """Set all environment variables needed for completely offline HF loading."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR

    print("🔌 No internet detected. Running in OFFLINE mode.")
    print(f"   → HF_HUB_OFFLINE=1")
    print(f"   → TRANSFORMERS_OFFLINE=1")
    print(f"   → HF_HOME={HF_CACHE_DIR}")
    print(f"   → TRANSFORMERS_CACHE={HF_CACHE_DIR}\n")


def enable_online_mode():
    """Unset offline variables to allow HuggingFace downloads."""
    for var in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]:
        if var in os.environ:
            del os.environ[var]

    print("🌐 Internet detected. Running in ONLINE mode.")
    print("   → HF_HUB_OFFLINE unset")
    print("   → TRANSFORMERS_OFFLINE unset\n")

def is_mps_running():
    """
    Returns True if the MPS control daemon is up and responding.
    """
    exe = shutil.which("nvidia-cuda-mps-control")
    if not exe:
        return False
    # Try to talk to the daemon (this is a client; it fails if the daemon isn't up)
    try:
        p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate("get_server_list\nquit\n", timeout=2.0)
        return p.returncode == 0
    except Exception:
        return False


if __name__ == "__main__":
    if internet_available():
        enable_online_mode()
    else:
        os.system("nvidia-cuda-mps-control -d")
        enable_offline_mode()
    if not is_mps_running():
        print("MPS control daemon is not running. Please start it before running this script:\n sudo nvidia-cuda-mps-control -d")
        sys.exit(1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsys-output", type=str, default="my_trace_report")
    parser.add_argument("--num-adapter", type=int)
    parser.add_argument("--num-token", type=int)
    parser.add_argument("--pool-size-lora", type=int)
    parser.add_argument("--no-lora-compute", action="store_true")
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--no-mem-pool", action="store_true")
    parser.add_argument("--enable-finetuning", action="store_true")

    ''' slora arguments '''
    args = parser.parse_args()

    
    if args.num_adapter is None: args.num_adapter = 1
    if args.num_token is None: args.num_token = 16000
    if args.pool_size_lora is None: args.pool_size_lora = 0
 
    cmd = f"python -m slora.server.api_server --max_total_token_num {args.num_token}"
    cmd += f" --model {base_model}"
    cmd += f" --tokenizer_mode auto"
    cmd += f" --pool-size-lora {args.pool_size_lora}"

    if args.enable_finetuning:
        cmd += f" --finetuning_config_path {finetuning_config_path}"
    else:
        cmd += f" --finetuning_config_path {no_finetuning_config_path}"

    num_iter = args.num_adapter // len(adapter_dirs) + 1
    for adapter_dir in adapter_dirs:
        cmd += f" --lora {adapter_dir}"

    cmd += " --swap"
    # cmd += " --scheduler pets"
    # cmd += " --profile"
    if args.no_lora_compute:
        cmd += " --no-lora-compute"
    if args.no_prefetch:
        cmd += " --prefetch False"
    if args.no_mem_pool:
        cmd += " --no-mem-pool"
    
    if half_model:
        cmd += " --half_model"
    if enable_unified_mem_manager:
        cmd += " --enable_unified_mem_manager"
        cmd += f" --unified_mem_manager_max_size {unified_mem_manager_max_size}"
    if enable_gpu_profile:
        profiler_cmd = f"nsys profile --cuda-memory-usage=true  --trace-fork-before-exec=true --force-overwrite true -o {args.nsys_output} "
        #profiler_cmd = f"nsys profile -t cuda,osrt,nvtx --capture-range=nvtx --capture-range-end=stop --cuda-memory-usage=true --force-overwrite true -o {args.nsys_output} "
        cmd += f" --enable_gpu_profile"
        cmd = profiler_cmd + cmd
    print(cmd)
    os.system(cmd)
