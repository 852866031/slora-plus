import argparse
import os
import sys

# base_model = "dummy-llama-7b"
base_model = "huggyllama/llama-7b"
#adapter_dirs = ["tloen/alpaca-lora-7b"]
adapter_dirs = ["tloen/alpaca-lora-7b", "MBZUAI/bactrian-x-llama-7b-lora"]
finetuning_lora_dir = "/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/eval/finetuning_adapter"
finetuning_config_path = "/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/eval/finetuning_config.json"

half_model = False
enable_unified_mem_manager = True
mem_manager_log_path = "/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/eval/mem_manager_log.txt"
enable_gpu_profile = False
unified_mem_manager_max_size = 8

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsys-output", type=str, default="my_trace_report")
    parser.add_argument("--num-adapter", type=int)
    parser.add_argument("--num-token", type=int)
    parser.add_argument("--pool-size-lora", type=int)
    parser.add_argument("--no-lora-compute", action="store_true")
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--no-mem-pool", action="store_true")

    ''' slora arguments '''
    args = parser.parse_args()

    
    if args.num_adapter is None: args.num_adapter = 1
    if args.num_token is None: args.num_token = 16000
    if args.pool_size_lora is None: args.pool_size_lora = 0
 
    cmd = f"python -m slora.server.api_server --max_total_token_num {args.num_token}"
    cmd += f" --model {base_model}"
    cmd += f" --tokenizer_mode auto"
    cmd += f" --pool-size-lora {args.pool_size_lora}"

    
    cmd += f" --finetuning_config_path {finetuning_config_path}"

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
    if mem_manager_log_path:
        cmd += f" --mem_manager_log_path {mem_manager_log_path}"
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
