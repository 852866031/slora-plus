"""Pure-FT (no inference) throughput at a chosen backward MPS %.

Launches the sglang server with finetuning on at --mps, POSTs /start_finetuning,
idles (zero inference traffic) for a window, and parses the
``[DeltaServe] real_backward #N: ... n_valid=N ... wall=T`` fire log to compute
tok/s over a clean 40s window. Mirrors the 10%/100% runs.
"""
import os, re, sys, time, socket, subprocess, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "meta-llama/Meta-Llama-3-8B"
CFG = os.path.join(HERE, "configs", "serving_config_finetuning_llama3_both.yaml")
CORPUS = "/mnt/weka/home/jianshu.she/slora-plus/eval/llama3/data/alpaca_1000_p95.txt"

mps = int(sys.argv[1]) if len(sys.argv) > 1 else 30
window = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0

def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

port = free_port()
log = f"/tmp/pure_ft_sglang_mps{mps}.log"
cmd = [
    sys.executable, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--host", "127.0.0.1", "--port", str(port),
    "--tp-size", "1", "--mem-fraction-static", "0.5",
    "--enable-finetuning", "--backward-mps-percentage", str(mps),
    "--finetune-config", CFG, "--disable-radix-cache",
    "--finetune-data-path", CORPUS,
]
env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = "0"  # pin to the MPS-served GPU
env.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
env.setdefault("CUDA_MPS_LOG_DIRECTORY", "/tmp/nvidia-mps-log")
env.setdefault("HF_HOME", "/mnt/weka/home/jianshu.she/.cache/huggingface")
env.pop("HF_HUB_OFFLINE", None)  # offline blocks the optional quant-config probe
# Route to the REAL MPS-isolated backward subprocess (else the scheduler uses
# the faux graph-replay stub). Mirrors auto_benchmark_sglang.py.
env["SGLANG_DS_REAL_BACKWARD"] = "1"
env["SGLANG_DS_BACKWARD_SUBPROCESS"] = "1"
env["SGLANG_DS_BACKWARD_MPS_PCT"] = str(mps)
env["SGLANG_DS_FT_START_ON_LAUNCH"] = "0"

print(f"[run] mps={mps}% port={port} log={log}", flush=True)
lf = open(log, "w")
proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)

def ready():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/get_model_info", timeout=2)
        return True
    except Exception:
        return False

t0 = time.time()
while time.time() - t0 < 600:
    if proc.poll() is not None:
        print("[run] server died; tail:", flush=True)
        os.system(f"tail -30 {log}"); sys.exit(1)
    if ready():
        break
    time.sleep(3)
print(f"[run] ready in {time.time()-t0:.0f}s; POST start_finetuning", flush=True)
try:
    urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/start_finetuning", method="POST", data=b"{}",
        headers={"Content-Type": "application/json"}), timeout=10).read()
except Exception as e:
    print(f"[run] start_finetuning failed: {e}", flush=True)

time.sleep(window + 8)  # let fires accumulate over the window (+slack)

proc.terminate()
try: proc.wait(timeout=20)
except Exception: proc.kill()
lf.close()

# Parse fires: [DeltaServe] real_backward #N: Xms loss=.. n_valid=N cum_tokens=.. wall=T
pat = re.compile(r"real_backward #(\d+): .* n_valid=(\d+) cum_tokens=\d+ wall=([\d.]+)")
fires = []
for line in open(log, errors="ignore"):
    m = pat.search(line)
    if m:
        fires.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
if not fires:
    print("[run] NO fires parsed; tail:", flush=True)
    os.system(f"grep -iE 'real_backward|bwd_child|error|Traceback' {log} | tail -20")
    sys.exit(1)

# clean window: from first fire, take `window` seconds
t_start = fires[0][2]
win = [f for f in fires if f[2] - t_start <= window]
nv = [f[1] for f in win]
tot = sum(nv)
span = win[-1][2] - win[0][2] if len(win) > 1 else window
tps = tot / window
print("\n=== sglang 8B @ {}% MPS — PURE FT idle ===".format(mps), flush=True)
print(f"{len(win)} fires, {tot} tok / {window:.0f}s = {tps:.0f} tok/s, "
      f"n_valid mean={tot/max(1,len(nv)):.0f} (span={span:.1f}s)", flush=True)
print("COMPARE: 10%=433 | 100%=881 | vLLM H200@10%(graph+save_attn)=410 | "
      "vLLM 5090@10%=1020 | vLLM A100@10%=345", flush=True)
