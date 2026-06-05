#!/usr/bin/env python3
"""test_publish_lora.py — §13: does the trained LoRA actually affect serving?

Launches a server with the in-process real backward AND publish hooks on
(SGLANG_DS_PUBLISH_LORA=1). Then:
  1. measure L0 = total logprob the model assigns to a target sequence
  2. overfit on that sequence (send it as is_finetuning=True N times → backward
     trains the LoRA masters, hooks apply them in the forward)
  3. measure L1 = total logprob again
PASS iff L1 > L0 by a clear margin — i.e. the served model now fits the sample
better, so training changed serving (not write-only anymore).

Control: a second sequence NOT trained on should move much less.

Usage: python scripts/test_publish_lora.py [--steps 200]
"""
from __future__ import annotations
import argparse, json, os, signal, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path

_PORT_DIR = Path(__file__).resolve().parent.parent
MODEL = ("/mnt/weka/home/jianshu.she/.cache/huggingface/hub/"
         "models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"
         "9213176726f574b556790deb65791e0c5aa438b6")

TARGET = ("DeltaServe interleaves a LoRA fine-tuning backward pass with ongoing "
          "inference on the same GPU so the accelerator stays saturated.")
CONTROL = ("The quick brown fox jumps over the lazy dog near the river bank "
           "while the sun sets behind the distant mountains.")


def post_json(port, path, payload):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def post(port, path):
    try:
        with urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}{path}", method="POST", data=b""), timeout=5) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


def health(port, timeout_s=240):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1)
    return False


def seq_logprob(port, text):
    """Total logprob the model assigns to `text` (sum of input-token logprobs)."""
    obj = post_json(port, "/generate", {
        "text": text,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "return_logprob": True, "logprob_start_len": 0,
    })
    if isinstance(obj, list):
        obj = obj[0]
    itl = obj.get("meta_info", {}).get("input_token_logprobs") or []
    return sum(lp for (lp, *_rest) in itl if lp is not None)


def train_once(port, text):
    post_json(port, "/generate", {
        "text": text,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "is_finetuning": True,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30320)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--subprocess", action="store_true",
                    help="run the backward in an MPS-capped subprocess (needs MPS daemon up) "
                         "— exercises the unified S12a+§13 path")
    args = ap.parse_args()

    log = _PORT_DIR / "output" / "server_publish.log"
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", MODEL,
           "--host", "127.0.0.1", "--port", str(args.port), "--tp-size", "1",
           "--mem-fraction-static", "0.5", "--disable-radix-cache",
           "--disable-cuda-graph", "--enable-finetuning", "--backward-mps-percentage", "10"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
           "SGLANG_DS_REAL_BACKWARD": "1", "SGLANG_DS_PUBLISH_LORA": "1",
           "SGLANG_DS_FT_START_ON_LAUNCH": "0"}
    if args.subprocess:
        env["SGLANG_DS_BACKWARD_SUBPROCESS"] = "1"
        env["SGLANG_DS_BACKWARD_MPS_PCT"] = "10"
        env["SGLANG_DS_PUBLISH_EVERY"] = "5"
        env.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
        env.setdefault("CUDA_MPS_LOG_DIRECTORY", "/tmp/nvidia-mps-log")
    print(f"[pub] launch (publish ON, {'SUBPROCESS+MPS' if args.subprocess else 'in-process'}): {log}")
    proc = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT,
                            env=env, preexec_fn=os.setsid)
    try:
        if not health(args.port):
            print("[pub] server did not come up"); print("".join(open(log).readlines()[-30:])); return 3
        # confirm hooks attached (in-process or parent-publisher under MPS)
        logtxt = open(log).read()
        hooks = ("§13 publish ON" in logtxt) or ("§13 publish under MPS" in logtxt)
        print(f"[pub] publish hooks attached: {hooks}")

        L0 = seq_logprob(args.port, TARGET)
        C0 = seq_logprob(args.port, CONTROL)
        print(f"[pub] before: target logprob={L0:.2f}  control logprob={C0:.2f}")

        post(args.port, "/start_finetuning")
        print(f"[pub] overfitting target ×{args.steps} ...")
        for k in range(args.steps):
            train_once(args.port, TARGET)
            if (k + 1) % 50 == 0:
                Lk = seq_logprob(args.port, TARGET)
                print(f"[pub]   step {k+1}: target logprob={Lk:.2f}  (Δ={Lk-L0:+.2f})")

        L1 = seq_logprob(args.port, TARGET)
        C1 = seq_logprob(args.port, CONTROL)
        print(f"[pub] after:  target logprob={L1:.2f} (Δ={L1-L0:+.2f})  "
              f"control logprob={C1:.2f} (Δ={C1-C0:+.2f})")

        target_up = (L1 - L0) > 1.0
        target_beats_control = (L1 - L0) > (C1 - C0)
        ok = target_up and target_beats_control
        print(f"[pub] RESULT: {'PASS' if ok else 'FAIL'} — trained sample logprob "
              f"{'rose' if target_up else 'did NOT rise'}; "
              f"target Δ {'>' if target_beats_control else '<='} control Δ")
        return 0 if ok else 1
    finally:
        try: os.killpg(proc.pid, signal.SIGTERM); proc.wait(timeout=15)
        except Exception:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except Exception: pass


if __name__ == "__main__":
    sys.exit(main())
