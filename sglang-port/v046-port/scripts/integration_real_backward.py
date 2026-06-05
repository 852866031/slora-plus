#!/usr/bin/env python3
"""integration_real_backward.py — does the real LoRA backward run INSIDE the
live sglang server without NaN/crash, and does inference stay sane?

The math layer is already proven correct offline (scripts/verify_real_backward.py:
manual grads == autograd to 1e-7). This is the *integration* gate:

  1. launch Llama-3.2-1B with --enable-finetuning + SGLANG_DS_REAL_BACKWARD=1
  2. warmup inference (gate closed) — record greedy outputs
  3. POST /start_finetuning
  4. send FT-tagged requests → real_backward fires on the bwd stream
  5. interleave inference → confirm outputs are byte-identical to warmup
     (proves the backward's grad buffers / bwd_stream don't corrupt inference —
     the NaN-in-softmax trap from the graph-pool aliasing rule)
  6. parse the server log: real_backward must fire ≥1×, every loss finite

NOTE on "loss doesn't decrease in-server": real_backward trains fp32 LoRA
masters that are NOT yet published back into the forward path (that's the
separate served-LoRA hot-publish item). So the per-fire loss reflects the
frozen base forward and is ~constant for a fixed prompt. Loss *descent* is
already proven by the offline overfit. Here we only require FINITE loss +
non-corrupted inference.

Exit 0 iff: ≥1 finite real_backward fire AND inference outputs unchanged.
"""
from __future__ import annotations

import argparse, json, os, re, signal, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PORT_DIR = _HERE.parent
DEFAULT_MODEL = ("/mnt/weka/home/jianshu.she/.cache/huggingface/hub/"
                 "models--meta-llama--Llama-3.2-1B-Instruct/snapshots/"
                 "9213176726f574b556790deb65791e0c5aa438b6")


def post(port, path):
    try:
        with urllib.request.urlopen(urllib.request.Request(
                f"http://127.0.0.1:{port}{path}", method="POST", data=b""),
                timeout=5) as r:
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


def generate(port, prompt, max_new, is_ft):
    payload = {"text": prompt,
               "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True},
               "stream": False, "is_finetuning": is_ft}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/generate",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        obj = json.loads(r.read().decode())
    if isinstance(obj, list):
        obj = obj[0]
    return obj.get("text", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--port", type=int, default=30310)
    ap.add_argument("--n-ft", type=int, default=24)
    args = ap.parse_args()

    log_path = _PORT_DIR / "output" / "server_integration_real.log"
    log_path.parent.mkdir(exist_ok=True)
    cmd = [sys.executable, "-m", "sglang.launch_server",
           "--model-path", args.model, "--host", "127.0.0.1", "--port", str(args.port),
           "--tp-size", "1", "--mem-fraction-static", "0.5",
           # Radix cache off: (1) greedy decode becomes deterministic so the
           # byte-identity corruption check is valid (with the cache on, a 2nd
           # identical request prefills-from-cache and tiny numeric drift flips
           # an argmax), and (2) every FT request does a real prefill so the
           # backward fires per request instead of being deduped to one.
           "--disable-radix-cache",
           "--enable-finetuning", "--backward-mps-percentage", "10"]
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
           "SGLANG_DS_REAL_BACKWARD": "1",
           "SGLANG_DS_FT_START_ON_LAUNCH": "0"}
    print(f"[itest] launch: {' '.join(cmd)}")
    print(f"[itest] env: SGLANG_DS_REAL_BACKWARD=1 SGLANG_DS_FT_START_ON_LAUNCH=0")
    proc = subprocess.Popen(cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
                            env=env, preexec_fn=os.setsid)
    print(f"[itest] pid={proc.pid} log={log_path}")
    try:
        if not health(args.port):
            print("[itest] server did not come up; tail:", file=sys.stderr)
            print("".join(open(log_path).readlines()[-40:]), file=sys.stderr)
            return 3
        print("[itest] healthy")

        prompt = "The capital of France is"
        warm = [generate(args.port, prompt, 16, False) for _ in range(2)]
        print(f"[itest] warmup greedy out: {warm[0]!r}")
        assert warm[0] == warm[1], "greedy inference is nondeterministic before FT?!"

        ok_gate = post(args.port, "/start_finetuning")
        print(f"[itest] POST /start_finetuning -> {ok_gate}")

        # Fire FT requests; interleave inference every few to probe corruption.
        # Unique FT prompt per request — distinct samples don't get served from
        # sglang's radix cache, so each one has a real prefill forward to capture
        # FT activations from (a fixed prompt would dedup to a single fire).
        ft_bases = [
            "Paris is the capital and most populous city of France",
            "Photosynthesis converts light energy into chemical energy in plants",
            "The mitochondria is the powerhouse of the cell in biology",
            "Newton's second law states that force equals mass times acceleration",
            "The Pacific Ocean is the largest and deepest of Earth's oceans",
            "Shakespeare wrote thirty seven plays and one hundred fifty four sonnets",
        ]
        inf_after = []
        for i in range(args.n_ft):
            ft_prompt = f"{ft_bases[i % len(ft_bases)]} (sample {i})."
            generate(args.port, ft_prompt, 8, True)
            if i % 6 == 5:
                inf_after.append(generate(args.port, prompt, 16, False))
        print(f"[itest] sent {args.n_ft} FT reqs; {len(inf_after)} interleaved inference probes")

        time.sleep(2)  # let trailing backward log lines flush
        text = open(log_path).read()
        fires = re.findall(r"real_backward #(\d+): [\d.]+ms loss=([\-\d.naeinf]+)", text)
        n_fire = len(fires)
        losses = []
        for _, lv in fires:
            try:
                losses.append(float(lv))
            except ValueError:
                losses.append(float("nan"))
        finite = [l for l in losses if l == l and abs(l) != float("inf")]
        n_nan = sum(1 for l in losses if l != l or abs(l) == float("inf"))
        print(f"[itest] real_backward fires={n_fire} finite={len(finite)} nan/inf={n_nan}")
        if losses:
            print(f"[itest] loss range: first={losses[0]:.4f} last={losses[-1]:.4f} "
                  f"min={min(finite):.4f} max={max(finite):.4f}" if finite else
                  f"[itest] NO finite losses")

        inf_unchanged = all(o == warm[0] for o in inf_after)
        print(f"[itest] inference unchanged across FT: {inf_unchanged}")
        if not inf_unchanged:
            for j, o in enumerate(inf_after):
                if o != warm[0]:
                    print(f"   probe {j}: {o!r}  (warm: {warm[0]!r})")

        ok = (n_fire >= 1) and (n_nan == 0) and (len(finite) >= 1) and inf_unchanged
        print(f"\n[itest] RESULT: {'PASS' if ok else 'FAIL'} "
              f"(fires>=1: {n_fire>=1}, no nan: {n_nan==0}, inf-sane: {inf_unchanged})")
        return 0 if ok else 1
    finally:
        try:
            os.killpg(proc.pid, signal.SIGTERM); proc.wait(timeout=15)
        except Exception:
            try: os.killpg(proc.pid, signal.SIGKILL)
            except Exception: pass


if __name__ == "__main__":
    sys.exit(main())
