#!/usr/bin/env python3
"""auto_benchmark_sglang.py — DeltaServe-on-sglang timeline benchmark.

Adapted from DeltaServe-vLLM's eval/auto_benchmark.py:
- Launches `python -m sglang.launch_server` with --enable-finetuning when --co
  is set; otherwise plain inference.
- Drives requests off the same timeline CSV format as the vLLM harness
  (timestamp_s, prompt_length, max_new_tokens).
- Each request is a streaming POST /generate (sglang's native SSE) — TTFT is
  the time to the first `data:` chunk; latency is the time to the chunk
  with finish_reason != null.
- A configurable fraction of timeline rows are sent with is_finetuning=True
  to exercise the FT dispatch path.
- Writes per-request metrics to output/timeline_results<suffix>.csv with the
  same columns as the vLLM harness.

Usage:
    python auto_benchmark_sglang.py --co --tight --ft-fraction 0.1
    python auto_benchmark_sglang.py --tight                  # inference-only
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import csv
import json
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import random
import string
import aiohttp

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_TIMELINE_DIR = _REPO / "eval" / "llama3" / "timelines"
OUTPUT_DIR = _HERE / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def detect_gpu_subdir() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True, timeout=2.0,
        )
        name = (out.strip().splitlines() or [""])[0].upper()
        if "A100" in name:
            return "A100"
        if "5090" in name:
            return "5090"
        if "H200" in name or "H100" in name:
            return "A100"  # closest baseline; treat H-series like A100
    except Exception:
        pass
    return "A100"


@dataclass
class TimelineRow:
    timestamp_s: float
    prompt_length: int
    max_new_tokens: int


def load_timeline(path: Path) -> List[TimelineRow]:
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(TimelineRow(
                timestamp_s=float(r["timestamp_s"]),
                prompt_length=int(r["prompt_length"]),
                max_new_tokens=int(r["max_new_tokens"]),
            ))
    rows.sort(key=lambda r: r.timestamp_s)
    return rows


def build_server_cmd(model_path: str, port: int, co: bool, mps_pct: int,
                     enable_inference_cuda_graph: bool = True,
                     store_corpus: Optional[str] = None,
                     mixed_chunk: bool = False,
                     disable_radix: bool = False) -> List[str]:
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "--tp-size", "1",
        "--mem-fraction-static", "0.5",
    ]
    if not enable_inference_cuda_graph:
        cmd += ["--disable-cuda-graph"]
    if disable_radix and not co:
        # inference-only baseline: match the co-serving engine config (radix off)
        # for a fair apples-to-apples latency comparison.
        cmd += ["--disable-radix-cache"]
    if co:
        cmd += ["--enable-finetuning", "--backward-mps-percentage", str(mps_pct)]
        # DeltaServe: FT prefills must be FULL (every token recomputed so the
        # activation hooks fire). The radix prefix cache would reuse cached KV and
        # leave the FT activation buffer near-empty (n_valid collapses to ~1).
        # ChunkCache (radix disabled) pure-frees every req on finish — no FT cache
        # pollution, full FT prefills. Inference loses prefix-cache reuse (fine for
        # the co-serve benchmark; run the inf-only baseline the same way).
        cmd += ["--disable-radix-cache"]
        # vLLM-parity: store-driven FT via the production CLI knob (no env var).
        # FT is driven continuously from this corpus instead of client-tagged.
        if store_corpus:
            cmd += ["--finetune-data-path", store_corpus]
        # §1.1: fuse decode+prefill into one MIXED step so FT prefill can ride a
        # step that also carries inference decode (the EAGER (T_in>0,B_d>0,T_ft>0)
        # shape the Phase-D dual-SLO gate admits into).
        if mixed_chunk:
            cmd += ["--enable-mixed-chunk", "--chunked-prefill-size", "2048"]
    return cmd


def start_finetuning_via_gate(port: int) -> None:
    """POST /start_finetuning to open the gate. Section 11 of the optimization doc."""
    import urllib.request, urllib.error
    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                f"http://127.0.0.1:{port}/start_finetuning",
                method="POST",
                data=b"",
            ),
            timeout=5,
        ) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError) as e:
        print(f"[bench] start_finetuning POST failed: {e}", file=sys.stderr)
        return False


def wait_for_health(port: int, timeout_s: float = 180) -> bool:
    import urllib.request, urllib.error
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionResetError, OSError):
            pass
        time.sleep(1)
    return False


# Reusable prompt token used to hit a given target prompt_length (close enough
# for benchmarking — sglang's tokenizer turns " hello" into ~1 token).
_PROMPT_FILLER = " hello" * 2048


def make_prompt_1(prompt_length: int) -> str:
    # Each " hello" is ~1 token after BPE; chain enough to exceed target.
    return _PROMPT_FILLER[:prompt_length * 6][: prompt_length * 6]

def make_prompt(length: int) -> str:
    words = []
    for _ in range(max(1, length)):
        word = "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 5)))
        words.append(word)
    return " ".join(words).capitalize() + "."

# Distinct FT prompts drawn from a real finetuning corpus. The timeline's
# inference prompts are all one fixed length (→ identical text → radix-cached),
# which is fine for inference but would dedup FT prefills down to a single
# backward fire. Real FT samples are distinct, so FT-tagged requests draw a
# fresh corpus line each time → one real prefill (and one backward) per sample.
_FT_CORPUS: List[str] = []


def load_ft_corpus(path: Optional[str]) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.is_absolute():
        p = _REPO / "eval" / "llama3" / "data" / path
    if not p.exists():
        print(f"[bench] FT corpus not found: {p} — FT reqs will reuse timeline prompts",
              file=sys.stderr)
        return []
    lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    print(f"[bench] loaded {len(lines)} FT samples from {p}")
    return lines


@dataclass
class RequestResult:
    rid: str
    sent_t: float
    ttft_s: Optional[float]
    latency_s: Optional[float]
    chunks: int
    avg_tbt_s: Optional[float]
    worst_tbt_s: Optional[float]
    is_ft: bool
    prompt_length: int
    max_new_tokens: int
    completion_tokens: int
    error: Optional[str]


async def stream_one(session: aiohttp.ClientSession, port: int,
                     row: TimelineRow, is_ft: bool, sent_t: float,
                     ft_prompt: Optional[str] = None) -> RequestResult:
    # FT-tagged requests use a distinct corpus sample (when available) so each
    # one does a real prefill; inference requests use the timeline-shaped prompt.
    prompt = ft_prompt if (is_ft and ft_prompt) else make_prompt(row.prompt_length)
    payload = {
        "text": prompt,
        "sampling_params": {
            "max_new_tokens": row.max_new_tokens,
            "temperature": 0,
            "ignore_eos": True,
        },
        "stream": True,
        "is_finetuning": is_ft,
    }
    t0 = time.monotonic()
    ttft = None
    tbts: List[float] = []
    last_chunk_t = None
    chunks = 0
    completion_tokens = 0
    rid = ""
    err = None
    try:
        async with session.post(
            f"http://127.0.0.1:{port}/generate", json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            async for line in resp.content:
                if not line:
                    continue
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[len(b"data:"):].strip()
                if data == b"[DONE]" or not data:
                    continue
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                now = time.monotonic()
                if ttft is None:
                    ttft = now - t0
                if last_chunk_t is not None:
                    tbts.append(now - last_chunk_t)
                last_chunk_t = now
                chunks += 1
                meta = obj.get("meta_info") or {}
                rid = meta.get("id") or rid
                completion_tokens = meta.get("completion_tokens", completion_tokens)
                if meta.get("finish_reason") is not None:
                    break
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    latency = (time.monotonic() - t0) if ttft is not None else None
    return RequestResult(
        rid=rid, sent_t=sent_t, ttft_s=ttft, latency_s=latency, chunks=chunks,
        avg_tbt_s=(sum(tbts)/len(tbts) if tbts else None),
        worst_tbt_s=(max(tbts) if tbts else None),
        is_ft=is_ft, prompt_length=row.prompt_length,
        max_new_tokens=row.max_new_tokens, completion_tokens=completion_tokens,
        error=err,
    )


async def drive(port: int, timeline: List[TimelineRow], ft_fraction: float,
                t_anchor: float) -> List[RequestResult]:
    results: List[RequestResult] = []
    pending: List[asyncio.Task] = []
    sent_count = 0
    async with aiohttp.ClientSession() as session:
        for i, row in enumerate(timeline):
            target_t = t_anchor + row.timestamp_s
            now = time.monotonic()
            if target_t > now:
                await asyncio.sleep(target_t - now)
            is_ft = (i % max(1, int(round(1/ft_fraction))) == 0) if ft_fraction > 0 else False
            ft_prompt = (_FT_CORPUS[sent_count % len(_FT_CORPUS)]
                         if (is_ft and _FT_CORPUS) else None)
            sent_count += 1
            task = asyncio.create_task(
                stream_one(session, port, row, is_ft, time.monotonic() - t_anchor, ft_prompt)
            )
            pending.append(task)
        results = await asyncio.gather(*pending)
    return results


def write_csv(path: Path, results: List[RequestResult]):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rid", "sent_t", "ttft_s", "latency_s", "chunks",
            "avg_tbt_s", "worst_tbt_s", "is_ft", "prompt_length",
            "max_new_tokens", "completion_tokens", "error",
        ])
        for r in results:
            w.writerow([
                r.rid, f"{r.sent_t:.4f}",
                f"{r.ttft_s:.4f}" if r.ttft_s is not None else "",
                f"{r.latency_s:.4f}" if r.latency_s is not None else "",
                r.chunks,
                f"{r.avg_tbt_s:.4f}" if r.avg_tbt_s is not None else "",
                f"{r.worst_tbt_s:.4f}" if r.worst_tbt_s is not None else "",
                int(r.is_ft), r.prompt_length, r.max_new_tokens,
                r.completion_tokens, r.error or "",
            ])


# --- server lifecycle ------------------------------------------------------
# The launched server and its children (sglang::scheduler, sglang::detokenizer,
# the backward subprocess) all share the server's process group via setsid, so
# killing the group reaps the whole tree. We register cleanup on atexit AND on
# SIGINT/SIGTERM so a Ctrl+C, a kill, or an unhandled exception can't leave
# orphans holding the port + GPU memory (the old end-of-main kill only ran on a
# clean exit).
_server_proc: Optional[subprocess.Popen] = None


def _kill_server() -> None:
    global _server_proc
    p = _server_proc
    _server_proc = None  # idempotent: a second call (atexit after a signal) no-ops
    if p is None or p.poll() is not None:
        return
    try:
        pgid = os.getpgid(p.pid)
    except ProcessLookupError:
        return
    print(f"[bench] cleaning up server process group {pgid}...", file=sys.stderr)
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _install_cleanup_handlers(server_proc: subprocess.Popen) -> None:
    global _server_proc
    _server_proc = server_proc
    atexit.register(_kill_server)

    def _on_signal(signum, _frame):
        # Kill the server, then exit so atexit/finally unwind normally. Exit code
        # follows the shell convention (128 + signal number).
        _kill_server()
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/weka/home/jianshu.she/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6",
                    help="HF model path or repo id")
    ap.add_argument("--port", type=int, default=30200)
    ap.add_argument("--co", action="store_true",
                    help="Enable --enable-finetuning + send is_finetuning=true on a fraction of reqs.")
    ap.add_argument("--ft-fraction", type=float, default=0.1,
                    help="Fraction of timeline rows to mark as FT (only meaningful with --co).")
    ap.add_argument("--mps-pct", type=int, default=20)
    ap.add_argument("--slo-min-interval-ms", type=int, default=0,
                    help="Section 4 SLO throttle: minimum ms between faux backward fires (0=unthrottled).")
    ap.add_argument("--ft-admit-rate", type=float, default=1.0,
                    help="Section 4 admit-rate throttle: fraction of incoming FT-tagged reqs that keep the tag (0.0-1.0).")
    ap.add_argument("--disable-inference-cuda-graph", action="store_true",
                    help="Disable sglang's cuda-graph for inference batches too. By default inference batches DO use cuda graph (FT batches always bypass via _has_ft check in model_runner).")
    ap.add_argument("--real-backward", action="store_true",
                    help="Task A: use real LoRA backward kernels (vs faux). Slower per fire but produces real grads + loss curves.")
    ap.add_argument("--backward-subprocess", action="store_true",
                    help="S12a: run the real backward in an MPS-capped subprocess (needs MPS daemon). Tags output _sub.")
    ap.add_argument("--backward-mps-pct", type=int, default=10,
                    help="MPS thread %% for the backward child (with --backward-subprocess).")
    ap.add_argument("--ft-corpus", default="alpaca_1000_p95.txt",
                    help="File of distinct FT samples (one per line) for FT-tagged requests, "
                         "resolved against eval/llama3/data/. Distinct prompts avoid radix-cache "
                         "dedup so the backward fires per sample. Empty string = reuse timeline prompts.")
    ap.add_argument("--store-driven", action="store_true",
                    help="vLLM-parity: drive FT from the corpus via the server's "
                         "--finetune-data-path CLI knob (no env vars), and send NO "
                         "client FT tags (forces --ft-fraction 0). Exercises the "
                         "production store-driven default path end-to-end.")
    ap.add_argument("--no-radix", action="store_true",
                    help="Disable the radix prefix cache even for inference-only "
                         "(co-serving already disables it); use for a fair "
                         "apples-to-apples baseline vs co-serving.")
    ap.add_argument("--mixed-chunk", action="store_true",
                    help="§1.1: enable sglang --enable-mixed-chunk so FT prefill can "
                         "ride a step that also carries inference decode (the EAGER "
                         "regime Phase-D dual-SLO admission targets).")
    sg = ap.add_mutually_exclusive_group()
    sg.add_argument("--tight", action="store_true")
    sg.add_argument("--loose", action="store_true")
    ap.add_argument("--timeline-gpu", default=None, help="Override timelines/<gpu>/ subdir (default: auto-detect).")
    ap.add_argument("--launch-server", action="store_true", default=True,
                    help="Launch the server (default). Pass --no-launch-server to use an already-running server.")
    ap.add_argument("--no-launch-server", dest="launch_server", action="store_false")
    args = ap.parse_args()

    # vLLM-parity store-driven mode: FT comes from the server's corpus, not client
    # tags. Force ft_fraction=0 so the client sends pure inference.
    store_corpus = None
    if args.store_driven:
        args.ft_fraction = 0.0
        p = Path(args.ft_corpus)
        if not p.is_absolute():
            p = _REPO / "eval" / "llama3" / "data" / args.ft_corpus
        store_corpus = str(p)
        print(f"[bench] STORE-DRIVEN FT via --finetune-data-path {store_corpus} "
              f"(ft_fraction forced to 0)")

    global _FT_CORPUS
    if args.co and args.ft_fraction > 0:
        _FT_CORPUS = load_ft_corpus(args.ft_corpus)

    gpu = args.timeline_gpu or detect_gpu_subdir()
    shape = "tight" if args.tight else ("loose" if args.loose else "tight")
    tl_path = _TIMELINE_DIR / gpu / f"timeline_{shape}.csv"
    if not tl_path.exists():
        print(f"timeline missing: {tl_path}", file=sys.stderr)
        sys.exit(2)
    timeline = load_timeline(tl_path)
    print(f"[bench] gpu={gpu} shape={shape} rows={len(timeline)} co={args.co} ft_frac={args.ft_fraction}")

    server_proc = None
    log_path = None
    if args.launch_server:
        log_path = OUTPUT_DIR / f"server_{shape}_{'co' if args.co else 'inf'}.log"
        cmd = build_server_cmd(
            args.model, args.port, args.co, args.mps_pct,
            enable_inference_cuda_graph=not args.disable_inference_cuda_graph,
            store_corpus=store_corpus,
            mixed_chunk=args.mixed_chunk,
            disable_radix=args.no_radix,
        )
        print(f"[bench] launching: {' '.join(cmd)}")
        # Section 11: launch with FT gate CLOSED so warmup runs without FT cost.
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")}
        if args.co:
            env["SGLANG_DS_FT_START_ON_LAUNCH"] = "0"
            # Section 4: optional SLO throttle (minimum interval between faux fires)
            if args.slo_min_interval_ms > 0:
                env["SGLANG_DS_BACKWARD_MIN_INTERVAL_MS"] = str(args.slo_min_interval_ms)
            # Section 4 (admit-side): fraction of FT-tagged reqs that keep the tag
            if args.ft_admit_rate < 1.0:
                env["SGLANG_DS_FT_ADMIT_RATE"] = str(args.ft_admit_rate)
            # Task A: real LoRA backward (instead of faux)
            if args.real_backward:
                env["SGLANG_DS_REAL_BACKWARD"] = "1"
            # S12a: run the real backward in an MPS-capped subprocess
            if args.backward_subprocess:
                env["SGLANG_DS_BACKWARD_SUBPROCESS"] = "1"
                env["SGLANG_DS_BACKWARD_MPS_PCT"] = str(args.backward_mps_pct)
                # the child + this server must be MPS clients
                env.setdefault("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps")
                env.setdefault("CUDA_MPS_LOG_DIRECTORY", "/tmp/nvidia-mps-log")
        server_proc = subprocess.Popen(
            cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
            env=env, preexec_fn=os.setsid,
        )
        print(f"[bench] server pid={server_proc.pid} log= {log_path}")
        _install_cleanup_handlers(server_proc)
        if not wait_for_health(args.port):
            print("[bench] server failed to come up; tail log:", file=sys.stderr)
            try:
                print("".join(open(log_path).readlines()[-40:]), file=sys.stderr)
            except Exception:
                pass
            _kill_server()
            sys.exit(3)
        print(f"[bench] server healthy")

    # Drop a couple of warmup requests so the first timeline row doesn't see cold start.
    # Gate is closed during warmup (Section 11): is_finetuning=True gets dropped to inference.
    print(f"[bench] warmup (gate closed, ft_fraction=0)...")
    asyncio.run(drive(args.port, timeline[:3], ft_fraction=0.0, t_anchor=time.monotonic()))

    # Section 11: open the gate AFTER warmup, before the timeline runs.
    if args.co:
        ok = start_finetuning_via_gate(args.port)
        print(f"[bench] POST /start_finetuning → {'OK' if ok else 'FAILED'}")

    t_anchor = time.monotonic()
    # Wall-clock of the timeline anchor: lets FT fires (logged with wall=time.time()
    # by the server, a different process) be aligned onto the timeline clock —
    # timeline_sec(fire) = fire_wall - anchor_wall. (wall & monotonic tick alike.)
    print(f"[bench] timeline_anchor_wall={time.time():.3f}")
    print(f"[bench] running timeline...")
    results = asyncio.run(drive(args.port, timeline, ft_fraction=(args.ft_fraction if args.co else 0.0), t_anchor=t_anchor))

    suffix = f"_{shape}{'_co' if args.co else '_inf'}"
    if args.co and args.real_backward:
        suffix += "_real"
    if args.co and args.backward_subprocess:
        suffix += "_sub"
    out_csv = OUTPUT_DIR / f"timeline_results{suffix}.csv"
    write_csv(out_csv, results)
    print(f"[bench] wrote {out_csv}")

    n_ok = sum(1 for r in results if r.error is None)
    n_err = len(results) - n_ok
    ft_count = sum(1 for r in results if r.is_ft)
    ttfts = [r.ttft_s for r in results if r.ttft_s is not None]
    lats = [r.latency_s for r in results if r.latency_s is not None]
    if ttfts:
        ttfts.sort()
        print(f"[bench] ttft_s: mean={sum(ttfts)/len(ttfts):.3f}  p50={ttfts[len(ttfts)//2]:.3f}  p95={ttfts[int(len(ttfts)*0.95)]:.3f}")
    if lats:
        lats.sort()
        print(f"[bench] latency_s: mean={sum(lats)/len(lats):.3f}  p50={lats[len(lats)//2]:.3f}  p95={lats[int(len(lats)*0.95)]:.3f}")
    print(f"[bench] reqs ok={n_ok} err={n_err}; ft_tagged={ft_count}")

    _kill_server()


if __name__ == "__main__":
    main()
