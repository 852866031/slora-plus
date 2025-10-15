# orchestrate_benchmark.py
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import string
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import aiohttp
import matplotlib.pyplot as plt

# ---------------- Defaults ----------------
DEFAULTS = {
    "server": "http://localhost:8000",
    "warmup": 5,
    "requests_per_second": 8,
    "duration_seconds": 5,
    "prompt_length": 50, #1:2
    "max_new_tokens": 20,
    "max_wait": 120.0,          # wait for server up
    "ft_poll_interval": 3.0,    # poll /finetuning_status every N seconds
    "ft_max_wait": 60.0,      # max time to wait for finetuning to finish
}

base_model = "huggyllama/llama-7b"
DEFAULT_LORA = "tloen/alpaca-lora-7b"


# ---------------- Prompt generation ----------------
def generate_random_sentence(length: int) -> str:
    """Generate a random sentence of `length` words."""
    words = []
    for _ in range(max(1, length)):
        word = "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
        words.append(word)
    return " ".join(words).capitalize() + "."


# ---------------- Request building ----------------
def make_payload(prompt: str, max_new_tokens: int) -> Dict:
    return {
        "model_dir": base_model,
        "lora_dir": DEFAULT_LORA,
        "inputs": prompt,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": int(max_new_tokens),
        },
    }


def prepare_schedule_rps(
    requests_per_second: int,
    duration_seconds: int,
    prompt_length: int,
    max_new_tokens: int,
) -> List[Tuple[int, float, Dict]]:
    """Precompute schedule for constant RPS load."""
    rps = max(0, int(requests_per_second))
    dur = max(0, int(duration_seconds))
    items: List[Tuple[int, float, Dict]] = []
    idx = 0

    for s in range(dur):
        if rps == 0:
            continue
        for i in range(rps):
            t_off = s + (i / rps) # time offset in seconds
            prompt = generate_random_sentence(prompt_length)
            items.append((idx, float(t_off), make_payload(prompt, max_new_tokens)))
            idx += 1

    items.sort(key=lambda x: x[1])
    return items


# ---------------- HTTP helpers ----------------
async def try_health(session: aiohttp.ClientSession, server: str, timeout_s: float = 1.0) -> bool:
    try:
        async with session.get(f"{server.rstrip('/')}/health", timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


async def try_generate_probe(session: aiohttp.ClientSession, server: str, timeout_s: float = 2.0) -> bool:
    try:
        async with session.post(
            f"{server.rstrip('/')}/generate",
            json=make_payload("ping", max_new_tokens=4),
            timeout=timeout_s,
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


async def wait_for_server(server: str, max_wait_s: float = 120.0, poll_period_s: float = 0.5) -> None:
    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        while True:
            if await try_health(session, server) or await try_generate_probe(session, server):
                print("[orchestrator] Server is up ✅")
                return
            if time.time() - t0 > max_wait_s:
                raise TimeoutError(f"Server didn't become healthy within {max_wait_s:.1f}s")
            await asyncio.sleep(poll_period_s)


# ---------- wait for finetuning finished ----------
async def wait_for_finetuning(
    server: str,
    poll_interval_s: float = DEFAULTS["ft_poll_interval"],
    max_wait_s: float = DEFAULTS["ft_max_wait"],
) -> None:
    """
    Poll POST /finetuning_status every `poll_interval_s` seconds until JSON {"finished":"true"}.
    API returns HTTP 200 for both finished and not finished.
    """
    url = f"{server.rstrip('/')}/finetuning_status"
    t0 = time.time()
    n = 0
    async with aiohttp.ClientSession(headers={"User-Agent": "OrchestratorFTPoller"}) as session:
        while True:
            n += 1
            try:
                async with session.post(url) as resp:
                    text = await resp.text()
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        payload = {}
                    finished = isinstance(payload, dict) and str(payload.get("finished", "")).lower() == "true"
                    if finished:
                        elapsed = time.time() - t0
                        print(f"[orchestrator] Finetuning finished after {elapsed:.1f}s (poll #{n}). ✅")
                        return
                    print(f"[orchestrator] Finetuning not finished yet (poll #{n}).")
            except Exception as e:
                print(f"[orchestrator] finetuning_status error: {e}")

            if time.time() - t0 > max_wait_s:
                raise TimeoutError(f"Finetuning didn't finish within {max_wait_s:.1f}s")
            await asyncio.sleep(poll_interval_s)


# ---------- include relative send time (t_rel) ----------
async def send_one(
    session: aiohttp.ClientSession,
    server: str,
    idx: int,
    payload: Dict,
    t0_ref: float,
) -> Tuple[int, float, float, str]:
    """
    Return (idx, t_rel, latency, status)
      t_rel   = time.monotonic() - t0_ref at the moment we send the request
      latency = wall time to get the full response
    """
    url = f"{server.rstrip('/')}/generate"
    t_send = time.monotonic()
    t_rel = t_send - t0_ref
    try:
        async with session.post(url, json=payload) as resp:
            body = await resp.read()
            latency = time.monotonic() - t_send
            try:
                data = json.loads(body)
                out = data.get("generated_text", ["<no-text>"])[0]
            except Exception:
                out = body.decode(errors="replace")
            print(f"[req {idx:04d}] @{t_rel:6.3f}s  {latency*1000:7.1f} ms", flush=True)
            return (idx, t_rel, latency, "ok")
    except Exception as e:
        latency = time.monotonic() - t_send
        print(f"[req {idx:04d}] @{t_rel:6.3f}s  FAILED after {latency*1000:7.1f} ms: {e}", flush=True)
        return (idx, t_rel, latency, "err")


async def warmup(server: str, n: int, prompt_length: int, max_new_tokens: int) -> None:
    print(f"[orchestrator] Warmup: sending {n} requests…")
    async with aiohttp.ClientSession(headers={"User-Agent": "OrchestratorWarmup"}) as session:
        tasks = [
            send_one(session, server, i, make_payload(generate_random_sentence(prompt_length), max_new_tokens), time.monotonic())
            for i in range(n)
        ]
        await asyncio.gather(*tasks)
    print("[orchestrator] Warmup done ✅")


# ---------------- concurrent schedule with exact due-times ----------------
async def run_schedule(server: str, schedule: List[Tuple[int, float, Dict]]) -> List[Tuple[int, float, float, str]]:
    """
    Execute the precomputed schedule with precise arrival times.
    All requests share a common start time (t0 = time.monotonic()) and each
    task sleeps until its absolute due time (t0 + t_off) before POSTing.
    Prints progress whenever a request finishes.
    """
    if not schedule:
        return []

    total = len(schedule)
    finished_count = 0
    lock = asyncio.Lock()

    t0 = time.monotonic()
    async with aiohttp.ClientSession(headers={"User-Agent": "OrchestratorLoad"}) as session:
        async def fire_at(idx: int, t_off: float, payload: Dict):
            nonlocal finished_count
            delay = (t0 + t_off) - time.monotonic()
            # did we consider the case delay < 0?, print something and check it
            if delay > 0:
                await asyncio.sleep(delay)
            result = await send_one(session, server, idx, payload, t0)

            # update finished count safely
            async with lock:
                finished_count += 1
                print(f"[progress] Finished {finished_count}/{total} requests", flush=True)
            return result

        tasks = [asyncio.create_task(fire_at(idx, t_off, payload)) for idx, t_off, payload in schedule]
        results = await asyncio.gather(*tasks)

    return results

# ---------------- Process control ----------------
def launch_server(requests_per_second: int, duration_seconds: int, enable_finetuning: bool):
    """Launch server wrapped with Nsight Systems."""
    import subprocess

    nsys_output = (
        f"nsys_report_rps{requests_per_second}"
        f"_dur{duration_seconds}s"
    )

    cmd = [
        sys.executable, "launch_server.py", "--nsys-output", nsys_output,
    ]
    if enable_finetuning:
        cmd.append("--enable-finetuning")
    print(f"[orchestrator] Launching server with Nsight:\n  {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
    return proc, nsys_output


def kill_server(proc) -> None:
    """
    Stop the launched server as if Ctrl-C was pressed.
    Sends SIGINT to the process group, then escalates if needed.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except Exception:
        pgid = None

    # Step 1: send Ctrl-C (SIGINT) to entire process group
    try:
        if pgid:
            os.killpg(pgid, signal.SIGINT)
        else:
            proc.send_signal(signal.SIGINT)
    except Exception:
        pass

    # Step 2: wait a bit for graceful shutdown
    try:
        proc.wait(timeout=20)
        return
    except Exception:
        pass

    # Step 3: escalate to SIGTERM
    try:
        if pgid:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
        return
    except Exception:
        pass

    # Step 4: final resort SIGKILL
    try:
        if pgid:
            os.killpg(pgid, signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        pass


def summarize(results) -> None:
    lat_ok = [lat for (_i, _t, lat, st) in results if st == "ok"]
    lat_all = [lat for (_i, _t, lat, _st) in results]
    if not lat_all:
        print("[orchestrator] No results collected.")
        return

    def pct(vals, p):
        vals = sorted(vals)
        k = int((p / 100.0) * (len(vals) - 1))
        return vals[k]

    print("\n== Latency Summary ==")
    print(f"Requests (ok/total): {len(lat_ok)}/{len(lat_all)}")
    if lat_ok:
        print(f"Mean    : {sum(lat_ok)/len(lat_ok):.4f} s")
        print(f"P50/P90 : {pct(lat_ok,50):.4f} / {pct(lat_ok,90):.4f} s")
        print(f"Min/Max : {min(lat_ok):.4f} / {max(lat_ok):.4f} s")


# ---------------- Artifacts: CSV + plot ----------------
def write_latency_csv(results, out_stem: str) -> Path:
    path = Path(f"{out_stem}.csv").resolve()
    with path.open("w") as f:
        f.write("idx,t_rel_s,latency_s,status\n")
        for idx, t_rel, lat, st in results:
            f.write(f"{idx},{t_rel:.6f},{lat:.6f},{st}\n")
    print(f"[orchestrator] Wrote CSV: {path}")
    return path

# ---------------- Main ----------------
async def main():
    ap = argparse.ArgumentParser(description="Launch server (nsys), warm up, then constant RPS load.")
    ap.add_argument("--server", default=DEFAULTS["server"])
    ap.add_argument("--warmup", type=int, default=DEFAULTS["warmup"])
    ap.add_argument("--requests-per-second", type=int, default=DEFAULTS["requests_per_second"])
    ap.add_argument("--duration-seconds", type=int, default=DEFAULTS["duration_seconds"])
    ap.add_argument("--prompt-length", type=int, default=DEFAULTS["prompt_length"])
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULTS["max_new_tokens"])
    ap.add_argument("--max-wait", type=float, default=DEFAULTS["max_wait"])
    ap.add_argument("--ft-poll-interval", type=float, default=DEFAULTS["ft_poll_interval"])
    ap.add_argument("--ft-max-wait", type=float, default=DEFAULTS["ft_max_wait"])
    ap.add_argument("--co", action="store_true")
    ap.add_argument("--inf", action="store_true")
    ap.add_argument("--enable-finetuning", action="store_true")
    ap.add_argument("--enable_nsys", action="store_true", help="Launch server with Nsight Systems")
    args = ap.parse_args()

    if args.co:
        args.enable_finetuning = True
    elif args.inf:
        args.enable_finetuning = False

    schedule = prepare_schedule_rps(
        requests_per_second=args.requests_per_second,
        duration_seconds=args.duration_seconds,
        prompt_length=args.prompt_length,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"[orchestrator] Prepared {len(schedule)} requests at {args.requests_per_second} RPS "
          f"for {args.duration_seconds}s.")

    proc, nsys_output = launch_server(
        args.requests_per_second, args.duration_seconds, args.enable_finetuning
    )
    nsys_rep = Path(f"{nsys_output}.nsys-rep").resolve()

    try:
        await wait_for_server(args.server, max_wait_s=args.max_wait)

        # (No finetuning wait here)
        #await warmup(args.server, args.warmup, args.prompt_length, args.max_new_tokens)
        results = await run_schedule(args.server, schedule)
        summarize(results)

        await wait_for_finetuning(
            args.server,
            poll_interval_s=args.ft_poll_interval,
            max_wait_s=args.ft_max_wait,
        )

        # artifacts
        if args.enable_finetuning:
            out_stem = "latency_co-serving"
        else:
            out_stem = "latency_inference"
        write_latency_csv(results, out_stem)

    finally:
        print("[orchestrator] Stopping server…")
        kill_server(proc)
        if args.enable_nsys:
            time.sleep(8)
            if nsys_rep.exists():
                print(f"\n✅ Nsight Systems report: {nsys_rep}")
            else:
                print(f"\n⚠️  Nsight report not found yet. Expected: {nsys_rep}")
        print("[orchestrator] Done.")
        os.system("pkill -f nsys")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[orchestrator] Interrupted — exiting…", flush=True)