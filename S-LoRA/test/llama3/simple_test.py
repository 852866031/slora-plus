#!/usr/bin/env python3
"""
orchestrate_run_once.py

- Launches launch_llama3.py in a separate process group
- Forces unbuffered child output so server prints are not "missing"
- Streams server logs live
- Forwards Ctrl+C (SIGINT) to the server process group and kills it *immediately-ish*
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from typing import Dict, Optional

import aiohttp

DEFAULTS = {
    "server": "http://localhost:9000",
    "timeline_csv": "/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/llama3/timeline_live.csv",
    "max_wait": 120.0,
    "ft_poll_interval": 3.0,
    "ft_max_wait": 60.0,
}

# ----------------------------
# Request helpers
# ----------------------------
def make_payload(prompt: str, base_model: str, lora_dir: str, max_new_tokens: int = 10) -> Dict:
    return {
        "model_dir": base_model,
        "lora_dir": lora_dir,
        "inputs": prompt,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": max_new_tokens,
        },
    }


async def try_health(session: aiohttp.ClientSession, server: str, timeout_s: float = 1.0) -> bool:
    try:
        async with session.get(f"{server.rstrip('/')}/health", timeout=timeout_s) as resp:
            return resp.status == 200
    except Exception:
        return False


async def try_generate_probe(
    session: aiohttp.ClientSession,
    server: str,
    base_model: str,
    lora_dir: str,
    timeout_s: float = 2.0,
) -> bool:
    try:
        async with session.post(
            f"{server.rstrip('/')}/generate",
            json=make_payload("ping", base_model=base_model, lora_dir=lora_dir, max_new_tokens=2),
            timeout=timeout_s,
        ) as resp:
            return resp.status == 200
    except Exception:
        return False


async def wait_for_server(
    server: str,
    base_model: str,
    lora_dir: str,
    max_wait_s: float = 180.0,
    poll_period_s: float = 0.5,
) -> None:
    t0 = time.time()
    async with aiohttp.ClientSession() as session:
        while True:
            if await try_health(session, server) or await try_generate_probe(session, server, base_model, lora_dir):
                print(f"[orchestrator] Server is up ✅ at {server}", flush=True)
                return
            if time.time() - t0 > max_wait_s:
                raise TimeoutError(f"Server didn't become healthy within {max_wait_s:.1f}s")
            await asyncio.sleep(poll_period_s)


async def send_one_request(
    server: str,
    prompt: str,
    base_model: str,
    lora_dir: str,
    max_new_tokens: int,
) -> str:
    url = f"{server.rstrip('/')}/generate"
    payload = make_payload(prompt, base_model=base_model, lora_dir=lora_dir, max_new_tokens=max_new_tokens)

    async with aiohttp.ClientSession(headers={"User-Agent": "RunOnceClient"}) as session:
        start = time.time()
        async with session.post(url, json=payload) as resp:
            body = await resp.read()
            latency = time.time() - start

            try:
                result = json.loads(body)
                generated = result.get("generated_text", ["<no-text>"])[0]
            except json.JSONDecodeError:
                generated = body.decode(errors="replace")

    print(f"[orchestrator] latency={latency*1000:.1f} ms", flush=True)
    return generated

async def exit_finetuning(session: aiohttp.ClientSession, server: str) -> bool:
    url = f"{server.rstrip('/')}/exit_finetuning"
    try:
        async with session.post(url) as resp:
            return resp.status == 200
    except Exception:
        return False
    
async def start_finetuning(session: aiohttp.ClientSession, server: str, timeout_s: float = 5.0) -> bool:
    """
    Call POST /start_finetuning on the given server.
    Returns True on success, False on failure.
    """
    url = f"{server.rstrip('/')}/start_finetuning"

    try:
        async with session.post(url, timeout=timeout_s) as resp:
            print("[orchestrator] start_finetuning response status:", resp.status)
            if resp.status == 200:
                return True
            return False
    except Exception:
        return False

# ----------------------------
# Process orchestration
# ----------------------------
def terminate_process_tree_fast(p, grace_s: float = 0.15) -> None:
    """
    Immediate-ish shutdown:
      - POSIX: SIGINT to process group, short grace, then SIGKILL
      - Windows: CTRL_BREAK_EVENT, short grace, then kill()
    """
    if p is None or p.poll() is not None:
        return

    if os.name == "posix":
        try:
            pgid = os.getpgid(p.pid)
            os.killpg(pgid, signal.SIGINT)
        except Exception:
            # fallback
            try:
                p.terminate()
            except Exception:
                pass

        # Very short grace period
        t0 = time.time()
        while time.time() - t0 < grace_s:
            if p.poll() is not None:
                return
            time.sleep(0.01)

        # Hard kill
        try:
            pgid = os.getpgid(p.pid)
            os.killpg(pgid, signal.SIGKILL)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launcher", default="launch_llama3.py", help="Path to launch_llama3.py")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--rank_id", type=int, default=0)
    ap.add_argument("--co", action="store_true")

    # request params
    ap.add_argument("--prompt", default="Instruction:\nSay hello in one short sentence.\n### Response: ")
    ap.add_argument("--max_new_tokens", type=int, default=10)

    # These should match what your server expects; override if needed
    ap.add_argument("--base_model", default="meta-llama/Meta-Llama-3-8B")
    ap.add_argument("--lora_dir", default="/home/jiaxuan/Documents/Projects/slora-plus/S-LoRA/test/llama3/adapters/llama3-toy-lora")
    #yzdnaufan/Llama-3-8b-Alpaca-Lora
    # wait params
    ap.add_argument("--max_wait_s", type=float, default=240.0)
    ap.add_argument("--poll_period_s", type=float, default=0.5)

    # log behavior
    ap.add_argument("--use-stdbuf", action="store_true", help="(POSIX only) wrap child with stdbuf -oL -eL")

    args = ap.parse_args()

    server = f"http://127.0.0.1:{args.port}"

    # Build launcher command
    cmd = [
        sys.executable,
        "-u",  # IMPORTANT: unbuffered child output so prints appear immediately
        args.launcher,
        "--port",
        str(args.port),
        "--rank_id",
        str(args.rank_id),
    ]
    if args.co:
        cmd.append("--enable-finetuning")

    # Optional: force line-buffering at the OS level for non-python output (POSIX only)
    if args.use_stdbuf and os.name == "posix":
        cmd = ["stdbuf", "-oL", "-eL"] + cmd

    print("[orchestrator] launching:", " ".join(cmd), flush=True)

    import subprocess

    p: Optional[subprocess.Popen] = None
    log_task: Optional[asyncio.Task] = None

    # Ensure child is unbuffered even if python -u isn't honored by some wrapper
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    stop_event = asyncio.Event()

    try:
        # Start server in its own process group so we can signal the whole group
        if os.name == "posix":
            p = subprocess.Popen(
                cmd,
                preexec_fn=os.setsid,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # line-buffer in parent reader
                env=env,
            )

        # Stream logs in the background
        async def pump_logs() -> None:
            assert p is not None
            assert p.stdout is not None
            loop = asyncio.get_running_loop()
            while True:
                line = await loop.run_in_executor(None, p.stdout.readline)
                if not line:
                    break
                print("[server]", line.rstrip(), flush=True)

        log_task = asyncio.create_task(pump_logs())

        # Forward Ctrl+C to the child process group and stop ASAP (POSIX)
        loop = asyncio.get_running_loop()

        def _on_sigint() -> None:
            if p is not None and p.poll() is None:
                terminate_process_tree_fast(p, grace_s=0.1)
            stop_event.set()

        try:
            loop.add_signal_handler(signal.SIGINT, _on_sigint)
        except NotImplementedError:
            # Windows: we rely on KeyboardInterrupt handler below
            pass

        # Wait for server (break early if Ctrl+C)
        waiter = asyncio.create_task(
            wait_for_server(
                server,
                base_model=args.base_model,
                lora_dir=args.lora_dir,
                max_wait_s=args.max_wait_s,
                poll_period_s=args.poll_period_s,
            )
        )
        stopper = asyncio.create_task(stop_event.wait())

        done, pending = await asyncio.wait({waiter, stopper}, return_when=asyncio.FIRST_COMPLETED)

        for t in pending:
            t.cancel()

        if stop_event.is_set():
            return

        # Send one request (also abort if Ctrl+C right before sending)
        if stop_event.is_set():
            return
        
        if args.co:
            print("[orchestrator] Starting finetuning via API call...", flush=True)
            async with aiohttp.ClientSession() as session:
                if not await start_finetuning(session, server):
                    print("[orchestrator] Failed to start finetuning", flush=True)
                    return
                else:
                    print("[orchestrator] Finetuning started successfully", flush=True)

        generated = await send_one_request(
            server=server,
            prompt=args.prompt,
            base_model=args.base_model,
            lora_dir=args.lora_dir,
            max_new_tokens=args.max_new_tokens,
        )
        print("\n=== GENERATED ===", flush=True)
        print(generated, flush=True)
        print("=================\n", flush=True)
        await asyncio.sleep(5)
        if args.co:
            print("[orchestrator] Exiting finetuning via API call...", flush=True)
            async with aiohttp.ClientSession() as session:
                if not await exit_finetuning(session, server):
                    print("[orchestrator] Failed to exit finetuning", flush=True)
                else:
                    print("[orchestrator] Exited finetuning successfully", flush=True)
        await asyncio.sleep(3)
    except KeyboardInterrupt:
        # Windows / fallback
        if p is not None:
            terminate_process_tree_fast(p, grace_s=0.1)
    finally:
        if p is not None and p.poll() is None:
            print("[orchestrator] shutting down server…", flush=True)
            terminate_process_tree_fast(p, grace_s=0.1)

        if log_task is not None:
            log_task.cancel()
            # Don't block forever on log_task cancellation
            try:
                await asyncio.wait_for(log_task, timeout=0.5)
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())