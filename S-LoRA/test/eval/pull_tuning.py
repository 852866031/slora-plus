# wait_finetuning.py
from __future__ import annotations

import argparse
import asyncio
import json
import time

import aiohttp


async def poll_finetuning(server: str, interval_s: float = 1.0) -> None:
    """
    Poll <server>/finetuning_status every `interval_s` seconds until finished.
    Your server returns:
      - 600 with {"status":"finished"} when done
      - 601 with {"error": "..."} while not finished
    """
    url = f"{server.rstrip('/')}/finetuning_status"
    started = time.time()

    async with aiohttp.ClientSession(headers={"User-Agent": "FinetuningPoller"}) as session:
        n = 0
        while True:
            n += 1
            try:
                async with session.post(url) as resp:
                    text = await resp.text()
                    try:
                        payload = json.loads(text)
                    except json.JSONDecodeError:
                        payload = {"raw": text}

                    if resp.status == 200 and isinstance(payload, dict) and payload.get("finished") == "true":
                        elapsed = time.time() - started
                        print(f"✅ Finetuning finished after {elapsed:.1f}s "
                              f"(poll #{n}, HTTP {resp.status}).")
                        return
                    else:
                        # Not finished yet; print a lightweight heartbeat
                        msg = payload.get("finished") if isinstance(payload, dict) else payload
                        print(f"[{n}] waiting… (HTTP {resp.status}) {msg}")
            except Exception as e:
                # Network hiccup—keep waiting
                print(f"[{n}] request failed: {e}")

            await asyncio.sleep(interval_s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Poll /finetuning_status until finished.")
    ap.add_argument("--server", type=str, default="http://localhost:8000",
                    help="Base URL of the inference server")
    ap.add_argument("--interval", type=float, default=2,
                    help="Polling interval in seconds (default: 1.0)")
    args = ap.parse_args()

    try:
        asyncio.run(poll_finetuning(args.server, args.interval))
    except KeyboardInterrupt:
        print("\nInterrupted — exiting…")