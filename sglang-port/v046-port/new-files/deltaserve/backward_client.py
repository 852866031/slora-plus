"""Parent-side client for the S12a backward subprocess.

Lives in the scheduler process (where model_runner runs). On start it:
  1. extract_base_state(model) → torch.save to /dev/shm
  2. spawn the MPS-capped backward child (it owns the channel addr, so no
     cross-process address passing is needed)
  3. {init} the child with the weights path, wait until it's built

Per FT prefill, `submit(snapshot)` ships the activations to the child
fire-and-forget with **backpressure**: if the child is still busy with the
previous backward, the new sample is DROPPED (a throttle) rather than queued —
bounds child memory + keeps the backward from falling behind inference.

Enabled by SGLANG_DS_BACKWARD_SUBPROCESS=1 (default off → in-process Path C).
Requires SGLANG_DS_REAL_BACKWARD=1 and, for real SM isolation, a running MPS
daemon (CUDA_MPS_PIPE_DIRECTORY set in the env).
"""
from __future__ import annotations

import logging
import os
import pickle
from typing import Optional

logger = logging.getLogger(__name__)

_MAX_INFLIGHT = 1  # drop new FT samples while a backward is in flight


class BackwardClient:
    def __init__(self, sock, proc, weights_path: str):
        self._sock = sock
        self._proc = proc
        self._weights_path = weights_path
        self._outstanding = 0
        self._sent = 0
        self._dropped = 0

    def _drain(self):
        """Non-blocking: pull any completed-backward replies to free in-flight slots."""
        import zmq
        while True:
            try:
                self._sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception:
                break
            self._outstanding = max(0, self._outstanding - 1)

    def submit(self, snapshot: dict, sample_lens=None) -> bool:
        """Ship one FT-prefill snapshot to the child. Returns True if sent,
        False if dropped (child busy). Non-blocking w.r.t. the backward compute."""
        self._drain()
        if self._outstanding >= _MAX_INFLIGHT:
            self._dropped += 1
            if self._dropped % 50 == 1:
                logger.warning(f"[DeltaServe] backward subprocess busy — dropped "
                               f"{self._dropped} FT samples so far (throttle)")
            return False
        cpu_snap = _snapshot_to_cpu(snapshot)
        try:
            self._sock.send(pickle.dumps({
                "op": "backward", "snapshot": cpu_snap,
                "sample_lens": sample_lens or [],
            }))
        except Exception as e:
            logger.warning(f"[DeltaServe] backward submit failed: {e}")
            return False
        self._outstanding += 1
        self._sent += 1
        return True

    def shutdown(self):
        try:
            self._sock.send(pickle.dumps({"op": "shutdown"}))
            self._sock.poll(3000)
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5)
        except Exception:
            try: self._proc.kill()
            except Exception: pass
        try: os.remove(self._weights_path)
        except OSError: pass


def _snapshot_to_cpu(snap: dict) -> dict:
    out = {}
    li = snap.get("layer_in")
    if isinstance(li, dict):
        out["layer_in"] = {k: (v.detach().to("cpu", copy=True) if hasattr(v, "detach") else v)
                           for k, v in li.items()}
    for k in ("final_in", "final_hidden", "concat_input_ids"):
        v = snap.get(k)
        out[k] = v.detach().to("cpu", copy=True) if hasattr(v, "detach") else v
    return out


def maybe_start_backward_subprocess(model) -> Optional[BackwardClient]:
    """Returns a BackwardClient if SGLANG_DS_BACKWARD_SUBPROCESS=1, else None
    (caller falls back to the in-process backward)."""
    if os.environ.get("SGLANG_DS_BACKWARD_SUBPROCESS", "0") != "1":
        return None
    try:
        import torch
        import zmq
        from sglang.srt.deltaserve.real_backward import extract_base_state
        from sglang.srt.deltaserve.backward_process import spawn_backward_process

        logger.warning("[DeltaServe] starting backward SUBPROCESS (S12a)…")
        state = extract_base_state(model)
        wpath = f"/dev/shm/ds_bwd_state_{os.getpid()}.pt"
        torch.save(state, wpath)

        addr = f"ipc:///tmp/ds_bwd_{os.getpid()}"
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.PAIR)
        sock.connect(addr)

        mps_pct = int(os.environ.get("SGLANG_DS_BACKWARD_MPS_PCT", "10"))
        proc = spawn_backward_process(addr, "in-server", mps_pct, env=dict(os.environ))

        sock.send(pickle.dumps({
            "op": "init", "weights_path": wpath,
            "lora_rank": int(os.environ.get("SGLANG_DS_LORA_RANK", "16")),
        }))
        if not sock.poll(180000):
            logger.warning("[DeltaServe] backward child init timed out — falling back in-process")
            try: proc.kill()
            except Exception: pass
            return None
        r = pickle.loads(sock.recv())
        if not r.get("ok"):
            logger.warning(f"[DeltaServe] backward child init failed: {r} — in-process fallback")
            return None
        logger.warning(f"[DeltaServe] backward subprocess ready (L={r.get('L')} "
                       f"D={r.get('D')} mps={mps_pct}%)")
        return BackwardClient(sock, proc, wpath)
    except Exception as e:
        logger.warning(f"[DeltaServe] backward subprocess start failed: {e} — in-process fallback")
        return None
