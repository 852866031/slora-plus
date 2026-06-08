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
    def __init__(self, sock, proc, weights_path: str, publisher=None, publish_every: int = 0):
        self._sock = sock
        self._proc = proc
        self._weights_path = weights_path
        self._outstanding = 0
        self._sent = 0
        self._dropped = 0
        # §13+S12a: parent-side publisher (holds LoRA masters + inference hooks).
        # The child ships trained masters every `publish_every` fires; we copy
        # them in so the hooks apply the adapter under MPS isolation.
        self._publisher = publisher
        self._publish_every = int(publish_every) if publisher is not None else 0
        self._syncs = 0

    def poll(self):
        """Worker-thread drain hook. Call once per forward step (even when not
        submitting) so the in-flight counter stays fresh and `is_busy()` —
        which the scheduler thread reads — self-clears when a backward finishes.

        中文：工作线程的"收割"钩子。每个 forward step 调用一次（即使本步不提交反向），
        这样在途计数 `_outstanding` 才能及时归零，调度线程读取的 `is_busy()` 才能在
        反向完成后自动解除繁忙状态。MUST 只在工作线程调用（zmq PAIR 套接字非线程安全）。
        """
        self._drain()

    def is_busy(self) -> bool:
        """Socket-free soft gate readable from ANY thread (plain int read is
        GIL-atomic). True while a backward is still in flight in the child, so
        the store-driven admit can pace new FT injection to the backward cadence
        (vLLM-faithful: fill the activation buffer, run one backward, then admit
        the next) instead of flooding prefills that get dropped on a busy child.

        中文：不触碰套接字的"软门控"，任何线程都能安全读取（Python 整数读取受 GIL 保护，
        是原子的）。当子进程里仍有反向在跑时返回 True，于是"语料驱动"的准入逻辑可以把新的
        微调注入节奏对齐到反向的节奏（与 vLLM 一致：先填满激活缓冲、跑一次反向、再准入下一批），
        而不是疯狂灌入大量 prefill 然后在繁忙的子进程上被丢弃。
        """
        return self._outstanding >= _MAX_INFLIGHT

    def _drain(self):
        """Non-blocking: pull completed-backward replies to free in-flight slots,
        and apply any synced LoRA masters to the parent-side publisher.

        中文：非阻塞地把"反向已完成"的回包取出来，释放在途名额；如果带回了同步的 LoRA
        master 权重，则应用到父进程侧的 publisher（供推理 hook 在 MPS 隔离下生效）。
        """
        import zmq
        while True:
            try:
                raw = self._sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception:
                break
            self._outstanding = max(0, self._outstanding - 1)
            if self._publisher is not None:
                try:
                    rep = pickle.loads(raw)
                    if isinstance(rep, dict) and rep.get("masters") is not None:
                        self._publisher.import_masters(rep["masters"])
                        self._syncs += 1
                except Exception as e:
                    logger.warning(f"[DeltaServe] master sync apply failed: {e}")

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
                "publish_every": self._publish_every,
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

        # §13+S12a: if publishing, build a parent-side holder with the inference
        # hooks. Its masters start at B=0 (zero delta) until the child's first
        # sync; then the hooks apply the trained adapter under MPS isolation.
        publisher = None
        publish_every = 0
        if os.environ.get("SGLANG_DS_PUBLISH_LORA", "0") == "1":
            try:
                from sglang.srt.deltaserve.real_backward import _RealBackward
                publisher = _RealBackward(model=model)
                publisher.attach_inference_hooks()
                publish_every = int(os.environ.get("SGLANG_DS_PUBLISH_EVERY", "10"))
                logger.warning(f"[DeltaServe] §13 publish under MPS: parent publisher "
                               f"built, syncing masters every {publish_every} fires")
            except Exception as e:
                logger.warning(f"[DeltaServe] parent publisher build failed: {e}")
                publisher = None
        return BackwardClient(sock, proc, wpath, publisher=publisher,
                              publish_every=publish_every)
    except Exception as e:
        logger.warning(f"[DeltaServe] backward subprocess start failed: {e} — in-process fallback")
        return None
