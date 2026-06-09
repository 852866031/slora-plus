"""DeltaServe backward subprocess — child entry point and parent-side spawner.

S12a: the child runs the REAL LoRA backward in its own process so it can be
MPS-capped (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE on the child only) and so its
GPU work no longer shares the scheduler thread / inference SMs uncontrolled.

Protocol (pickled dicts over a ZMQ PAIR socket; the child binds, parent connects):
  parent → child:
    {"op":"init", "weights_path": <path>, "lora_rank":r, "lora_alpha":a,
     "lr":lr, "weight_decay":wd}      # one-time: torch.load the base-weight
                                       # state dumped by extract_base_state and
                                       # build the backward from it
    {"op":"backward", "snapshot": <dict of CPU tensors>}   # one FT prefill
    {"op":"shutdown"}
  child → parent:
    {"ok":True, ...} for init/shutdown; {"loss":x,"n_valid":n,"ms":t} for backward.

The parent should send "backward" fire-and-forget when it doesn't need to block
on the result (the child runs the backward async w.r.t. inference). The standalone
round-trip test uses request/reply to assert correctness.
"""

import os
import pickle
import subprocess
import sys
import time
from typing import Optional

import zmq

_MPS_PERCENTAGE_ENV = "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"


def main(channel_addr: str, model_name: str, mps_pct):
    mps_pct = int(mps_pct)

    # Bind the IPC socket *before* the heavy torch/sglang import so the parent's
    # connect succeeds immediately; the first recv blocks until import + (later)
    # weight load finish.
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PAIR)
    sock.bind(channel_addr)

    import logging
    logging.basicConfig(level=logging.WARNING)
    log = logging.getLogger("ds.bwd_child")
    log.warning(f"[bwd_child] up, mps_pct={os.environ.get(_MPS_PERCENTAGE_ENV)}")

    import torch  # noqa: heavy import, after socket bind
    from sglang.srt.deltaserve.real_backward import build_real_backward_from_state

    rb = None
    try:
        while True:
            try:
                raw = sock.recv()
            except zmq.ContextTerminated:
                break
            msg = pickle.loads(raw)
            op = msg.get("op")

            if op == "shutdown":
                sock.send(pickle.dumps({"ok": True}))
                break

            if op == "init":
                t0 = time.monotonic()
                state = torch.load(msg["weights_path"], map_location="cpu", weights_only=False)
                rb = build_real_backward_from_state(
                    state, device="cuda",
                    lora_rank=msg.get("lora_rank", 16),
                    lora_alpha=msg.get("lora_alpha", 32.0),
                    lr=msg.get("lr", 5e-6),
                    weight_decay=msg.get("weight_decay", 0.01),
                )
                log.warning(f"[bwd_child] init: loaded weights + built backward "
                            f"in {time.monotonic()-t0:.1f}s")
                sock.send(pickle.dumps({"ok": True, "L": rb.L, "D": rb.D}))
                continue

            if op == "backward":
                if rb is None:
                    sock.send(pickle.dumps({"error": "not initialized"}))
                    continue
                snap = msg["snapshot"]
                # Move CPU snapshot tensors onto the child's GPU.
                snap = _snapshot_to_cuda(snap)
                sample_lens = msg.get("sample_lens", [])
                t0 = time.monotonic()
                elapsed = rb.process(snap, sample_lens=sample_lens)
                ms = (time.monotonic() - t0) * 1000
                # process() returns enqueue time; sync to report true GPU time when asked.
                if msg.get("sync"):
                    torch.cuda.synchronize()
                    ms = (time.monotonic() - t0) * 1000
                reply = {
                    "loss": float(rb._cum_loss / max(1, rb._call_count)),
                    "last_call_ms": ms, "calls": rb._call_count,
                }
                # §13+S12a: periodically ship the trained masters back so the
                # parent's inference hooks apply the adapter (publish under MPS).
                pub_every = int(msg.get("publish_every", 0))
                if pub_every > 0 and (rb._call_count % pub_every == 0):
                    torch.cuda.synchronize()  # masters must be post-step
                    reply["masters"] = rb.export_masters_cpu()
                sock.send(pickle.dumps(reply))
                continue

            sock.send(pickle.dumps({"error": f"unknown op: {op!r}"}))
    finally:
        sock.close(linger=0)
        ctx.term()


def _snapshot_to_cuda(snap: dict) -> dict:
    import torch
    out = dict(snap)
    li = snap.get("layer_in")
    if isinstance(li, dict):
        out["layer_in"] = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in li.items()}
    for k in ("final_in", "final_hidden", "concat_input_ids"):
        v = snap.get(k)
        if hasattr(v, "cuda"):
            out[k] = v.cuda()
    return out


def spawn_backward_process(
    channel_addr: str,
    model_name: str,
    mps_pct: int,
    env: Optional[dict] = None,
) -> subprocess.Popen:
    """Spawn the backward subprocess with the child-only MPS env applied.

    The parent's ``os.environ`` is *not* mutated — only the child's env dict.
    """
    child_env = dict(os.environ if env is None else env)
    # mps_pct >= 100 or <= 0 → UNCAPPED: don't pin the backward to a fixed MPS
    # partition. A fixed cap (e.g. 10%) permanently FLOORS FT throughput — even
    # when inference is idle the backward can only use that % of the GPU, so it
    # can't fill the trough (FT throughput stays low with no inference, which is
    # backwards). Uncapped, the backward uses the idle GPU at full speed and the
    # SLO scheduler throttles FT when inference is busy (the real co-serving
    # design). 中文：固定 MPS 上限会把 FT 吞吐永久焊死，空闲也填不满谷；放开后反向能吃满
    # 空闲 GPU，繁忙时由 SLO 调度器节流。
    if 0 < int(mps_pct) < 100:
        child_env[_MPS_PERCENTAGE_ENV] = str(int(mps_pct))
    else:
        child_env.pop(_MPS_PERCENTAGE_ENV, None)
    return subprocess.Popen(
        [sys.executable, __file__, channel_addr, model_name, str(int(mps_pct))],
        env=child_env,
    )


if __name__ == "__main__":
    # When invoked as a script the in-tree sglang package may not be on
    # sys.path; walk up to the package root (deltaserve/ -> srt/ -> sglang/ -> root).
    _pkg_root = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )
    if _pkg_root not in sys.path:
        sys.path.insert(0, _pkg_root)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
