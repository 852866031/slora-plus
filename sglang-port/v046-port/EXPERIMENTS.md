# DeltaServe sglang-port — Experiment Log

Append-only. One entry per run. Written BEFORE launch; actuals filled in after.

---

### A-verify — offline correctness of the real LoRA backward math   (2026-06-05, H200 ×1)

**Reflection / audit.** The port shipped `real_backward.py` (full impl: fp32 LoRA
masters q/k/v/o per layer, head_backward + per-layer layer_forward/layer_backward
walk, fused AdamW, bwd_stream overlap) but the README described it as "faux only,
Task A not done" and it had **never been verified on GPU**. Before claiming it
trains, I need proof the manual backward computes correct gradients — the CLAUDE.md
precision rules (fp32 GQA scores bit-for-bit, RMSNorm fp32) say this is exactly
where llama3 backward bugs hide and silently plateau loss. Justified now because
every downstream optimization (S12/S8/S10) is meaningless on a wrong backward.

**Prediction (mine).** I expect the ported math to be correct (it's a line-by-line
port of the known-good DSV-vLLM kernels) → gradcheck max-rel-error < 2e-3.
FALSIFIED if any LoRA grad disagrees with autograd by >2e-3, or the GQA kv-repeat
path (Hq≠Hkv) shows a systematic error.

- Goal/criteria: manual `layer_backward`/`head_backward` grads == autograd to <2e-3;
  manual-grad AdamW curve == autograd-grad AdamW curve to <2e-3 over 60 steps.
- Command: `python scripts/verify_real_backward.py` (toy GQA Hq=4 Hkv=2 L=2,
  packed seq_lens=[5,3], rank=6, fp32).
- Predicted: gradcheck worst-rel ≈ 1e-6; overfit curves coincide. FALSIFIED if >2e-3.
- **Actual:** gradcheck worst-rel = **2.47e-07** (at L1.kA), Δloss=1.6e-7 → PASS.
  Overfit: manual 4.2187→3.5966, max|manual−autograd| = **8.68e-04 (0.024% of loss)**
  → PASS. Both pass, exit 0.
- Decision: math layer is correct, incl. GQA + multi-sample packing. Proceed to the
  in-server integration test (A-itest).

---

### A-itest — real backward runs inside the live server, inference stays sane   (2026-06-05, H200 ×1)

**Reflection / audit.** Math is proven (A-verify). Remaining risk is *integration*:
the backward runs on a separate CUDA stream (`bwd_stream`) concurrent with
inference, and CLAUDE.md's load-bearing memory-pool rule warns that backward grad
buffers aliasing the graph pool surface as **NaN in inference softmax that looks
unrelated**. So the integration test's real job is not "loss goes down" — it's
"does enabling real backward corrupt inference?". Loss descent is already proven
offline and can't show in-server anyway: the trained fp32 LoRA masters are not yet
published back into the forward (separate served-LoRA item), so per-fire loss
reflects the frozen base forward and is ~constant for a fixed prompt.

**Prediction (mine).** real_backward fires ≥1× with finite loss, no NaN, and greedy
inference output is byte-identical before vs during FT. FALSIFIED if any loss is
NaN/inf, the server crashes, or interleaved inference output changes/garbles.

- Goal/criteria: ≥1 finite `[DeltaServe] real_backward` fire, 0 NaN/inf losses,
  interleaved greedy inference byte-identical to warmup.
- Command: `python scripts/integration_real_backward.py --port 30310 --n-ft 24`
  → launches `python -m sglang.launch_server --model-path <Llama-3.2-1B-Instruct>
  --tp-size 1 --mem-fraction-static 0.5 --enable-finetuning
  --backward-mps-percentage 10` with `SGLANG_DS_REAL_BACKWARD=1
  SGLANG_DS_FT_START_ON_LAUNCH=0`, gate opened after warmup.
- Data: fixed FT prompt repeated ×24; probe prompt "The capital of France is" (greedy,
  16 tok) sampled at warmup + every 6th FT fire.
- Model/loss: Llama-3.2-1B-Instruct; trainable = attention LoRA q/k/v/o rank 16
  (fp32 master), AdamW lr 5e-6; per-sample shift-CE.
- Predicted: fires ≈ 24, all finite, inference unchanged. FALSIFIED if NaN/crash/garble.
- **Actual:** PASS (exit 0). fires=**24**, finite=24, nan/inf=**0**, inference byte-identical
  across all FT fires. Steady-state per-fire = **45.5ms mean** (min 42.8, max 58.5;
  fire #1 = 243ms one-time init). Losses 2.99–4.01 (vary by distinct prompt; flat
  across fires as expected — LoRA not yet published to forward).
  - **Two harness findings, not backward bugs:**
    (1) With sglang radix cache ON, a fixed FT prompt dedups to a *single* fire
        (23/24 prefills served from cache → no forward to capture). Real FT corpora
        are distinct, so this is a test artifact; `--disable-radix-cache` gives 24/24.
    (2) Greedy decode (temperature=0) is NOT byte-deterministic across two identical
        requests with radix cache ON (cache-prefill vs fresh-prefill numeric drift
        flips an argmax). Byte-identity corruption check requires `--disable-radix-cache`.
- Decision: **Task A verified** — real backward is correct (A-verify) AND runs in-server
  without corrupting inference (A-itest). Cost is 45.5ms/fire on 1B (~5–6× the faux's
  ~8ms) → real co-serving overhead will exceed the faux benchmark's +130%; this is the
  motivation for S12a (subprocess+MPS) + backward-compute opt. Next: re-run the
  apples-to-apples benchmark with real backward (#48), using a DISTINCT FT corpus so
  the backward fires per sample under production radix-cache-on inference.
