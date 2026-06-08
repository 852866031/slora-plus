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

---

### A-bench-1b — real-backward co-serving overhead, in-process bwd_stream (the "before S12a" baseline)   (2026-06-05, H200 ×1)

**Reflection / audit.** Task A is verified correct + safe. Now measure the actual
co-serving cost of the real backward as it stands today (in-process, on bwd_stream,
NO MPS isolation) so S12a has a before/after. Discovery while setting this up: the
tight timeline is 224 rows ALL at prompt_length=80 → identical text → radix-cache
dedups every prefill (so the prior *faux* numbers are themselves under a radix-cache
confound, and a real-backward run there would barely fire). Fix: FT-tagged requests
now draw DISTINCT alpaca samples (`--ft-corpus alpaca_1000_p95.txt`, ~100-tok p50),
inference keeps the timeline prompt. Iterating on 1B (fast); 8B apples-to-apples is
the final step after the optimizations land.

**Prediction (mine).** With real backward at ~45ms/fire (and bigger at ~100-tok FT
samples → likely 80-150ms/fire) sharing SMs with inference and no MPS, co-serving
TTFT will blow up vs inf-only — I expect co TTFT ≥ 2× inf TTFT, worse than the faux
+130%. FALSIFIED if co TTFT ≈ inf TTFT (would mean the backward isn't actually
contending — e.g. not firing, or overlapping for free).

- Goal/criteria: quantify inf-only vs co-serving(real) TTFT mean/p95 on 1B tight.
  This is a measurement, not a pass/fail gate.
- Command (both, back-to-back, server auto-launched each):
  `python auto_benchmark_sglang.py --tight --port 30401`  (inf-only baseline)
  `python auto_benchmark_sglang.py --co --tight --real-backward --ft-fraction 0.25
   --ft-corpus alpaca_1000_p95.txt --port 30402`  (co, real backward, distinct FT)
- Data: tight timeline 224 reqs @ pl=80; FT = every 4th req, distinct alpaca sample.
- Model: Llama-3.2-1B-Instruct; real backward q/k/v/o rank16 fp32, AdamW lr5e-6,
  in-process bwd_stream (no subprocess, no MPS).
- Predicted: inf TTFT ~20-40ms; co TTFT ≥ 2× inf. FALSIFIED if co ≈ inf.
- **Actual (before fix):** inf-only TTFT mean=**10ms** p95=14ms, latency mean=158ms.
  co-serving(real, in-process) TTFT mean=**97ms** (p95=123, **~10× inf**), latency
  mean=**4670ms** (**~30× inf**). Prediction confirmed in direction; blowup far worse
  than 2×.
  - **Root cause found:** 838 backward fires from only 56 FT requests. n_valid histogram:
    only 36 fires are real prefills (n_valid 60+); 785 fires (94%) are spurious
    DECODE-step fires (n_valid 1–19) — when ≥2 FT requests decode together the dispatch
    fired the backward on unrelated single tokens (garbage grads) AND burned ~44ms each.
    838×44ms = 37s of backward stuffed into a 25s timeline → that IS the 30× latency,
    not inherently-slow backward.
- Decision: **fix the dispatch to fire only on FT prefill (EXTEND mode), never decode**
  (`model_runner._forward`: gate on `forward_mode.is_extend()`). SFT is forward-only →
  one backward per FT sequence. Re-measure (A-bench-1b-fix). This is a correctness fix
  (no more training on garbage) AND should be the dominant perf win — bigger than S12a.

---

### A-bench-1b-fix — prefill-only backward gate (the "after" for the decode-fire fix)   (2026-06-05, H200 ×1)

**Hypothesis.** Gating the backward to FT prefill only drops fires from ~838 to ~56
(one per FT sample), cutting ~94% of backward GPU time. Expect co-serving TTFT/latency
to collapse toward the inf-only baseline — most of the 30× was spurious decode fires.

- Command: `python auto_benchmark_sglang.py --co --tight --real-backward --ft-fraction
  0.25 --ft-corpus alpaca_1000_p95.txt --port 30403` (same as before; only the
  server-side dispatch gate changed).
- Predicted: fires ≈ 56 (down from 838); co TTFT mean ≲ 40ms (down from 97); latency
  mean ≲ 1s (down from 4.67s). FALSIFIED if fires stay high or latency stays ~4s.
- **Actual:** PASS, prediction hit. fires=**56** (53 real prefills n_valid≥20, 3 short
  samples), TTFT mean=**35ms** p95=75ms (was 97/123), latency mean=**644ms** (was 4670).
  **Latency 7.3× better, TTFT 2.8× better** from the prefill-only gate alone.
  - vs inf-only: co TTFT 35 vs 10 (3.5×), latency 644 vs 158 (4×) — residual is 56 real
    backward fires (~100ms each at n_valid~100) contending on shared SMs.
- Decision: land the gate in the patch + commit. Residual 3.5–4× is the target for
  S12a (MPS isolation) and backward-compute opt (graph the real attention bwd). Next
  realism lever: FT requests are forward-only in real SFT — sending max_new_tokens=1
  for FT would remove their 80 pointless decode steps (still adds decode load now).

---

### A-bench-8b — real-backward co-serving on Llama-3-8B (honest headline numbers)   (2026-06-05, H200 ×1)

**Reflection.** README TL;DR headline is still faux-backward on 8B. Now that the
real backward is verified + the prefill-only gate landed, get the honest 8B
inf-vs-co-real overhead (our own before/after; the vLLM cross-comparison is
confounded by the same radix-cache/uniform-prompt issue and would need a vLLM
re-run — marked separately). 8B backward will be heavier per fire than 1B
(2× layers, 2× head_dim) → expect bigger residual.

- Command: `auto_benchmark_sglang.py --tight --port 30501 --model <8B>` (inf) and
  `--co --tight --real-backward --ft-fraction 0.25 --ft-corpus alpaca_1000_p95.txt
   --port 30502 --model <8B>` (co).
- Predicted: inf TTFT ~30-60ms; co TTFT 3-6× inf; fires ≈ 56 (gate working).
  FALSIFIED if fires ≫ 56 (gate regressed) or OOM.
- **Actual:** inf-only TTFT mean=**17ms** p95=20ms, latency mean=**497ms**.
  co(real+gate) TTFT mean=**72ms** p95=137ms (**+324%**), latency mean=**1391ms**
  (**+180%**). fires=**56** (gate works on 8B), steady **94.2ms/fire** (max 102).
  Prediction hit (co TTFT 4.2× inf, in the 3-6× band).
- Decision: honest 8B headline = +324% TTFT / +180% latency with REAL backward +
  prefill gate (vs the old faux table's +130%/+234% — faux understated TTFT because
  faux fires are 8ms vs real 94ms). Update README TL;DR to these real numbers. Residual
  is 56×94ms = 5.3s of backward on shared SMs in a 25s window → S12a (MPS) is the
  structural fix. vLLM cross-comparison still needs a vLLM re-run under the distinct-FT
  / radix-aware methodology before it's apples-to-apples.

---

### S12a-derisk — does MPS thread-% capping protect inference latency?   (2026-06-05, H200 ×1)

**Reflection.** Before building the subprocess+IPC+MPS machinery (multi-hour, several
failure modes), validate the premise cheaply: on THIS H200 + CUDA, does MPS honor a
thread-% cap so a compute-hog co-tenant can't starve a latency-sensitive process?
Synthetic proxies (victim = 8B-prefill-shaped GEMM bursts; hog = 32-layer
backward-shaped GEMM sequence) on one GPU, 3 conditions. The in-process backward
can't yield (it runs on the scheduler thread — gpu_grant.maybe_pause would deadlock),
so a separate MPS-capped process is the only structural fix; this tests if it works.

- Command: `bash scripts/mps_probe.sh` (victim alone / +hog no-MPS / +hog MPS@10%).
- Predicted: no-MPS victim degrades ≥1.5×; MPS@10% recovers victim to ≲1.3× baseline
  while hog keeps useful throughput. FALSIFIED if MPS@10% ≈ no-MPS (cap not honored).
- **Actual:** victim mean latency — alone **5.10ms**, +hog no-MPS **10.03ms (1.97×)**,
  +hog MPS@10% **5.95ms (1.17×)**. Hog throughput 4303→956 fires/14s under the cap
  (still ~68/s, ample for FT). PASS — MPS honors the cap and protects the victim.
- Decision: **S12a is GO.** Build the backward as an MPS-capped subprocess. Real win
  is likely larger than this synthetic 1.97×→1.17× because the real 8B co-serving TTFT
  hit was 4.2× (heavier/burstier backward) — more contention to recover. Next: solve
  weight-sharing into the child (check first whether sglang's qkv layout == raw HF so
  the child can load weights independently and safely, else use CUDA-IPC).

---

### S12a-inserver — backward in MPS-capped subprocess, in-server (1B)   (2026-06-05, H200 ×1)

**Reflection.** M1 proved the subprocess backward is bit-exact + MPS-safe standalone.
Now wire it into the server (model_runner spawns the child, ships snapshots
fire-and-forget with drop-on-busy backpressure) and measure whether MPS isolation
recovers the co-serving TTFT. Baseline = in-process real backward (A-bench-1b-fix):
TTFT 35ms, latency 644ms vs inf-only TTFT 10ms / 158ms.

- Command: `auto_benchmark_sglang.py --co --tight --real-backward --backward-subprocess
  --backward-mps-pct 10 --ft-corpus alpaca_1000_p95.txt --port 30601` (MPS daemon up).
- Predicted: subprocess+MPS TTFT < in-process 35ms (toward inf-only 10ms); some FT
  samples dropped by backpressure. FALSIFIED if TTFT ≈ in-process (MPS not helping) or
  child errors / no fires.
- **Actual (ft0.25, matched):** co sub+MPS TTFT mean=**21ms** p95=30ms, latency
  mean=**536ms**. vs in-process 35ms/644ms and inf-only 10ms/158ms.
  child fires=**54** (of 56; 1 dropped by backpressure, ~96% trained), 0 errors,
  "backward subprocess ready L=16 D=2048 mps=10%".
  - **TTFT 35→21ms (−40%)**; co-serving TTFT overhead +250%→+110% over inf-only.
    latency 644→536ms (−17%, decode-heavy so less MPS-sensitive).
  - Bonus: backward in a separate address space → cannot corrupt inference memory
    (the graph-pool NaN risk is structurally eliminated, not just avoided).
- Decision: **S12a (first half) lands** — backward as MPS-capped subprocess, real
  TTFT win, training throughput preserved (96%). Regenerate patch + new-file
  drop-in (backward_client.py), update README. Second half (CUDA-IPC zero-copy
  activations, vs the current CPU-roundtrip) is the next S12 lever; also an 8B
  sub+MPS run to confirm the win scales.

---

### S12a-8b — subprocess+MPS scales to 8B   (2026-06-05, H200 ×1)

- Command: same as S12a-inserver but `--model <Llama-3-8B>` `--port 30603`.
- **Actual (8B tight, ft0.25):** co sub+MPS TTFT mean=**37ms** p95=73ms, latency
  mean=**1057ms**. vs in-process 72ms/1391ms and inf-only 17ms/497ms.
  child fires=**39/56** (≈30% dropped by backpressure — 8B backward ~94ms vs bursty
  FT arrivals), 0 errors, "subprocess ready L=32 D=4096 mps=10%".
  - **TTFT 72→37ms (−49%)**; overhead +324%→+118% over inf-only. Win scales like 1B.
  - Honest tradeoff: at 8B the heavier backward + drop-on-busy sheds ~30% of FT
    training throughput under bursts to protect inference latency. Motivates the
    next levers: CUDA-IPC zero-copy (cheaper per-fire IPC) and/or a small bounded
    queue (train slightly stale instead of dropping).
- Decision: S12a validated on both model sizes. Update README 8B TL;DR with the
  isolated row. Next: §12 second-half (CUDA-IPC) or §13 served-LoRA publish.

---

### S13-publish — does the trained LoRA actually affect serving?   (2026-06-05, H200 ×1)

**Reflection.** Until now the backward trained the LoRA masters write-only —
inference used base weights, so serving never reflected training (the per-fire
loss was flat by construction). §13 applies the trained delta in the inference
forward via hooks on qkv_proj/o_proj (NOT a base merge — that would break the
backward's frozen-base rematerialization). Proof = overfit one sample, watch its
SERVED logprob rise while a control sample stays put.

- Command: `python scripts/test_publish_lora.py --port 30320 --steps 200`
  (server: SGLANG_DS_REAL_BACKWARD=1 SGLANG_DS_PUBLISH_LORA=1, in-process backward,
  --disable-radix-cache --disable-cuda-graph, gate opened after baseline).
- Predicted: target logprob rises clearly with steps; control ≈ flat; no NaN/divergence.
  FALSIFIED if target flat (publish broken) or control moves as much (global corruption).
- **Actual:** PASS. target logprob −162.0 → −25.6 (**Δ +136.5 nats**), monotone over
  steps (+18.6 @50, +50.1 @100, +98.6 @150, +136.5 @200). control −59.2 → −58.7
  (Δ **+0.5**). Hooks attached confirmed; no NaN; feedback loop stable.
- Decision: **§13 works in-process** — training genuinely fine-tunes the served model,
  specific to the trained sample. Caveats: (1) verified with CUDA graph OFF; graph-on
  is expected to work (optimizer.step updates params in place, captured matmuls read
  the same tensor memory) but is untested here. (2) Works in the IN-PROCESS backward;
  the SUBPROCESS path (S12a) trains masters in the child's address space, so §13+S12a
  needs a cross-process publish (child → parent master sync) — that's the follow-up.

---

### S12a+S13 — served-LoRA publish UNDER MPS isolation (unified)   (2026-06-05, H200 ×1)

**Reflection.** S12a (MPS subprocess) and §13 (publish) were mutually exclusive:
the child trains masters in its address space; the publish hooks live in the
inference process. Unify: child ships masters back every K fires; parent holds a
publisher (_RealBackward(model) → hooks + masters, B=0 → zero delta until first
sync) and copies them in so the hooks apply the trained adapter under MPS.

- Command: `python scripts/test_publish_lora.py --subprocess --port 30321 --steps 200`
  (SGLANG_DS_BACKWARD_SUBPROCESS=1 + SGLANG_DS_PUBLISH_LORA=1, MPS daemon up,
  PUBLISH_EVERY=5; --disable-cuda-graph).
- Predicted: target served logprob rises (masters sync child→parent, hooks apply);
  control flat; smaller than in-process §13 due to backpressure drops. FALSIFIED if
  target flat (sync/publish broken).
- **Actual:** PASS. target logprob −162.0 → −143.1 (**Δ +18.9**, monotone:
  +0.9/+5.5/+11.8/+18.9 @ 50/100/150/200), control −59.2 → −59.1 (Δ +0.1).
  Smaller than in-process §13 (+136.5) because only **51/200 fires** landed — 101+
  dropped by drop-on-busy backpressure (sequential overfit outran the ~50ms child
  backward). Per-effective-step gain comparable; mechanism correct.
- Decision: **S12a and §13 compose** — MPS-isolated backward that also fine-tunes
  serving. The drops are the overfit-stress artifact (back-to-back requests); real
  FT load is spaced out. 7/14 sections, both headline features now combinable.

---

### vLLM-rematch — apples-to-apples vs DeltaServe-vLLM, MATCHED protocol   (2026-06-05, H200 ×1)

**Reflection.** The README's vLLM comparison (vLLM ≈0% co overhead, sglang +324%) was
apples-to-oranges: sglang IN-PROCESS (no MPS) vs a favorable older vLLM run on a
different timeline. User asked to verify. Re-cloned + reinstalled DeltaServe-vLLM
(precompiled vllm 0.21.1rc1.dev, dserve-vllm env), generated the missing dummy LoRA
weights, pointed it at the SAME 224-req tight timeline sglang used, same 8B model,
same alpaca corpus, same rank-16, both under MPS. Compared to sglang subprocess+MPS.

- Commands: `eval/auto_benchmark.py --tight` and `--co --tight --timeline-gpu A100
  --model <8B>` (DSV-vLLM env); sglang side from S12a-8b.
- vLLM backward verified HEALTHY through the window: loss 4.74→3.10 over 49 batches
  (24.7k tokens), continuous bwd_log; exitcode=1 was at shutdown only. (Harness post-
  proc threw NameError `cutoff_iso` AFTER writing the results CSV — a DSV-vLLM harness
  bug, doesn't affect the data.)
- **Actual (8B, matched 224-req tight, both MPS):**

  |                | inf TTFT | co TTFT mean/p50/p95 | inf LAT | co LAT mean/p50/p95 | FT trained |
  |----------------|---------:|---------------------:|--------:|--------------------:|-----------:|
  | DSV-vLLM 14/14 |  29.3 ms |   61 / 32 / **269**  | 648 ms  |  **697**/664/936    | ~24.7k tok |
  | sglang+MPS 7/14|  17.3 ms |   37 / 32 / **73**   | 497 ms  |  1057/1083/1188     | ~3.9k tok  |

- **Findings (mixed — neither dominates):**
  1. Inference baseline: **sglang faster** (17 vs 29 ms TTFT) — engine advantage.
  2. **vLLM wins E2E latency under co-serving**: +8% (648→697) vs sglang +113%
     (497→1057), AND it trained ~6× more FT tokens. Its async-schedule + interruptible
     stack (§8/§10, which the port lacks) keeps total request time smooth.
  3. **sglang+MPS wins the TTFT tail**: co p95 73 ms vs vLLM 269 ms — MPS hard-isolation
     protects first-token latency tightly; vLLM's tight-timeline TTFT has a real tail.
- Decision: replace the misleading "vLLM ≈0% / sglang +324%" with this matched table.
  The honest gap = E2E latency under co-serving (+113% vs +8%), which maps exactly to
  the unimplemented §8 (async sched) + §10 (forward_interruptible). Caveat retained: FT
  injection differs (continuous store-driven vs request-tagged) → ~6× FT-volume gap; a
  fully FT-volume-matched run would need the sglang port to drive continuous store FT.

---

### install-from-scratch — true clean-env install (found+fixed a dep bug)   (2026-06-06, H200 ×1)

**Reflection.** Earlier install checks used `--system-site-packages` (reused base
torch) or `--no-deps` (only proved the fork tree resolves). Neither is a true
from-scratch install. Did one: brand-new venv (no system packages),
`pip install -e sglang-fork[all]` pulling the entire stack from PyPI, then launch
a server and serve a request.

- **Bug found:** server crashed at model load with
  `ModuleNotFoundError: No module named 'transformers.masking_utils'`. Root cause:
  stock sglang 0.4.6.post5 leaves `compressed-tensors` UNPINNED, so a fresh resolve
  (2026-06) grabs `compressed-tensors==0.15.0.1`, which imports `transformers.masking_utils`
  — absent in sglang's pinned `transformers==4.51.1`. Base env only worked because it
  had the older compatible 0.9.x. The `--no-deps`/system-site-packages checks all
  missed it.
- **Fix:** pin `compressed-tensors<0.10` in the fork's pyproject (commit 7b10047).
- **Verified after fix:**
  - clean venv + `pip install -e sglang-fork[all]` → compressed-tensors resolves to
    **0.9.4**, `import sglang` + deltaserve + compressed_tensors all clean.
  - server boots Llama-3.2-1B `--enable-finetuning --backward-mps-percentage 10`:
    backward subprocess spawned (mps_pct=10), FinetuneAccumulator attached (34 hooks),
    "server is fired up", and `/generate` returns a coherent completion
    ("The capital of France is" → " Paris. The Eiffel Tower is located in Paris.").
- Decision: from-scratch install is now genuinely verified end-to-end (deps → import →
  server boot → serve). The dep-pin is the kind of bug only a true clean install catches.

---

### S-eagerfix — let FT-decode batches keep the CUDA graph (co latency)   (2026-06-06, H200 ×1)

**Reflection (loop iter 2).** Hunting the +113% co-serving E2E latency gap vs vLLM
(+8%). Found it's not mainly backward SM contention (that's MPS-isolated) — it's that
`_forward_raw` forced EAGER for ANY batch carrying FT tokens, including FT *decode*
steps. Since the prefill-only gate means decode captures nothing, forcing FT-decode
batches eager just strips the CUDA graph off the whole co-located batch → decode for
everyone in that batch slows down. Fix: gate eager on FT *prefill* only
(`not _ft_prefill`), matching the capture gate.

- Command: `auto_benchmark_sglang.py --co --tight --real-backward --backward-subprocess
  --backward-mps-pct 10 --ft-fraction 0.25 --ft-corpus alpaca_1000_p95.txt --port 30701` (8B, MPS up).
- Baselines: inf-only 17/497ms; co (before this fix) 37/1057ms; vLLM 29/697ms.
- Predicted: co latency drops from 1057 toward ~650-750ms (FT-decode now graphed),
  TTFT ~unchanged. FALSIFIED if latency stays ~1057.
- **Actual:** co latency mean=**735ms** (p50 768, p95 862), TTFT mean=39ms p50=31 p95=74.
  53 backward fires, 0 errors. vs co-before-fix 1057ms and vLLM 697ms.
  - **Co latency 1057→735ms (−30%)**; overhead vs sglang inf-only +113%→**+48%**.
  - **Now within +5.5% of vLLM's 697ms** (was +52%). sglang TTFT p95 73ms still far
    beats vLLM's 269ms. The eager-decode fix was the dominant lever — bigger than MPS.
- Decision: latency gap with vLLM essentially closed by this one fix. Commit+push+README.
  Remaining: estimator-gated dispatch (TTFT polish + honor the explicit SLO-port ask).

---

### S-slo-live — estimator live in dispatch + opt-in SLO gate   (2026-06-06, H200 ×1)

**Reflection (loop iter 4).** Latency goal already met (735ms ≈ vLLM 697). This
honors the explicit SLO-estimator ask: make the ported 3-regime estimator LIVE in
the real (request-tagged) dispatch — model_runner builds StepFeatures per step,
times each step with deferred CUDA events (no per-step sync), feeds note_step()
(online refit @256), and consults should_fire_backward() before the LoRA backward.
Gate is opt-in (SGLANG_DS_SLO_GATE=1); default-off path is byte-unchanged.

- Command: `SGLANG_DS_SLO_GATE=1 auto_benchmark_sglang.py --co --tight --real-backward
  --backward-subprocess --backward-mps-pct 10 --ft-fraction 0.25 --ft-corpus
  alpaca_1000_p95.txt --port 30702` (8B, MPS up).
- Predicted: estimator refits mid-run (log lines); latency stays ≤735 (lightweight
  timing, no regression); gate may shave a few backward fires under decode load →
  TTFT neutral-to-slightly-better. FALSIFIED if latency regresses >5% or no refit.
- **Actual:** TTFT mean=**31ms** p50=30 p95=57 (was 39/31/74), latency mean=**723ms**
  (was 735), refits=**9** (estimator trained online), 0 errors. Gate-ON improved BOTH:
  TTFT 39→31 (now +7%% of vLLM's 29; was +34%%), latency 735→723 (+3.7%% of vLLM's 697).
  p95 TTFT 57ms vs vLLM 269ms.
- Decision: SLO estimator is now fully ported AND live in dispatch, and it HELPS.
  **sglang co-serving is at parity with DeltaServe-vLLM** (TTFT 31 vs 29, latency 723
  vs 697) while keeping faster inference + far tighter TTFT tail. Goal met → wind down.

---

### S-slo-review — CORRECTION: the SLO gate was a no-op; estimator help was variance   (2026-06-06, code review)

**Self-review finding (user asked to review the code).** The S-slo-live entry above
attributed TTFT 39→31ms / latency 735→723ms to the SLO gate. That was WRONG:

- **The gate never fired a deferral.** It was consulted only on FT-prefill steps,
  where `build_step_features` sets `b_d=0`, but the only defer condition was
  `b_d > 0 and pred > TBT`. So `b_d==0` always → no defer ever → zero effect.
- Plus a double-count bug: `hyp.t_ft = baseline.t_ft + ft_tokens` on a baseline that
  already included `t_ft` (masked by the no-op).
- Therefore **735→723 / 39→31 was run-to-run variance, not the estimator.** The honest
  parity number is the EAGER-FIX result: **735 ms / 39 ms (+5.5% of vLLM's 697 ms)**.

**What's actually true:** the 3-regime estimator is correctly ported, live, and
refitting online (all regimes get samples). The *gate* is the part that didn't work.

**Fix:** gate now keys on the recent *decode* step time vs the TBT budget (the right
signal — the backward runs concurrently with decode, and the prefill step it's
consulted on has b_d=0); double-count removed; opt-in (`SGLANG_DS_SLO_GATE=1`),
`SGLANG_DS_SLO_DEFER_FRAC` tunable. **Honest caveat:** on this workload decode steps
are ~10–30 ms ≪ 150 ms TBT, so even the corrected gate rarely triggers — parity here
is from the eager-fix + MPS, NOT the SLO gate. The gate would matter on a heavier /
non-MPS (in-process) workload where the backward actually pushes decode toward TBT.
The coordinator's slo_gate_backward (same old double-count) is dead code — the live
path uses coserve_slo — left as-is, noted here.

---

### S-store-driven — vLLM-parity store-driven FT: backward-cadence admission gate   (2026-06-08)

**Goal.** Make sglang co-serving *behaviorally* identical to DeltaServe-vLLM:
FT comes from a tokenized **corpus store** with **SLO-gated, continuous** admission
(vLLM `ft_scheduler.admit_ft_to_step`), NOT client-request-tagged. Built opt-in
behind `SGLANG_DS_STORE_DRIVEN=1`; the working request-tagged default is untouched.

**Pieces (all committed earlier this loop):** `finetuning_corpus.py` (FinetuningStore,
faithful port), `ft_inject.py` (make_ft_req → prefill-only Req), `coserve_slo.py`
(3-regime estimator live), mixin `get_next_batch_to_run` admit+inject.

**Run 1 (store-driven, NO admission pacing) — 8B tight, MPS 10%:**
`SGLANG_DS_STORE_DRIVEN=1 SGLANG_DS_FT_DATA=alpaca_1000_p95.txt
auto_benchmark_sglang.py --co --tight --real-backward --backward-subprocess
--backward-mps-pct 10 --ft-fraction 0`.
- **Actual:** latency mean=**528ms**, TTFT **20ms**, 224 ok, 0 err, ft_tagged=**0**.
- **BUT only 5 backward fires, all in the first ~6s, then ZERO** for the remaining
  ~50s (estimator kept refitting 256→3328 steps → inference ran the whole time).
  Server log: `backward subprocess busy — dropped 1/51/101 FT samples (throttle)`.
- **Diagnosis:** the mixin paced injection on the FAST prefill cadence, not the SLOW
  backward cadence. FT prefills flooded the queue, prefilled+retired fast, but the
  single-flight backward subprocess **dropped** 100+ of them while busy. So FT
  stalled after 5 fires — and 528ms looked "faster than vLLM" only because FT barely
  ran. NOT parity. (Honesty check: a fast number with stalled FT is a regression.)

**Fix — admission gated on backward-subprocess idle (`BackwardClient.is_busy()`):**
- `BackwardClient.is_busy()` — socket-free in-flight read (GIL-atomic), safe from the
  scheduler thread. `BackwardClient.poll()` — worker-thread drain hook called every
  forward step in `model_runner._forward` so the counter self-clears even when not
  submitting (the zmq PAIR socket has a single owner = the worker thread).
- mixin `_admit_and_inject_ft` skips admit while `is_busy()` → injection cadence =
  backward cadence (vLLM-faithful fill-buffer → one backward → admit next).

**Run 2 (busy-gate) — same cmd:**
- Predicted: drops→~0, fires→continuous (~100+), latency rises toward vLLM (FT now
  contends all timeline). FALSIFIED if fires stay <10 or drops stay >50.
- **Actual:** latency mean=**687ms** p50=685 p95=797, TTFT **33ms**, 224 ok, 0 err,
  ft_tagged=0. **41 backward fires** (was 5), spread across the whole timeline,
  descending loss. **Drops capped at 1** (was 101+).
- **Parity:** inference **687ms vs vLLM 697ms (−1.4%)**, TTFT 33 vs 29ms, with FT
  training *continuously* from the store (ft_tagged=0) — behaviorally like vLLM, not
  the client-tagged shortcut. Latency rose 528→687 because FT now actually runs the
  whole window (real contention), which is the point.

**Honest caveat (fire count).** 41 fires vs vLLM's ~151 in the same window. This is
backward *throughput*, not behavior: the backward is single-flight in an MPS-10%
subprocess (~400–780ms each on 8B), deliberately throttled to protect inference. The
admission *mechanism* now matches vLLM; closing the 41→151 gap is a backward-speed
axis (S2 graphed backward / MPS-% tuning), tracked separately.

**Known limitation.** `claim` without `commit_claimed` on backward-done: samples are
served once then consumed (one pass through the corpus); fine for this benchmark
(360 samples ≫ 41 fires) but commit/retire wiring + the FT-only-idle drain edge
(poll needs a forward step) are the next increments before making the flag default.

---

### S-forward-interruptible / store-driven-by-default   (2026-06-08)

**Goal.** Finish "behaviorally identical to vLLM": make store-driven FT the
**default** (vLLM is store-driven), wire it through a production CLI knob (not just
env), and decide what of vLLM's `forward_interruptible` 3-tier preemption is worth
porting.

**Decisions (with rationale).**
- **Store-driven is now the default** whenever a corpus is resolvable. New
  `--finetune-data-path` plumbs `server_args → FinetuneConfig.data_path`; the mixin
  treats store-driven as on unless `SGLANG_DS_STORE_DRIVEN=0`. With no corpus the
  store stays None and the legacy client-request-tagged path is byte-unchanged, so
  existing harnesses still work. `SGLANG_DS_FT_DATA` env still overrides.
- **Request-tag suppression**: when the store is active, client `is_finetuning`
  tags are dropped (the corpus drives FT) so a request can't be trained twice.
- **Tier A (inference-first admit guard) ported but OPT-IN** (`SGLANG_DS_FT_TIER_A=1`).
  Default OFF on purpose: as a blanket "skip FT while any inference is waiting" it
  would STARVE FT under a dense timeline (waiting queue almost always holds
  inference) and regress the proven continuous-fire throughput — and parity is
  already met without it (backward is MPS-isolated, FT prefills are tiny). vLLM uses
  would_step_be_ft_only only for a grace-poll decision, not a blanket admit gate.
- **Tier C (mid-forward abort) deliberately NOT ported.** It needs an input-socket
  thread + abort event + activation hooks that raise mid-forward + a runner rollback
  path, all wired into vLLM's schedule()/execute_model() split. On this co-serve
  workload the benefit is marginal: FT prefills are <=cap (256) tokens / ~ms, the
  backward is MPS-isolated in a subprocess, and inference parity (within ~2%) is
  already achieved. Ported the mechanism's cheap, high-value parts (store-driven
  pacing + tier A knob) instead of the heavy, low-ROI interrupt.

**Verification — 8B tight, via the production CLI path (NO env vars):**
`auto_benchmark_sglang.py --co --tight --store-driven --real-backward
--backward-subprocess --backward-mps-pct 10` → server gets `--enable-finetuning
--finetune-data-path <corpus>`, client sends ZERO FT tags.
- Predicted: store loads from config.data_path (no env), continuous fires, latency
  ~= the 687ms env-var run. FALSIFIED if store doesn't load or fires <10.
- **Actual:** `store-driven FT: loaded 360 samples` (from config, no env) ✓;
  latency mean=**711ms** p50=705 p95=828, TTFT **37ms**, 224 ok, **0 err**,
  ft_tagged=0, **35 continuous backward fires**, 1 drop, 0 tracebacks.
- **Parity holds on the default path:** 711ms vs vLLM 697ms (+2.0%), TTFT 37 vs
  29ms. Store-driven FT is now the out-of-the-box behavior, matching vLLM, with the
  request-tagged path preserved as a fallback when no corpus is configured.

**State of the port.** sglang DeltaServe co-serving is now behaviorally aligned with
DeltaServe-vLLM: continuous corpus-driven SLO-admitted FT by default, 3-regime
estimator live, epoch-correct claim/commit, inference within ~2% of vLLM. Remaining
deltas are throughput-side (41→151 fire gap = backward speed: S2 graphed backward /
MPS-% tuning) and the un-ported tier-C interrupt (marginal here), both documented.

---

### S-phaseD — full predictive dual-SLO FT admission (admit_ft_to_step)   (2026-06-08)

**Goal.** Replace the coarse greedy+busy-gate admission with vLLM's real 5-stage
iterative SLO controller (`ft_scheduler.admit_ft_to_step`), so FT throughput is
governed by a *predicted* dual TTFT/TBT budget — the controller behind the plan's
"FT fills inference troughs, backs off on bursts" success criterion.

**Implemented** (`finetune_scheduler_mixin.py`):
- `_current_step_features()` — decode part from `running_batch` (b_d, k=Σseqlen),
  prefill part from `waiting_queue` head within `max_prefill_tokens`, earliest
  waiting arrival (`time_stats.wait_queue_entry_time`) for the TTFT deadline.
- `admit_ft_to_step()` — stages 1-5: phase gate (`coserving_admission_phase`),
  baseline-without-FT prediction + TTFT/TBT headroom, iterative greedy with
  per-sample **EAGER-regime** prediction admitting while BOTH TTFT (waiting
  prefill) and TBT (running decode × `decode_only_safety_margin`) hold, then
  claim. Leaky-bucket + proportional shapers ported (default off). Cold-start →
  token-cap-only. Reads the same `get_slo()` estimator trained by `model_runner`
  in the worker thread (shared singleton, same process under overlap).

**Verification 1 — no regression (1B, store-driven CLI, no mixed-chunk):**
estimator ready=True (online refit, rmse ~0.0015), `admit_ft_to_step` 0 errors /
0 tracebacks, 211ms/14ms, 224 ok, ft_tagged=0, 116 continuous fires.

**Verification 2 — the gate provably responds to the SLO budget (8B tight, A/B/C
on `SGLANG_DS_MAX_TBT_SLO`):**

| Run | max_tbt_slo | backward fires | inference latency |
|---|---:|---:|---:|
| A loose  | 0.5 s   | 55 | 660 ms |
| B tight  | 0.02 s  | 53 | 661 ms |
| C xtight | 0.001 s | **13** | **535 ms** |

- Predicted: tightening TBT below the forward step time makes `admit_ft_to_step`
  back off (baseline/EAGER predictions exceed the budget → admit 0), dropping
  fires and freeing GPU for inference. FALSIFIED if fires don't drop.
- **Actual:** A→C fires **55→13 (−76 %)**, latency **660→535 ms** (FT contention
  removed) — the controller demonstrably trades FT throughput for inference
  headroom as the SLO budget tightens. The gate engages once the budget approaches
  the (small) forward step time.

**Honest caveats:**
1. **The gate is ~no-op at *realistic* TBT (0.1-0.15 s) on this hardware.** 8B
   forward steps are sub-20 ms (the 250-340 ms backward is async in the MPS
   subprocess, NOT part of `T_step`), so a normal TBT SLO never binds; FT cadence
   is set by the busy-gate/backward cadence + inference interleaving. The gate
   only bites when TBT is pushed near the step time (run C). On slower/bigger
   setups (or in-process backward) where co-serve steps approach the TBT budget,
   it would bind at realistic SLOs — that's the regime vLLM's reference plot is in.
2. **The nutanix trough-fill/burst-backoff A/B (the plan's headline success plot)
   is NOT reproduced** — `timeline_nutanix.csv` is proprietary/gitignored and
   absent from this tree (only the reference PNG is present). Runs A/B/C use the
   steady `tight` timeline; the budget-sweep above is the substitute proof that
   the controller gates correctly. Reproducing the exact anti-correlation plot
   needs the proprietary timeline + a workload that stresses TBT at realistic SLOs.

**Plan status after this:** A (estimator) ✅, C (live trace/refit) ✅, D
(admit_ft_to_step) ✅ implemented+verified-gating; B partial (busy-gate pacing,
not the full reserve/occupancy/reopen lifecycle); E (profiler), F (full config
parity), G (YAML loader), H (RPS/fwd-token throttles) NOT done. Mixed-chunk
(§1.1) available via `--mixed-chunk` but not yet A/B'd.

---

### S-phaseH — H.1 RPS burst throttle (model-free admission gate)   (2026-06-08)

**Goal.** A model-free, fast-reacting gate that closes FT admission under
inference arrival bursts — the piece that can produce the "back off FT during
bursts" behavior on hardware where the SLO gate is no-op (forward steps sub-20ms,
so a realistic TBT never binds — see S-phaseD caveat).

**Implemented** (`finetune_scheduler_mixin.py`, config `finetune.py`): a sliding-
window inference-arrival-rate tracker (`_rps_note_arrivals` hooked in
`process_input_requests`, inference reqs only) + `_rps_check_throttle` with
spatial (close_rps/open_rps band) + temporal (close_time) hysteresis + idle-bypass
(rps==0 releases immediately), gating the top of `_admit_and_inject_ft`. Config:
`rps_throttle_{enable,close_rps,open_rps,window_s,close_time}`; env overrides
`SGLANG_DS_RPS_{THROTTLE,CLOSE,OPEN,WINDOW_S,CLOSE_TIME}` (usable before Phase G).

**NOTE:** the vLLM reference for this throttle (`check_rps_throttle`/`RpsTracker`)
is NOT present in the available DeltaServe-vLLM checkout — implemented from the
plan's prose spec (§H.1), not a line-by-line port.

**Verification (1B store-driven, tight timeline ~4.6 RPS):**
| Config | backward fires | inference latency |
|---|---:|---:|
| throttle OFF (baseline) | ~106–116 | ~204 ms |
| throttle ON (close_rps=3, open=2, window=1s) | **26** | **172 ms** |

- Predicted: with close_rps below the timeline's active RPS, the gate engages and
  FT admission closes → fewer fires + lower inference latency. FALSIFIED if fires
  don't drop.
- **Actual:** fires **~110 → 26 (−75 %)**, latency **204 → 172 ms**, 0 errors. The
  RPS throttle demonstrably trades FT throughput for inference headroom under load.

**Caveat / follow-up:** the steady `tight` timeline (~4.6 RPS) mostly holds the
gate *engaged* (RPS > close), so this shows the gate working but not the full
trough-fill/burst-backoff *dynamic*. Demonstrating the anti-correlation SHAPE
needs a constructed quiet→burst→quiet timeline (the proprietary nutanix one is
absent) + a temporal FT-throughput-vs-load plot — tracked as the remaining
demonstration. H.2 (fwd-token backward pause) not yet ported.
