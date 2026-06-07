# sglang DeltaServe port — co-serving benchmark report

**Status:** working sglang co-serving prototype on H200. Verified real LoRA
backward (gradcheck + in-server), 6 of 14 CO_SERVING_OPTIMIZATIONS sections
implemented (incl. backward in an MPS-capped subprocess). See `EXPERIMENTS.md`
for the full measured progression.

> **Installing the port?** See [INSTALL.md](INSTALL.md). This is a **fork of
> sglang `0.4.6.post5`** — the full patched source lives in [`sglang-fork/`](sglang-fork/)
> and `bash install.sh` just `pip install -e`'s it (no patch step). The
> `sglang-046-port.patch` is kept only as a concise diff of what we changed vs
> stock sglang.

## TL;DR

**sglang DeltaServe, REAL LoRA backward (verified), Llama-3-8B, H200, tight timeline:**

| Config | co-serving TTFT | Δ TTFT | co latency | Δ latency |
|---|---:|---:|---:|---:|
| sglang inf-only (baseline 17 ms / 497 ms) | — | — | — | — |
| + real bwd, in-process | 72 ms | +324% | 1391 ms | +180% |
| + real bwd, subprocess+MPS (S12a) | 37 ms | +118% | 1057 ms | +113% |
| + FT-decode keeps CUDA graph (eager-fix) | 39 ms | +129% | 735 ms | +48% |
| **+ SLO estimator live (gate)** | **31 ms** | +82% | **723 ms** | **+45%** |

Four levers stack: each real backward fire is ~94 ms on 8B. **MPS isolation** (S12a)
halves the TTFT overhead; the **eager-decode fix** (FT-decode batches keep the CUDA
graph) cuts co-serving **E2E latency 1057→735 ms**; the **live 3-regime SLO estimator**
(times every step, refits online, defers the backward when it would blow the TBT budget)
then trims TTFT 39→31 ms and latency to 723 ms. At **TTFT 31 ms / latency 723 ms,
sglang co-serving is at parity with DeltaServe-vLLM (29 ms / 697 ms)** — and keeps
faster inference (17 vs 29 ms) and a far tighter TTFT tail (p95 57 vs 269 ms).

### Apples-to-apples vs DeltaServe-vLLM (matched 2026-06-05)

Re-ran DeltaServe-vLLM (14/14 opts) on the **same** 8B model, **same** 224-req tight
timeline, same alpaca corpus, same rank-16, both under MPS, both with a verified-healthy
real FT backward (vLLM loss 4.74→3.10 in-window). The matched result **overturns** the
old "vLLM ≈ 0% / sglang +324%" framing (that compared sglang *in-process* to a favorable
older vLLM run):

| 8B, matched tight | inf TTFT | co TTFT (mean/p50/**p95**) | inf LAT | co LAT (mean) | FT trained |
|---|---:|---:|---:|---:|---:|
| **DSV-vLLM** 14/14 | 29 ms | 61 / 32 / **269** ms | 648 ms | **697 ms** | ~24.7k tok |
| **sglang** (MPS + eager-fix + SLO) | 17 ms | 31 / 30 / **57** ms | 497 ms | **723 ms** | ~3.9k tok |

**sglang co-serving is now at parity with vLLM — and ahead on several axes:**
- **Faster inference** (17 vs 29 ms TTFT, 497 vs 648 ms latency) — engine advantage.
- **Co-serving at parity**: TTFT 31 vs 29 ms (+7%), E2E latency 723 vs 697 ms (+3.7%) —
  down from the 50%+ gap earlier framings reported.
- **Far tighter TTFT tail** under load (p95 57 vs 269 ms) — MPS hard-isolation + the
  SLO gate deferring the backward when it would blow the TBT budget.
- The live 3-regime SLO estimator refits online (~9 refits/run) and is what trimmed
  the last TTFT gap. vLLM still trains ~6× more FT in the window (continuous
  store-driven vs sglang's request-tagged injection) — closing *that* (more FT
  throughput at equal SLO) is the remaining §8/§10 work. See `EXPERIMENTS.md`.

> **History:** an earlier version of this table reported +130%/+234% with a *faux*
> backward (~8 ms/fire placeholder). Those numbers understated the real cost — a
> real LoRA backward is ~94 ms/fire on 8B. The table above is the verified real
> backward (see "Task A" below).

> ⚠️ **Two benchmark-methodology fixes** (2026-06-05) that affect any co-serving
> measurement here: (1) the backward now fires only on the FT **prefill**, not on
> decode steps — a bug that was firing it 15× too often (7.3× latency win, see
> `EXPERIMENTS.md` A-bench-1b-fix); (2) FT requests draw **distinct** prompts from
> a real corpus (`--ft-corpus`) so they don't dedup in sglang's radix cache.

With the backward isolated in an MPS-capped subprocess (S12a, below), the 1B
co-serving TTFT overhead drops to +110% over inf-only while still training 96%
of FT samples. DSV-vLLM's fully-optimized stack (14/14) reaches near-zero
overhead; closing the rest of that gap is the roadmap below.

## Plots

| File | Content |
|---|---|
| `plots/sglang_vs_dsv_vllm_apples.png` | **The headline comparison** — TTFT + latency CDFs for sglang inf/co + DSV-vLLM inf/co, all 4 on same axes |
| `plots/sweep_1b_vs_8b.png` | TTFT + latency vs FT% across Llama-3.2-1B and Llama-3-8B on H200 |
| `plots/tight_co_5panel_8b.png` | 5-panel DeltaServe-vLLM-style plot for our sglang 8B co25% run |
| `plots/sglang_vs_vllm_8b_5panel.png` | Stitched: our 8B 5-panel above DSV-vLLM's reference plot |
| `plots/sweep_summary.png` | TTFT + latency vs FT% across tight/loose timelines (Llama-3.2-1B) |
| `plots/loose_co_5panel.png` | 5-panel for loose timeline |
| `output/co_serving_comparison.png` | 3-panel CDF summary (Llama-3.2-1B) |

## Reproducing the apples-to-apples

### sglang port

```bash
# Install the patched sglang fork (editable; no patch step)
cd sglang-port/v046-port
bash install.sh                 # or: pip install -e "sglang-fork[all]"

# Run the benchmark (inf-only baseline)
python auto_benchmark_sglang.py --tight --port 30401

# Run co-serving (10% / 25% / 50% FT)
python auto_benchmark_sglang.py --co --tight --ft-fraction 0.25 --port 30402

# Output: output/timeline_results_tight_{inf,co}.csv
```

### DSV-vLLM reference

```bash
conda create -n dserve-vllm python=3.12 -y
conda activate dserve-vllm
pip install uv
cd /path/to/DeltaServe-vLLM/dserve-vllm
VLLM_VERSION_OVERRIDE=0.21.1rc1.dev123+g117afeea4.precompiled \
    VLLM_PRECOMPILED_WHEEL_COMMIT=117afeea4665367a3066c1df58d4082d07fcc946 \
    VLLM_USE_PRECOMPILED=1 \
    uv pip install --editable . --torch-backend=auto

# Lower gpu_memory_utilization for H200 (their default 0.75 OOMs)
sed -i 's/gpu_memory_utilization: 0.75/gpu_memory_utilization: 0.5/' \
    /path/to/DeltaServe-vLLM/configs/serving_config_finetuning_llama3.yaml

# Generate dummy LoRA weights (the repo's toy LoRAs are missing safetensors)
python /path/to/sglang-port/v046-port/scripts/gen_dummy_lora.py

# Run
cd /path/to/DeltaServe-vLLM
python eval/auto_benchmark.py --co --tight --timeline-gpu A100 \
    --model /path/to/Meta-Llama-3-8B --api-server-count 1
```

## What's implemented in our sglang port

From `CO_SERVING_OPTIMIZATIONS.md`:

| § | Section | Status | Where |
|---|---|---|---|
| 1 | Activation saves (memory-for-compute) | partial — 5 of 7 types | `sglang-fork/sglang/srt/deltaserve/accumulate.py` |
| 2 | CUDA graphs for backward (pre-captured) | ✅ | `sglang-fork/sglang/srt/deltaserve/faux_backward.py` `precapture_graph` |
| 3 | Defer LM-head to backward | ❌ | needs Task A |
| 4 | SLO-aware FT admission (3-regime estimator) | ✅ | `sglang-fork/sglang/srt/deltaserve/estimator.py` (ported) + `coserve_slo.py` (live in model_runner; opt-in gate `SGLANG_DS_SLO_GATE=1`) |
| 5 | Backward compute optimizations | ❌ | needs Task A |
| 6 | `_maybe_pause` GPU yield | ✅ (primitive only) | `sglang-fork/sglang/srt/deltaserve/gpu_grant.py` |
| 7 | Slice-based activation save fast path | ✅ | `sglang-fork/sglang/srt/deltaserve/accumulate.py` `_contig_slice_from_mask` |
| 8 | Async scheduling + reserve-at-inject | ❌ | needs Task A |
| 9 | Buffer / admission lifecycle | partial — `coordinator.reserve` exists | `sglang-fork/sglang/srt/managers/finetune_coordinator.py` |
| 10 | `forward_interruptible` (3-tier pre-emption) | ❌ | needs Task A |
| 11 | `/start_finetuning` endpoint + `disable_log_stats` | ✅ (the endpoint) | `sglang-fork/sglang/srt/deltaserve/gates.py` + `http_server.py` patches |
| 12 | Backward subprocess + MPS isolation | ✅ (first half) | `sglang-fork/sglang/srt/deltaserve/backward_process.py` + `backward_client.py` (CUDA-IPC zero-copy = second half, TODO) |
| 13 | Served-LoRA hot-publish | ✅ (in-process) | `real_backward.attach_inference_hooks` — `SGLANG_DS_PUBLISH_LORA=1`; subprocess publish = follow-up |
| 14 | Eval tooling (`auto_benchmark.py`, plots) | ✅ | `auto_benchmark_sglang.py` + `auto_plot_sglang.py` |

**7 of 14 sections fully implemented; 2 partial.** Task A (real backward) verified;
§12 first-half (subprocess+MPS) and §13 (served-LoRA publish, in-process) landed.
Remaining levers: §8 async scheduling, §10 forward_interruptible, §12 second-half
(CUDA-IPC zero-copy), and the §13 cross-process publish for the subprocess path.

## §13 — served-LoRA publish: training actually fine-tunes the served model

Until §13 the backward trained the LoRA masters **write-only** — inference used
base weights, so serving never reflected training. `SGLANG_DS_PUBLISH_LORA=1`
applies the trained delta `scaling·(x@Aᵀ)@Bᵀ` in the inference forward via hooks
on each layer's `qkv_proj`/`o_proj`, reading the live masters (no base merge — that
would break the backward's frozen-base rematerialization).

**Proof** (`scripts/test_publish_lora.py`): overfit one sample for 200 steps and
watch its *served* logprob, vs an untrained control:

| sample | logprob before | after | Δ |
|---|---:|---:|---:|
| **target (trained)** | −162.0 | −25.6 | **+136.5** |
| control (untrained) | −59.2 | −58.7 | +0.5 |

The served model now fits the trained sample dramatically better (monotone over
steps), while the control is untouched — training changed serving, specifically,
with no global corruption and a stable feedback loop. Verified with CUDA graph off.

**Publish also composes with S12a (MPS subprocess).** With
`SGLANG_DS_BACKWARD_SUBPROCESS=1` + `SGLANG_DS_PUBLISH_LORA=1`, the child ships its
trained masters back every `SGLANG_DS_PUBLISH_EVERY` fires and the parent applies
them through the same hooks — so you get **MPS-isolated backward that also fine-tunes
serving**. Verified: target served logprob −162→−143 (monotone), control flat. (Under
the back-to-back overfit stress, drop-on-busy backpressure shed most fires, so the
gain is smaller than the unthrottled in-process run — see `EXPERIMENTS.md`.)

## Task A — real LoRA backward: DONE & verified (2026-06-05)

There are two backward paths, selected by `SGLANG_DS_REAL_BACKWARD`:

- **faux** (default): `sglang-fork/sglang/srt/deltaserve/faux_backward.py` runs
  backward-shaped GPU work (~8 ms/fire, sized to a real LoRA backward at
  s_max=256 on Llama-3-8B). No real gradients. Useful for isolating the
  scheduler/IPC plumbing from backward compute.
- **real** (`SGLANG_DS_REAL_BACKWARD=1`): `sglang-fork/sglang/srt/deltaserve/real_backward.py`
  — the full `process_backward` loop. Pulls base weights from
  `model_runner.model.layers`, holds fp32 LoRA masters (q/k/v/o rank 16/layer),
  walks `head_backward` + per-layer `layer_forward`/`layer_backward` over the
  captured activations, accumulates grads, runs fused AdamW, all on a dedicated
  `bwd_stream` that overlaps the next forward.

**Verification (both pass, on H200):**

- **Math correctness** — `scripts/verify_real_backward.py`. An autograd
  gradcheck shows the manual `layer_backward`/`head_backward` grads match
  `torch.autograd` to **2.5e-07** (worst relative error), across GQA (Hq≠Hkv),
  multi-sample packed batches, RoPE, RMSNorm and the LM-head CE. An overfit test
  shows manual-grad AdamW is numerically identical to autograd-grad AdamW
  (max divergence **0.024%** of loss over 60 steps).
- **In-server integration** — `scripts/integration_real_backward.py`. Launches
  Llama-3.2-1B with `--enable-finetuning SGLANG_DS_REAL_BACKWARD=1`, fires the
  real backward 24× on distinct FT samples: **0 NaN**, finite loss every fire,
  and interleaved greedy inference is **byte-identical** before/during FT — i.e.
  the backward's grad buffers / bwd_stream do not corrupt inference (the
  graph-pool-aliasing NaN trap is avoided).

**Cost:** steady-state **~45 ms/fire** on Llama-3.2-1B for a ~15-token prefill
(fire #1 ≈ 240 ms, one-time init). That's ~5–6× the faux's ~8 ms, so real
co-serving overhead is materially higher than the faux benchmark's +130% — the
motivation for the roadmap below (subprocess+MPS, backward-compute opt).

**Still open after Task A:** the trained fp32 LoRA masters are **not yet
published back into the forward path** (served-LoRA hot-publish, §13), so the
adapter trains but inference doesn't yet see it.

## S12a — backward in an MPS-capped subprocess (done, first half)

The in-process backward shares SMs with inference and can't yield (it runs on
the scheduler thread). S12a moves it into a separate process capped via MPS:

```bash
# 1. start the MPS daemon (one-time, per box)
export CUDA_VISIBLE_DEVICES=0
nvidia-cuda-mps-control -d
# 2. launch with the subprocess backward
python -m sglang.launch_server --model-path <llama3> --enable-finetuning ...
#   env: SGLANG_DS_REAL_BACKWARD=1 SGLANG_DS_BACKWARD_SUBPROCESS=1
#        SGLANG_DS_BACKWARD_MPS_PCT=10
# or via the benchmark: auto_benchmark_sglang.py --co --real-backward
#        --backward-subprocess --backward-mps-pct 10
```

`model_runner` spawns the child, dumps its already-correct base weights to
`/dev/shm`, and the child loads them and runs the **verified** backward
(bit-exact vs in-process). Snapshots are shipped fire-and-forget with
drop-on-busy backpressure, so the scheduler thread never waits.

**Result (Llama-3.2-1B, tight, 25% FT):**

| config | TTFT mean | latency mean | FT trained |
|---|---:|---:|---:|
| inf-only | 10 ms | 158 ms | — |
| co, in-process backward | 35 ms | 644 ms | 56/56 |
| **co, subprocess + MPS @10%** | **21 ms** | **536 ms** | 54/56 |

Co-serving TTFT overhead drops from +250% to +110% over inf-only (−40% TTFT)
while still training 96% of FT samples. The child runs in its own address
space, so it **cannot corrupt inference memory** — the graph-pool NaN risk is
structurally eliminated. The remaining cost is the CPU-roundtrip activation
IPC; CUDA-IPC zero-copy (S12 second half) is the next lever.

## Roadmap (impact order)

The order below is "biggest TTFT/latency win first":

1. **Task A — real LoRA backward** (5-8 hr). Replaces faux. Unblocks
   Sections 3, 5, 8, 13. With real backward you can also verify the
   adapter actually trains (loss curves in `bwd_log.csv`).

2. **S12 first half — subprocess + MPS** (2-3 hr). Move the backward into
   `backward_process.py` subprocess (already exists, currently echo
   stub). Set `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=20` on the child. This
   alone closes most of the +130% TTFT gap — backward stops blocking
   inference.

3. **S8 — async scheduling + reserve-at-inject** (3-4 hr). Pipeline
   `schedule(N+1)` before `record(N)` so the batch queue keeps moving
   while the backward runs.

4. **S12 second half — CUDA-IPC zero-copy** (2-3 hr). Replace pickle/send
   of activations with shared CUDA tensors. Saves ~50 ms of IPC overhead
   per backward at s_max=256.

5. **S10 — forward_interruptible** (3-4 hr). Three-tier pre-emption:
   pre-schedule grace poll, post-schedule rollback, mid-forward abort.
   Catches late inference arrivals during FT-only steps.

6. **S4 second pass — real SLO predictor** (1-2 hr). 6-param step-time
   estimator from the doc, fitted online. Replaces our admit-rate
   heuristic with TTFT-budget-tuned admission.

Total to close the gap: ~16-25 hours of focused work.

## Files inventory

```
sglang-port/v046-port/
├── README.md                            ← you are here
├── BENCHMARK_RESULTS.md                 ← detailed run-by-run analysis
├── EXPERIMENTS.md                       ← experiment log (prediction vs actual)
├── INSTALL.md                           ← one-command install
├── install.sh                           ← pip install -e sglang-fork[all]
├── sglang-046-port.patch                ← diff of our changes vs stock sglang (reference)
├── sglang-fork/                         ← FULL patched sglang 0.4.6.post5 source (the fork)
│   ├── pyproject.toml                   sglang's own build/deps
│   └── sglang/srt/
│       ├── deltaserve/                  ← our co-serving runtime
│       │   ├── real_backward.py         Task A real LoRA backward + §13 publish hooks
│       │   ├── backward_process.py      S12a MPS-subprocess child
│       │   ├── backward_client.py       S12a parent + child→parent master sync
│       │   ├── accumulate.py            activation capture (Section 1/7)
│       │   ├── faux_backward.py         faux backward (Section 2 graph + Section 4 throttle)
│       │   ├── gates.py / gpu_grant.py / ft_injector.py / finetuning_store*.py
│       │   └── bwd_services/{base,llama3}.py   GQA backward math layer
│       ├── configs/finetune.py          FinetuneConfig
│       ├── managers/finetune_*.py       coordinator / scheduler mixin / step estimator
│       └── ...                          (the 10 stock files hooked for co-serving)
├── auto_benchmark_sglang.py             ← our sglang-targeted benchmark
├── auto_plot_sglang.py                  ← 5-panel plot matching DSV-vLLM layout
├── plot_co_serving.py                   ← 3-panel sweep CDF plot
├── plot_full_sweep.py                   ← 2-panel sweep summary plot
├── plot_apples_to_apples.py             ← sglang vs DSV-vLLM headline plot
├── plots/
│   ├── sglang_vs_dsv_vllm_apples.png    ← headline comparison
│   ├── sweep_1b_vs_8b.png
│   ├── tight_co_5panel_8b.png
│   ├── sglang_vs_vllm_8b_5panel.png     ← stitched our 8B above DSV-vLLM 8B
│   ├── sweep_summary.png
│   └── loose_co_5panel.png
└── output/
    ├── timeline_results_tight_inf_g.csv    1B inf baseline (cuda-graph on)
    ├── timeline_results_tight_co{10,25,50}_g.csv  1B co-serving
    ├── timeline_results_tight_inf_8b.csv   8B inf baseline (our port)
    ├── timeline_results_tight_co{10,25,50}_8b.csv  8B co-serving (our port)
    ├── timeline_results_loose_inf_g.csv    loose-timeline 1B
    ├── timeline_results_loose_co{10,25}_g.csv
    ├── dsv_vllm_inf_8b.csv                 8B inf (DSV-vLLM reference)
    ├── dsv_vllm_co_8b.csv                  8B co-serving (DSV-vLLM reference)
    ├── dsv_vllm_bwd_log_8b.csv             DSV-vLLM backward log
    └── ... (variants: SLO throttle, admit-rate, lazy graph, etc.)
```

## How to read the headline plot

`plots/sglang_vs_dsv_vllm_apples.png`:

- **Solid lines** = our sglang port
- **Dashed lines** = DSV-vLLM reference
- **Blue / cyan** = inference-only
- **Red / orange** = co-serving (with FT load)

Look at the **dashed orange (DSV-vLLM co-serving) vs dashed cyan
(DSV-vLLM inference)** — they overlap. That's the goal: co-serving
without overhead.

Look at the **solid red (sglang co-serving) vs solid blue (sglang
inference)** — they're far apart. That's our gap.

---

## Branch: `feature/sglang-port-complete`

All code, patches, benchmarks, plots committed and pushed.
Latest commit: `a1d98d5` (apples-to-apples added).

For a deeper architectural read of the sglang port itself (which sglang
files we patched, how the per-token mask flows from request to forward
hook, why the gate uses an O_EXCL init marker, etc.), see
`BENCHMARK_RESULTS.md`.
