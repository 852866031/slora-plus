# sglang DeltaServe port — co-serving benchmark report

**Status:** working sglang co-serving prototype on H200. Real activation
capture, real backward subprocess plumbing, 5 of 14 CO_SERVING_OPTIMIZATIONS
sections implemented. Benchmarked apples-to-apples against the reference
DeltaServe-vLLM stack on the same hardware.

> **Installing the port?** See [INSTALL.md](INSTALL.md). The repo ships the
> delta over stock `sglang==0.4.6.post5` (17 drop-in files +
> `sglang-046-port.patch`), not a full sglang fork — `bash install.sh` applies
> it to a stock install.

## TL;DR

| Engine | Stack | inf-only TTFT | co-serving TTFT | Δ TTFT | Δ latency |
|---|---|---:|---:|---:|---:|
| sglang | baseline | **18 ms** | — | — | — |
| sglang | + 5 of 14 opts | — | 42 ms | **+130%** | **+234%** |
| DSV-vLLM | baseline | 24 ms | — | — | — |
| DSV-vLLM | + 14 of 14 opts | — | **22 ms** | -6% | -4% (≈ free) |

Bench config (both engines): H200 GPU, Llama-3-8B, tight timeline (224
reqs / ~25s span, 80-token prompts × 80-token output).

> ⚠️ **These sglang co-serving numbers are with the *faux* backward** (~8 ms/fire
> placeholder). The real LoRA backward is now implemented and verified (see
> "Task A" below) but costs ~45 ms/fire on 1B — the real-backward apples-to-apples
> re-run is in progress and will replace this table.

Two facts worth highlighting:

1. **Inference baseline favors sglang** — 18 ms vs 24 ms TTFT (24% faster).
   That gap is just sglang's normal inference advantage over vLLM; not a
   DeltaServe thing.
2. **DSV-vLLM's full optimization stack achieves near-zero co-serving
   overhead** — co-serving TTFT/latency are within noise of inference-only.
   Their backward log shows 151 backward fires in 25s; inference never feels
   it. This is what production DeltaServe looks like.

Our sglang port currently pays +130% TTFT / +234% latency under the same
load because we implemented 5 of 14 optimizations (Sections 2, 4, 6, 7, 11
from `CO_SERVING_OPTIMIZATIONS.md`). The remaining 9 are blocked on real
GQA backward kernels (Task A — see "Roadmap" below).

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
# Apply the port to a system sglang 0.4.6.post5 install
SYS=$(python -c "import sglang, os; print(os.path.dirname(sglang.__file__))")
patch -p1 -d "$SYS/.." < sglang-port/v046-port/sglang-046-port.patch
cp -r sglang-port/v046-port/new-files/deltaserve "$SYS/srt/"
cp sglang-port/v046-port/new-files/finetune.py "$SYS/srt/configs/"
cp sglang-port/v046-port/new-files/finetune_*.py sglang-port/v046-port/new-files/step_time_estimator.py "$SYS/srt/managers/"

# Run the benchmark (inf-only baseline)
cd sglang-port/v046-port
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
| 1 | Activation saves (memory-for-compute) | partial — 5 of 7 types | `new-files/deltaserve/accumulate.py` |
| 2 | CUDA graphs for backward (pre-captured) | ✅ | `new-files/deltaserve/faux_backward.py` `precapture_graph` |
| 3 | Defer LM-head to backward | ❌ | needs Task A |
| 4 | FT admission (admit-rate + fire-throttle) | ✅ (heuristic) | `new-files/deltaserve/faux_backward.py` + scheduler.py |
| 5 | Backward compute optimizations | ❌ | needs Task A |
| 6 | `_maybe_pause` GPU yield | ✅ (primitive only) | `new-files/deltaserve/gpu_grant.py` |
| 7 | Slice-based activation save fast path | ✅ | `new-files/deltaserve/accumulate.py` `_contig_slice_from_mask` |
| 8 | Async scheduling + reserve-at-inject | ❌ | needs Task A |
| 9 | Buffer / admission lifecycle | partial — `coordinator.reserve` exists | `new-files/finetune_coordinator.py` |
| 10 | `forward_interruptible` (3-tier pre-emption) | ❌ | needs Task A |
| 11 | `/start_finetuning` endpoint + `disable_log_stats` | ✅ (the endpoint) | `new-files/deltaserve/gates.py` + `http_server.py` patches |
| 12 | CUDA-IPC zero-copy weight/activation sharing | ❌ | needs subprocess architecture |
| 13 | Served-LoRA hot-publish | ❌ | needs Task A + LoRAManager integration |
| 14 | Eval tooling (`auto_benchmark.py`, plots) | ✅ | `auto_benchmark_sglang.py` + `auto_plot_sglang.py` |

**5 of 14 sections fully implemented; 2 partial.** Bottleneck for the
remaining 7: real GQA backward kernels (Task A in the roadmap below).

## Task A — real LoRA backward: DONE & verified (2026-06-05)

There are two backward paths, selected by `SGLANG_DS_REAL_BACKWARD`:

- **faux** (default): `new-files/deltaserve/faux_backward.py` runs
  backward-shaped GPU work (~8 ms/fire, sized to a real LoRA backward at
  s_max=256 on Llama-3-8B). No real gradients. Useful for isolating the
  scheduler/IPC plumbing from backward compute.
- **real** (`SGLANG_DS_REAL_BACKWARD=1`): `new-files/deltaserve/real_backward.py`
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
adapter trains but inference doesn't yet see it; and the apples-to-apples TL;DR
above is still a *faux*-backward measurement — the real-backward re-run is in
progress.

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
├── sglang-046-port.patch                ← 377-line diff against system sglang
├── new-files/                           ← 13 new files to drop into sglang
│   ├── finetune.py                      Phase 1 FinetuneConfig
│   ├── finetune_coordinator.py          Phase 7 coordinator (S4, S9)
│   ├── finetune_scheduler_mixin.py      Phase 7 scheduler mixin
│   ├── step_time_estimator.py           Phase 8 estimator (stub)
│   └── deltaserve/
│       ├── accumulate.py                Section 1 + Section 7 hooks
│       ├── faux_backward.py             Section 2 graph + Section 4 throttle
│       ├── gates.py                     Section 11 /start_finetuning
│       ├── gpu_grant.py                 Section 6 maybe_pause primitive
│       ├── backward_process.py          Phase 6 subprocess scaffold
│       ├── ft_injector.py               Phase 3 injector
│       ├── finetuning_store.py          Phase 5 store
│       ├── finetuning_store_stub.py
│       └── bwd_services/
│           ├── base.py                  ABC
│           └── llama3.py                ← math layer ported, service class stub
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
