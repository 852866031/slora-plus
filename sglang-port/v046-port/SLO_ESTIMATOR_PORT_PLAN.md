# Porting DeltaServe-vLLM's SLO estimator + FT admission to the sglang integration

A phased implementation plan to replace the sglang port's stub SLO machinery
(rolling-mean `StepTimeEstimator`, always-`True` `can_admit`, static
random admit-rate) with the real three-regime execution-time estimator,
SLO-aware iterative FT admission, offline profiler, and live trace
collection — ported from `DeltaServe-vLLM/dserve-vllm/vllm/deltaserve/`.

## Desired behavior (the target this plan is built to hit)

**Finetuning throughput should be inversely coupled to inference load.** When no
inference is arriving, the GPU should be ~100% finetuning — FT admission runs
wide open and FT throughput is high. When inference spikes, the scheduler should
throttle FT admission so inference TTFT/TBT stay within SLO — FT throughput
drops, ceding the GPU back to inference. The whole point of the
estimator + admission gate + burst throttles in this plan is to make FT
throughput dynamically fill the *troughs* of the inference workload without ever
degrading inference during the *peaks*.

![Desired co-serving behavior — FT throughput fills inference troughs, backs off on spikes](compare_temporal_both_5090.pdf)

The plot above (the DeltaServe-vLLM reference on the bursty `nutanix` timeline,
RTX 5090) is the behavior we want to reproduce in sglang:
- **Top** — the inference workload: bursty request rate (gray) + output
  tokens/s (green), with clear troughs around t≈80–110 s and t≈200 s.
- **Bottom** — E2E latency (scatter) + **FT throughput** (shaded, right axis).
  The temporally-throttled variant (**DeltaServe-vLLM-Temp**) holds E2E latency
  to **+1.9 %** over the inference-only baseline (vs **+143.9 %** for naive
  co-serving) while delivering **1008 tok/s of FT — +98.9 %** over the
  un-throttled **507 tok/s**. It achieves this by ramping FT *up* in the
  inference troughs and backing *off* during the bursts — the shaded FT band
  rises exactly where the top-panel inference load falls.

The success criterion for this port: an sglang A/B run that shows the same
shape — FT throughput anti-correlated with inference load, near-baseline
inference latency under bursts, and high FT throughput when inference is idle.

---

Reference files (read these first, all under `DeltaServe-vLLM/`):
- `dserve-vllm/vllm/deltaserve/estimator.py` — the analytic model.
- `dserve-vllm/vllm/deltaserve/ft_scheduler.py` — admission gate + live trace recording + wiring.
- `dserve-vllm/vllm/deltaserve/coordinator.py` — admission state + budget + timing inbox.
- `dserve-vllm/vllm/deltaserve/profiling_batch_generator.py` — offline shape sweep.
- `dserve-vllm/vllm/v1/engine/core.py:574 profile_execution_model` — the launch profiler driver.
- `dserve-vllm/vllm/v1/worker/gpu_model_runner.py:4430` — the runner's deferred CUDA-event timing ring.
- `SLO_ESTIMATOR_REDESIGN.md` — the design rationale for the three-regime model.

---

## 0. What we're porting, and the gap today

### 0.1 The four components in the vLLM reference

1. **Analytic model** (`MergedExecutionEstimator`). One linear step-time formula,
   fit per *composition regime* by least squares:

   ```
   T_step ≈ α·S + β·T_in + γ·T_ft + δ·B_d + ε·K + c
   ```

   | Symbol | Meaning |
   |---|---|
   | `T_in` | total prefill tokens this step (**FT tokens are a subset**) |
   | `T_ft` | FT prefill tokens this step (⊆ `T_in`) |
   | `S` | `Σ nᵢ²` over prefill samples (self-attention work); exact from `prefill_lens`, else `T_in²/P` proxy |
   | `B_d` | number of decode requests |
   | `K` | total decode context (KV) tokens |
   | `c` | constant per-step overhead |

   **Three regimes**, partitioned by batch composition (each maps 1:1 to a
   CUDA-graph runtime mode):
   - `INF_PREFILL` (`T_ft==0, T_in>0`) → reduced matrix `[S,T_in,B_d,K,1]` (no γ).
   - `EAGER` (`T_ft>0`, FT forces eager) → full `[S,T_in,T_ft,B_d,K,1]`.
   - `DECODE_ONLY` (`T_in==0,B_d>0`) → `[B_d,K,1]` (no prefill terms).

   Per-regime RMSE drives a pessimistic `pred ×= 1 + 1.5·RMSE` safety margin.
   Refit every `REFIT_EVERY=256` steps over full history; `MIN_FIT_SAMPLES=8`
   per regime; cold-start falls back to any fitted regime.

2. **SLO-aware FT admission** (`FinetuneScheduler.admit_ft_to_step`). A 5-stage
   iterative greedy loop (no closed-form solve):
   - **Stage 0** preconditions via `coord.next_ft_budget()>0` (FT started, no
     backward in flight, admission open, buffer has room) + store has work.
   - **Stage 1** extract upcoming-step features *before* FT (`_current_step_features`).
   - **Stage 2** phase gate (`coserving_admission_phase: prefill` denies FT on
     decode-only steps; `both` lets the estimator decide).
   - **Stage 3** baseline-without-FT prediction in the matching regime
     (`DECODE_ONLY`/`INF_PREFILL`) + TTFT/TBT headroom check.
   - **Stage 4** iterative greedy: pop samples smallest-first; for each, build a
     hypothetical with the sample added and `predict(..., regime=EAGER)` (FT
     forces eager); admit while TTFT & max-TBT SLOs hold. Optional shapers
     (leaky-bucket `match_prefill_workload_factor`, proportional
     `ft_tokens_admission_constrain_factor`) wrap as outer pre-filters.
   - **Stage 5** commit: `store.claim(admitted)`, build `Request`s, `coord.reserve`.

3. **Offline profiler** (`ProfilingShapeGenerator` + `profile_execution_model`).
   At launch, before serving, drive synthetic batches through the *live*
   scheduler to seed the estimator: a shape sweep (`_prefill/_decode/_coserve/
   _mixed` + EAGER-coverage `_ft_on_decode/_ft_on_mixed/_ft_only_idle`),
   `warmup` (record off) then `recorded × profile_num_repeats`, ending in one
   `data_fit`. The sweep is built to break collinearities (prefill
   decomposition variants for α-vs-β; decode `B_d × K` grid for δ,ε; co-serve
   grid for γ).

4. **Live trace collection** (`StepExecutionTracker` + runner CUDA-event ring).
   - Features for the realized batch are stamped at `schedule()` time
     (`_features_from_output`) and frozen onto the `SchedulerOutput`
     (`_ft_step_features`, `_ft_step_predicted`, `_ft_step_was_graph`).
   - The runner times each step with a **ring of 4 CUDA-event pairs**, reading
     a slot's elapsed time 4 steps later (already complete → no hot-path sync,
     preserves async pipelining), and pushes `(features, duration, was_graph,
     predicted)` to `coord.push_sample`.
   - `schedule()` drains `coord.drain_completed_samples()` into the tracker and
     refits every 256 steps. Optional predicted-vs-actual CSV
     (`batch_prediction_stats_path`, `validate_estimator`).

### 0.2 What the sglang port has today (the gap)

| Component | vLLM reference | sglang port today |
|---|---|---|
| Analytic model | 3-regime lstsq `MergedExecutionEstimator` | `StepTimeEstimator` = per-kind **rolling mean**, never fed |
| Admission | iterative SLO loop `admit_ft_to_step` | `coord.can_admit()` → `return True`; `reserve()` budget gate runs but **attaches no FT tokens** |
| Profiler | `profile_execution_model` launch pass | none |
| Trace collection | CUDA-event ring → tracker → refit | `record_step` defined but **never called**; estimator buffers always empty |
| Actual FT throttle | the estimator | binary start flag + static random `SGLANG_DS_FT_ADMIT_RATE` + prefill-only-fire + drop-if-busy |

The scaffolding (`finetune_coordinator.py`, `step_time_estimator.py`,
`finetune_scheduler_mixin.py`, `ft_injector.py`, `finetuning_store*.py`,
`gates.py`) exists but is inert. This plan fills it in.

---

## 1. Architecture mapping: vLLM V1 → sglang v0.4.6

The port is mostly mechanical *except* for five structural differences that
shape the design. Read these before touching code.

### 1.1 Batch model — sglang is actually simpler

vLLM V1 runs **one flattened token-budget batch per step** mixing chunked
prefill + decode, so regime classification had to be composition-derived from
a single batch. sglang's `get_next_batch_to_run()` (`scheduler.py:1322`)
returns **either** a prefill batch (`get_new_batch_prefill` → `EXTEND`/`MIXED`)
**or** the running decode batch (`update_running_batch` → `DECODE`). The
`ForwardMode` enum (`forward_batch_info.py:53`) gives us the regime almost for
free:

| sglang `forward_mode` | + FT tokens? | Regime |
|---|---|---|
| `EXTEND` | no | `INF_PREFILL` |
| `EXTEND`/`MIXED` | yes | `EAGER` |
| `MIXED` (chunked prefill + running decode) | no | `INF_PREFILL` (`B_d>0,T_in>0` — formula covers it) |
| `DECODE` | n/a (FT never decodes) | `DECODE_ONLY` |
| `IDLE` | yes (FT-only) | `EAGER` (`T_in=t_ft`, `B_d=0`) |

**Keep the composition-derived selector** (`features.regime()`) rather than
keying on `forward_mode` — it's the single source of truth and already handles
the degenerate `B_d==0`/`K==0` cases. `forward_mode` becomes the observability
cross-check (the analogue of vLLM's vestigial `was_graph`).

### 1.2 Feature extraction maps to concrete sglang batch fields

`ScheduleBatch` / `ModelWorkerBatch` (`schedule_batch.py`) already carry
everything `StepFeatures` needs:

| `StepFeatures` | sglang source |
|---|---|
| `t_in` | `batch.extend_num_tokens` (EXTEND/MIXED) |
| `prefill_lens` / `S` | `batch.extend_lens` (per-request prefill lengths) |
| `t_ft` | `Σ extend_lens[i]` for `i` where `is_finetuning_flags[i]` (already populated — `schedule_batch.py:1668`) |
| `b_d` | count of decode reqs (`DECODE` batch size, or running-decode subset of `MIXED`) |
| `k` | `Σ seq_lens` over decode reqs (`batch.seq_lens` / `seq_lens_sum`) |

`is_finetuning_flags` and `is_finetuning_mask` already exist on the batch and
forward_batch — reuse them; no new plumbing for the FT subset.

### 1.3 FT source — adopt the vLLM self-inject model (key decision)

Today FT samples arrive as **external tagged requests** from the harness, and
the only throttle is the random admit-rate. The vLLM reference **self-injects**
from a `FinetuningStore` corpus *inside the scheduler under the SLO gate*. The
port already has `finetuning_store.py` + `ft_injector.py` stubs anticipating
this.

**Recommendation: adopt the self-inject model.** The admission gate is the
mechanism that decides *how many* FT tokens ride each step — that only works if
the scheduler controls the FT supply. Under the new design:
- The harness POSTs `/start_finetuning` (already wired in `gates.py` +
  `http_server.py`) and stops sending FT-tagged generate requests.
- `FinetuneSchedulerMixin` calls `admit_ft_to_step()` each step, pulls admitted
  samples from the store via the injector, and adds them to the waiting queue
  as `is_finetuning=True`, `max_new_tokens=1`, FT-LoRA-routed `Req`s — so
  `get_new_batch_prefill` batches them with inference exactly like the current
  externally-tagged reqs (the `model_runner.forward` FT path at
  `model_runner.py:1215` is unchanged and remains the execution mechanism).

This is the largest behavioral change. If keeping external FT requests is a
hard requirement, the fallback is a *gated pending queue* (Stage 4 decides
which already-arrived FT reqs to release into the batch this step) — same
estimator/admission core, different supply. The plan below assumes self-inject;
Phase 5 notes the pending-queue variant.

### 1.4 Step timing — the CUDA-event ring still applies

sglang's `event_loop_overlap` (`scheduler.py:672`) pipelines like vLLM async
(`schedule(N+1)` overlaps `forward(N)`), so a hot-path `cuda.synchronize()`
would serialize it. Port the **ring of CUDA-event pairs** read N steps later.
`event_loop_normal` is synchronous and a simpler timing would suffice, but the
ring works for both — implement once, gate on `coord is not None`. Wrap
`run_batch` (`scheduler.py:1569`).

### 1.5 No vLLM `CUDAGraphMode` dispatcher

vLLM's `_will_use_graph` queries `CudagraphDispatcher`. sglang's graph
eligibility lives in `cuda_graph_runner.py` and co-serve already forces eager
(`_forward_raw`). For the regime model we **don't need a graph predicate** —
the regime is composition-derived. Stamp `forward_mode` as the observability
field and skip the dispatcher port entirely.

### 1.6 Backward execution mode — async subprocess only (scope)

The sglang port has **two** backward execution paths
([`model_runner.py:1249-1255`](sglang-port/v046-port/sglang-fork/sglang/srt/model_executor/model_runner.py:1249)):

1. **In-process, synchronous** — `run_faux_backward(snapshot, self.model)` (and
   `SGLANG_DS_REAL_BACKWARD=1` without a subprocess) runs the backward on the
   scheduler/runner thread, blocking until it returns.
2. **Async subprocess** — `_ds_bwd_client.submit(snapshot)` ships to the
   MPS-capped child ([`backward_client.py`](sglang-port/v046-port/sglang-fork/sglang/srt/deltaserve/backward_client.py),
   `SGLANG_DS_BACKWARD_SUBPROCESS=1`).

**This plan targets the async subprocess path only.** The whole admission
lifecycle — reserve activation-buffer rows at admit, close admission when the
buffer fills, let the backward drain it concurrently, reopen on completion —
presupposes that the backward runs *concurrently* and signals completion
*asynchronously*. The in-process synchronous backward has no "pending" window
(it blocks, consumes the snapshot inline, and returns before the next schedule),
so the `pending_backward` / `poll_backward` / reopen-admission machinery is
both unnecessary and inapplicable there. **Out of scope:** SLO-aware admission
on top of the in-process backward. If `enable_finetuning` is on but the
subprocess is off, the admission gate should fall back to the existing
synchronous behavior (or refuse to arm — implementer's choice). Make the SLO
admission gate require `_ds_bwd_client is not None`.

---

## 2. Phased implementation plan

Each phase ends independently testable; don't start a phase until the prior
phase's acceptance test passes. All new code under
`sglang/srt/deltaserve/` or `sglang/srt/managers/`. Default-off / no-FT paths
must stay byte-identical (the existing port invariant).

### Phase A — Analytic model (`MergedExecutionEstimator`)

**Goal:** the three-regime estimator exists and fits synthetic data correctly,
in isolation (no scheduler wiring yet).

**Files:**
- New `sglang/srt/managers/step_time_estimator.py` (replace the rolling-mean
  stub) — port `StepFeatures`, `StepParams`, `StepExecutionTracker`,
  `MergedExecutionEstimator` from `estimator.py` ~verbatim. Drop the
  `vllm.deltaserve.dprint` import (use sglang's logger). Keep
  `REGIME_*`, `MIN_FIT_SAMPLES=8`, `REFIT_EVERY=256`, the `1+1.5·RMSE` margin,
  per-regime reduced design matrices, cold-start fallback.
- Keep the old `record_step(kind, ms)` shim on the estimator *temporarily* so
  `faux_backward`'s existing telemetry doesn't break, or migrate those callers.

**Acceptance:** unit test (`test_merged_estimator.py`, port from
`DeltaServe-vLLM/tests/test_merged_estimator.py`): `_select` routes each
composition shape to the right regime; `data_fit` partitions a synthetic
tracker; fitting `y = known α·S+…+c` recovers coefficients to ~1e-6; cold-start
fallback works with only one regime fitted; pessimistic margin applied.

**LoC:** ~400 (mostly verbatim) + tests.

### Phase B — Coordinator budget + timing inbox

**Goal:** `FinetuneCoordinator` exposes the real admission-budget surface and a
push/drain timing inbox the runner and scheduler use.

**Files:** extend `sglang/srt/managers/finetune_coordinator.py`:
- Port from `coordinator.py`: `space_remaining()`, `next_ft_budget()`
  (`ft_started AND admission_open AND not pending_backward → min(per_step_budget,
  space_remaining())`), `reserve(n, samples)` → write offset, `release_reserve`,
  `note_injection(next_sample_len, admitted_now)`, `poll_backward()`,
  `snapshot_admission`/`restore_admission`, the `_completed_samples` inbox
  (`push_sample` / `drain_completed_samples`), `record_timing` gate,
  `fill_count` / `reserved_fill` / `admission_open` / `pending_backward` /
  `epoch_flush_pending` state.
- **Delete** the stub `can_admit()` (the SLO decision moves to
  `admit_ft_to_step`) and the unused rolling-mean `step_time_estimator`
  member — the estimator now lives on the scheduler mixin.

**Admission lifecycle — buffer-full gate + backward-completion handshake
(load-bearing; this is the piece the current port is missing).**

The scheduler must *track how many FT tokens it has admitted into the
activation buffer* and stop admitting when it's full — and the backward
subprocess must tell it when the buffer has drained so it can reopen. The
current port does neither (the coordinator's `fill_count`/`reserved_fill` are
stubs nothing feeds, and `BackwardClient` never touches admission state). Port
the vLLM coordinator's lifecycle and wire it to the existing sglang IPC:

1. **Occupancy counter + admission flag (close when full).** Maintain
   `fill_count` (committed FT rows) + `reserved_fill` (admitted-but-not-yet-saved
   rows) so `space_remaining() = capacity − fill_count − reserved_fill`. On every
   admit, `reserve(n)` adds `n` to `reserved_fill`; `note_injection(next_sample_len,
   admitted_now)` then **peeks the smallest remaining sample** and, if it can no
   longer fit the free space (or the epoch drained), sets `admission_open = False`
   and raises `epoch_flush_pending` so the partial buffer gets trained instead of
   wedging. `next_ft_budget()` returns 0 whenever `admission_open` is False / a
   backward is pending / `space_remaining() <= 0`, so `admit_ft_to_step` (Phase D)
   stops admitting automatically. **This is the "FT admission flag" requirement:**
   admission closes the instant the buffer can't grow.
2. **Dispatch closes admission.** When the accumulated buffer is flushed to the
   backward (`_ds_bwd_client.submit(...)` succeeds), set `pending_backward = True`
   and `admission_open = False` — no new FT is admitted while a backward owns the
   buffer.
3. **Completion handshake reopens admission.** The MPS child already sends a
   completion reply per backward — `BackwardClient._drain()`
   ([`backward_client.py:45`](sglang-port/v046-port/sglang-fork/sglang/srt/deltaserve/backward_client.py:45))
   receives it today but only decrements `_outstanding` + imports masters; it
   **does not inform the scheduler**, and it only runs when the *next* sample is
   submitted. Fix both:
   - Add `BackwardClient.poll() -> bool` that calls `_drain()` and returns
     whether a completion was observed this call (or expose the reply payload).
   - The FT mixin calls `coord.poll_backward()` **every scheduler step** (top of
     the event-loop iteration, the `schedule()` analogue). `poll_backward()`
     calls `client.poll()`; on a completion it resets `fill_count = 0`
     (and `reserved_fill` for the committed batch), clears `pending_backward`,
     sets `admission_open = True`, and fires
     `on_backward_done(samples) → store.commit_claimed(samples)`.
   - This is the only reliable reopen path — without a per-step poll, admission
     stays closed until coincidental new traffic, stalling FT after the first
     flush. (`has_requests()`-style stepping must also keep the engine alive
     while `pending_backward` so the poll actually runs during inference idle —
     port the vLLM `has_requests` FT clause.)
4. **Backpressure already exists** (`_MAX_INFLIGHT=1`, drop-if-busy). Keep it as
   the safety net, but the admission flag should make drops rare: admission is
   closed *before* a second backward would be needed, so `submit` shouldn't be
   called while one is in flight.

**Acceptance:** unit test — `next_ft_budget` honors each gate; `reserve` /
`release_reserve` keep `space_remaining` consistent; `note_injection` closes
`admission_open` when the next sample won't fit; a simulated completion reply
through `poll_backward` resets occupancy + reopens admission + commits the
claimed samples; `push_sample` / `drain_completed_samples` round-trip;
`snapshot`/`restore` is exact. Integration: with the subprocess on, drive FT
to buffer-full and confirm admission closes, the backward fires once, and
admission reopens within one step of the child's completion reply (no stall).

**LoC:** ~300 (the lifecycle + the `BackwardClient.poll` ack add ~50 over the
plain budget surface).

### Phase C — Live trace collection (timing ring + tracker drain + refit)

**Goal:** every served step is GPU-timed and recorded; the estimator refits
online. Verifiable *before* admission is wired (estimator trains passively while
the static throttle still governs FT).

**Files:**
- `sglang/srt/managers/scheduler.py` `run_batch` (1569) / event loops: add the
  4-slot CUDA-event ring (port `gpu_model_runner.py:4430-4468`). On each step:
  record start/end events into the current ring slot; drain the slot's previous
  occupant (4 steps old → complete via `_end_evt.query()`) and
  `coord.push_sample(prev_feats, elapsed_s, forward_mode, prev_predicted)`.
  Gate on `coord.record_timing`.
- `FinetuneSchedulerMixin`: add `_features_from_realized_batch(batch)` (the
  `_features_from_output` analogue — extract `StepFeatures` from the
  sglang batch per §1.2; reconstruct pre-step `num_computed_tokens` so
  single-step prefills classify as prefill, not decode — this bug bit the vLLM
  port, see `ft_scheduler.py:427-436`). Stamp `batch._ft_step_features`,
  `_ft_step_predicted`, `_ft_step_forward_mode` right after the batch is built.
- In the event loop (top of each iteration, the analogue of vLLM `schedule()`):
  drain `coord.drain_completed_samples()` into `self._tracker`; call
  `if self._tracker.check_refit(): self._estimator.data_fit(self._tracker)`.
- Construct `self._estimator = MergedExecutionEstimator()` and
  `self._tracker = StepExecutionTracker()` in the mixin/`__init__`. Read SLO
  targets + `batch_prediction_stats_path` from `FinetuneConfig`.

**Acceptance:** run a co-serving benchmark (`auto_benchmark_sglang.py --co`);
confirm (a) the tracker accumulates samples across all three regimes, (b) a
refit fires at 256 steps and logs per-regime RMSE, (c) the predicted-vs-actual
CSV shows the regimes populated. No admission behavior change yet.

**LoC:** ~300.

### Phase D — SLO-aware iterative FT admission (`admit_ft_to_step`)

**Goal:** replace the static random admit-rate with the iterative SLO loop.

**Files:**
- `FinetuneSchedulerMixin`: add `_current_step_features()` (decode part from the
  running batch, prefill part from the waiting-queue head within
  `max_num_scheduled_tokens` — sglang equivalent is the prefill token budget in
  `get_new_batch_prefill`) and `admit_ft_to_step()` (port `ft_scheduler.py:190-402`
  ~verbatim: 5 stages, baseline regime prediction, per-sample EAGER prediction,
  TTFT/TBT headroom). Read `ttft_slo` / `max_tbt_slo` / `avg_tbt_slo` and
  `coserving_admission_phase` from config.
- Hook into `get_next_batch_to_run`: after the base picks the inference batch,
  call `admit_ft_to_step()`; for each admitted sample, build a `Req` via
  `ft_injector._make_request(sample)` and enqueue into `waiting_queue` *before*
  `get_new_batch_prefill` runs so FT rides the prefill batch (per §1.3). On a
  decode-only or idle step with `phase=="both"`, inject an FT-only `EXTEND`
  batch.
- Retire the `SGLANG_DS_FT_ADMIT_RATE` / external-tag path in `scheduler.py:957`
  (or keep it behind a `legacy_ft_admission` flag for A/B).
- Wire `ft_injector.py` + `finetuning_store.py` from stubs to real: `store.load`
  the corpus, `claim`/`commit_claimed`/`release_claimed`, `pop_next(exclude=)`
  smallest-first, `advance_epoch`. The vLLM `finetuning_store.py` is a clean
  port target (drop the oversized-sample-at-load deadlock, already fixed there).

**Acceptance:** co-serving benchmark with the estimator warm (seed via Phase E
or a long warmup). Confirm (a) FT admission backs off during inference bursts
(TTFT-critical steps admit 0 FT), (b) TTFT/TBT SLO satisfaction stays near
target, (c) FT throughput is non-trivial during quiet windows. A/B against the
old static-rate path.

**LoC:** ~500 + store/injector ~250.

### Phase E — Offline profiler (launch seeding)

**Goal:** the estimator is fitted *before* serving, so admission is SLO-aware
from request 1 (not after a 256-step cold start).

**Files:**
- New `sglang/srt/deltaserve/profiling_batch_generator.py` — port
  `ProfilingShapeGenerator` ~verbatim (it emits plain int lists, no engine
  objects; the EAGER-coverage shapes `_ft_on_decode/_ft_on_mixed/_ft_only_idle`
  are already in the reference).
- New profiler driver — the `profile_execution_model` analogue. sglang has no
  `EngineCore.profile_execution_model`; the natural host is the `Scheduler` (or
  a one-shot routine called from `Scheduler.__init__` after model load, before
  `event_loop_*`). It must: set `_profiling_mode=True` + `coord.profiling=True`,
  detach the backward, build synthetic `Req`s (`new_profiling_request`
  analogue), drive them through `get_next_batch_to_run` + `run_batch` directly
  (bypassing the recv loop), run `warmup` with `record_timing=False` then
  `recorded × profile_num_repeats` with timing on, drain into the tracker, and
  `data_fit`. `reset_coord()` between shapes; `purge_profiling_requests` after.
- Gate on `finetune.profile_on_launch` + `finetune.profile_num_repeats`.

**sglang-specific risk:** driving the scheduler/runner *outside* the event loop
with synthetic requests is the heaviest part of the port — sglang's
`run_batch`/`process_batch_result` assume event-loop context (tp_worker,
result handling). Budget time to either (a) reuse the existing sglang warmup
path's batch-construction, or (b) drive at the `tp_worker.forward_batch_*` level
and read the timing ring directly. Validate the driver on a tiny model first.

**Acceptance:** launch with `profile_on_launch=true`; confirm the profiler runs
the full sweep, logs per-regime RMSE at the end, and `_estimator.is_ready` is
True before the first real request. Admission is SLO-aware from step 1.

**LoC:** ~250 (generator, verbatim) + ~300 (driver).

### Phase F — Config field parity + eval + polish

**Goal:** `FinetuneConfig` reaches field parity with the vLLM reference so every
SLO/admission knob is settable; eval tooling exposes the new levers.

**Files:**
- `sglang/srt/configs/finetune.py` (`FinetuneConfig`): the sglang dataclass
  already has most fields but is **missing** the ones the new admission/estimator
  paths need — add, mirroring `DeltaServe-vLLM/dserve-vllm/vllm/config/finetune.py`:
  `coserving_admission_phase` (`prefill`/`both`), `decode_only_ft_safety_margin`,
  `match_prefill_workload_factor`, `validate_estimator`,
  `estimator_validation_path`, `save_attn_qkv`, `save_attn_ctx`,
  `fwd_token_throttle_enable` / `fwd_token_throttle`, the `rps_throttle_*` group,
  and `print_step_mode`. (`ttft_slo`, `avg_tbt_slo`, `max_tbt_slo`,
  `ft_tokens_admission_constrain_factor`, `profile_on_launch`,
  `profile_num_repeats`, `start_on_launch`, `batch_prediction_stats_path`,
  `bwd_log_path`, `forward_interruptible`, `ft_only_admission_grace_ms`,
  `max_saved_finetuning_tokens` already exist.) Field parity is a prerequisite
  for Phase G's YAML loader.
- `auto_benchmark_sglang.py`: add `--scheduler {prefill,both}` and a
  predicted-vs-actual export, mirroring the vLLM eval tooling.
- Drop the dead `faux_backward` `SGLANG_DS_BACKWARD_MIN_INTERVAL_MS` throttle
  and the static `SGLANG_DS_FT_ADMIT_RATE` path (superseded by the SLO gate) or
  document them as legacy A/B baselines.

**Acceptance:** end-to-end A/B (`prefill` vs `both`, estimator-on vs static-rate)
showing P99 TTFT improvement and/or FT-throughput uplift at equal SLO
compliance — the headline metric.

### Phase G — DeltaServe sectioned-YAML config support (`--finetune-config`)

**Goal:** drive the sglang co-serving server from a DeltaServe-style sectioned
YAML — e.g. `DeltaServe-vLLM/configs/serving_config_finetuning_llama3.yaml` —
instead of hand-assembling CLI flags, so the FT/SLO knob surface is configured
exactly as it is on the vLLM side.

**What the config file is.** A sectioned YAML the vLLM port loads via
[`config_loader.py`](../../DeltaServe-vLLM/dserve-vllm/vllm/deltaserve/config_loader.py):

| Section | Disposition |
|---|---|
| `finetune` | folded into `FinetuneConfig` kwargs |
| `slo` | folded into `FinetuneConfig` (SLO/admission knobs) |
| `debug` | folded into `FinetuneConfig` (observability flags) |
| `server` | passthrough extra (`host`/`port`/`rank_id`/`api_server_count`) |
| `adapters` | passthrough extra (inference LoRA paths) |
| `model` / `engine` / `parallel` / `lora` | merged → engine kwargs (one bag) |

Relative `*path*` values under `finetune`/`adapters` are resolved against the
YAML's own directory. Unknown finetune keys are rejected; unknown engine keys
raise from the engine-arg constructor.

**Current sglang state.** `server_args.py:1437` already declares a
`--finetune-config <path>` flag (`ServerArgs.finetune_config: Optional[str]`),
but **nothing reads it** — `scheduler.py:203` hard-codes
`FinetuneConfig(enable_finetuning=True)`, dropping every YAML knob. No YAML
loader exists in the fork.

**The vocabulary problem (the crux).** The `finetune`/`slo`/`debug` sections are
**vocab-identical** across vLLM and sglang (both are `FinetuneConfig` fields) →
port verbatim. But the engine-vocab sections are written in vLLM `EngineArgs`
names, which don't match sglang `ServerArgs`, and some invert:

| YAML key (vLLM vocab) | sglang `ServerArgs` | note |
|---|---|---|
| `model` | `model_path` | rename |
| `max_model_len` | `context_length` | rename |
| `gpu_memory_utilization` | `mem_fraction_static` | rename (semantics ≈ same) |
| `enforce_eager: false` | `disable_cuda_graph: false` | **rename + same polarity** (both "off" = graphs on) — but `enforce_eager:true` ⇒ `disable_cuda_graph:true` |
| `tensor_parallel_size` | `tp_size` | rename |
| `max_loras` | `max_loras_per_batch` | rename |
| `enable_lora` | *(inferred from `lora_paths`)* | no bool in sglang |
| `adapters.lora_path_N` | `lora_paths` (list) | restructure |
| `dtype`,`tokenizer_mode`,`trust_remote_code`,`host`,`port`,`max_lora_rank` | same | passthrough |
| `api_server_count`, `disable_log_stats` | — | vLLM-frontend-only; no sglang analogue (warn + ignore) |

**Files:**
- New `sglang/srt/deltaserve/config_loader.py` — port `config_loader.py`,
  retargeted from `EngineArgs`+vLLM-`FinetuneConfig` to sglang
  `ServerArgs`+sglang-`FinetuneConfig`:
  - `load_yaml_config(path)` — read + validate shape + resolve relative
    `*path*` values under `finetune`/`adapters` (verbatim from the reference).
  - `split_config(cfg)` — fold `finetune`+`slo`+`debug` → `FinetuneConfig(**body)`
    (sglang dataclass; reject unknown keys to match the reference's
    `extra=forbid`); merge `model`/`engine`/`parallel`/`lora` → one
    `server_args_kwargs` bag; return `server` + `adapters` as `extras`.
  - **Engine-key translation:** a small alias map (the table above) applied to
    the merged engine bag before it hits `ServerArgs`, plus the
    `enforce_eager → disable_cuda_graph` and `adapters.lora_path_* → lora_paths`
    special cases, and a warn-and-drop list for vLLM-frontend-only keys
    (`api_server_count`, `disable_log_stats`). After translation, any remaining
    key must be a real `ServerArgs` field name (passthrough), preserving the
    loader's "unknown engine section = bag of kwargs" philosophy.
- Wire `--finetune-config` in the launch path (`server_args.py` post-init or
  `launch_server`): when set, `load_yaml_config` → `split_config`; apply the
  engine kwargs onto `ServerArgs` (YAML provides defaults, explicit CLI flags
  win), stash the built `FinetuneConfig` so the scheduler uses it instead of the
  hard-coded one, and set `enable_finetuning` from `finetune.enable_finetuning`.
- Thread the YAML-built `FinetuneConfig` to the Scheduler: replace
  `scheduler.py:203 FinetuneConfig(enable_finetuning=True)` with the
  loaded config (serialize it onto `ServerArgs`/`PortArgs` so it survives the
  spawn to the scheduler subprocess, the same way `server_args` already crosses
  that boundary).
- Ship a sglang-vocab config: `sglang-port/v046-port/configs/serving_config_finetuning_llama3.yaml`
  — the converted twin of the vLLM file (engine sections in sglang vocab,
  `finetune`/`slo`/`debug` copy-pasted), plus a `_both.yaml` variant for the
  unified-phase scheduler. Keep the alias map so the *original* vLLM file also
  loads (with a one-time "translated N legacy keys" log) for cross-checking.

**Decision to confirm:** support the literal vLLM YAML via the alias map
(convenience, mild brittleness) **and** ship a native sglang-vocab config
(recommended canonical form), or only the latter. The plan above does both; drop
the alias map if exact-file compatibility isn't needed.

**Acceptance:**
- `load_yaml_config` + `split_config` unit test: the reference
  `serving_config_finetuning_llama3.yaml` produces a `FinetuneConfig` with the
  expected SLO/finetune fields and a `ServerArgs` kwargs bag with translated
  engine keys; relative `data_path`/`finetuning_lora_path` resolve absolute;
  unknown finetune key raises; `api_server_count` warns-and-drops.
- End-to-end: `--finetune-config configs/serving_config_finetuning_llama3.yaml`
  launches the co-serving server with the YAML's SLO knobs in effect (verify via
  the startup config dump + that `FinetuneConfig.ttft_slo` etc. reach the
  scheduler), no CLI flag soup.

**LoC:** ~200 (loader, mostly verbatim + the alias table) + the converted config.

**Sequencing:** depends only on Phase F (field parity). Land it right after F —
once the YAML loader exists, every later A/B run (D's `prefill`-vs-`both`, E's
`profile_on_launch`) is configured by editing the YAML instead of threading new
CLI flags. (If you want YAML-driven config *before* the SLO work, F+G can land
first as a standalone increment — they don't depend on A–E.)

### Phase H — Burst back-off throttles (RPS admission gate + fwd-token backward pause)

**Goal:** port the two *coarse, reactive* burst-protection knobs from
`configs/serving_config_finetuning_llama3_A100.yaml`. They are **complementary
to the SLO predictor**, not part of it: the estimator (Phase D) decides
admission from *predicted step time*; these two react to cheaper signals
(*inference arrival rate* and *raw batch size*) that move faster than a 256-step
refit and need no model. Both default OFF; the YAML enables them for the A100
profile. Together they "back off FT in all directions during bursts."

#### H.1 `rps_throttle_*` — closes FT **admission** under arrival bursts

**What it does** ([`coordinator.py:251 check_rps_throttle`](../../DeltaServe-vLLM/dserve-vllm/vllm/deltaserve/coordinator.py:251)):
a sliding-window inference-arrival-rate gate over `coord.admission_open`.
- A `RpsTracker` (deque of arrival timestamps, `coordinator.py:35`) yields
  `rps(now) = arrivals_in_last_window / window_s`.
- Each scheduler step calls `check_rps_throttle(now, close_rps, open_rps,
  window_s, close_time)`:
  - **Engage** (`admission_open=False`) when `rps > close_rps`.
  - **Release** (`admission_open=True`) when `rps < open_rps`.
- **Spatial hysteresis:** the band between `close_rps` (A100: 20) and `open_rps`
  (19) — RPS *inside* the band leaves the gate as-is, so it can't flap at the
  boundary. (`open_rps` must be `< close_rps`.)
- **Temporal hysteresis:** `close_time` (A100: 0.5s) — once engaged, release is
  deferred until `close_time` has elapsed since engage, *unless* `rps == 0`
  (idle-bypass releases immediately, since there's no inference to protect).
- **"Option B"**: it overloads the existing `admission_open` flag (which
  `next_ft_budget` already ANDs on) rather than adding a new gate — so no change
  to the admission-budget surface. It interleaves with the buffer-full /
  backward-start / backward-done writers of `admission_open` under the
  single-threaded scheduler.

This is a *faster-reacting, model-free* complement to Phase D: it can slam FT
admission shut at the very start of a burst, before the estimator's per-step
prediction (or the next refit) catches up.

**sglang integration:**
- Port `RpsTracker` + `check_rps_throttle` onto `FinetuneCoordinator` (Phase B
  host), plus the `rps_throttle_active` / `rps_throttle_engaged_at` latches.
- **Arrival hook:** call `coord.rps_tracker.note_arrival(monotonic())` once per
  inference request as it enters the scheduler. In sglang that's
  `Scheduler.recv_requests` / `handle_generate_request`
  (`scheduler.py` intake) — the analogue of vLLM's input-queue drain
  (`core.py:1734`). Tag *inference* arrivals only (skip FT/profiling reqs).
- **Per-step call:** in the FT mixin's step (the `schedule()` analogue, top of
  the event-loop iteration), call `coord.check_rps_throttle(...)` with the
  config values when `rps_throttle_enable`. `next_ft_budget()` already returns 0
  while `admission_open` is False, so `admit_ft_to_step` (Phase D) backs off
  automatically — no extra wiring in the admission loop.

#### H.2 `fwd_token_throttle_*` — pauses the running **backward** on big batches

**What it does** ([`gpu_model_runner.py:4479-4527`](../../DeltaServe-vLLM/dserve-vllm/vllm/v1/worker/gpu_model_runner.py:4479)):
extends the *backward-pause* (GPU-yield) condition. Today the backward is paused
for a step iff `pending_backward AND feats.t_in > 0` (prefill present —
TTFT-critical). With the throttle on, the pause also fires on large
**decode-heavy** batches:

```
pause = pending_backward AND (
    t_in > 0                                  # today's rule (prefill present)
    OR (fwd_token_throttle_enable AND total > fwd_token_throttle)   # new rule
)
where total = prefill + ft + decode_count   # the (total=N) field of the batch log
```

Resume is automatic the moment both conditions go False (a `_throttle_held`
latch + an idle-release safety net handle the case where the engine goes idle
while throttle-paused, so the backward isn't stuck paused). A100 config: fire
when a batch exceeds `200` total tokens. Where `rps_throttle` denies *new* FT
into the buffer, this yields more GPU from the *in-flight* backward whenever the
inference batch is heavy — even on pure-decode steps the prefill-only rule would
have let run concurrently.

**sglang integration:**
- The sglang port **already has the prefill-gated yield** — `run_batch` revokes
  the backward's `gpu_grant` when `batch.forward_mode.is_extend()`
  (`scheduler.py:1586-1594`). This is the exact site to extend: compute
  `total = extend_num_tokens + decode_count` for the batch and OR in
  `(fwd_token_throttle_enable and total > fwd_token_throttle)` alongside the
  existing `is_extend()` revoke. Port the `_throttle_held` latch + idle-release
  safety net so a throttle-paused backward resumes when load drops.
- Pure runner/scheduler-side; no estimator, no admission-state coupling. Reads
  the config every step so it can be toggled live.

**Acceptance (both):**
- `rps_throttle`: unit-test `check_rps_throttle` — engage above `close_rps`,
  stay engaged inside the hysteresis band, defer release for `close_time`,
  idle-bypass at rps 0. E2E: under a bursty timeline, confirm FT admission
  closes within ~`window_s` of a burst and reopens after it, and TTFT during
  bursts improves vs. throttle-off.
- `fwd_token_throttle`: E2E with a decode-heavy workload — confirm the backward
  yields (grant revoked) on `total > threshold` steps and resumes below it;
  per-cycle backward log shows reduced concurrency during big batches; TBT under
  load improves vs. throttle-off.

**LoC:** ~150 (RpsTracker + check_rps_throttle + two hooks) + ~40
(fwd-token extension of the existing grant-revoke).

**Sequencing:** independent of the SLO estimator — needs only Phase B
(coordinator `admission_open` + `next_ft_budget`) for H.1 and the existing
`gpu_grant` site for H.2. Can land alongside B/C, before D. They make the
*already-shipped* co-serving path burst-aware even without the predictor, so
they're a cheap early win.

---

## 3. Recommended sequencing & dependencies

```
A (model) ─┬─> C (trace collect, passive) ─┐
B (coord) ─┴─> H (burst throttles) ─────────┼─> D (admission) ─> F (config parity) ─> G (YAML loader)
                                  E (profiler) ──────────────┘
```

- **F + G** are landable as a standalone increment before A–E — they don't
  depend on the SLO work, only on `FinetuneConfig` field parity. Doing G early
  means every later phase is configured by YAML instead of new CLI flags.
- **H** (burst throttles) is independent of the SLO estimator — it needs only
  Phase B's coordinator (`admission_open`) and the existing `gpu_grant` site.
  It can land alongside B/C as an early, model-free burst-protection win.

- **A + B** are independent, pure-Python, fully unit-testable — do them first
  and in parallel.
- **C** depends on A + B; lands the estimator training passively under the
  *existing* static throttle — lowest-risk way to validate the model on real
  GPU data before changing any admission behavior.
- **D** depends on C (needs a warm estimator) — this is where behavior changes.
- **E** depends on A + B + the feature-extraction from C; it's the highest-risk
  piece (driving the scheduler off-loop) and can land after D (cold-start
  fallback in `admit_ft_to_step` already handles an unfitted estimator).
- **F** throughout.

Minimal viable SLO admission = A + B + C + D (cold-start via online refit only).
E removes the cold-start window. Recommend shipping A–D first, then E.

---

## 4. sglang-specific risks / gotchas

1. **Off-loop profiler driver (Phase E)** — the single biggest unknown. vLLM had
   a clean `EngineCore.profile_execution_model` seam; sglang doesn't. De-risk by
   landing A–D first (online refit works without it).
2. **`MIXED` forward mode** — sglang's chunked-prefill MIXED batch has both
   `T_in>0` and `B_d>0` with no FT → `INF_PREFILL` regime (the formula's
   `δ·B_d+ε·K` terms cover the decode part). Make sure `_features_from_realized_batch`
   splits extend vs decode rows correctly for MIXED (use `extend_lens` length vs
   total batch size). The vLLM `_features_from_output` pre-step
   `num_computed_tokens` reconstruction is the precedent — replicate it.
3. **Overlap scheduler pipelining** — the timing ring's "read 4 steps later"
   depth must exceed sglang's overlap depth (vLLM used RING=4 > batch-queue
   depth 2). Verify sglang's overlap depth and size the ring accordingly.
4. **`pending_backward` handshake (designed in Phase B, §B "Admission
   lifecycle").** sglang's backward is a best-effort `_ds_bwd_client.submit`
   that *drops if busy* and never informs the scheduler; vLLM's coordinator
   assumes a strict admit→close→drain→reopen cycle. The resolution: track
   buffer occupancy + close admission when full, and add a
   `BackwardClient.poll()` that surfaces the child's existing completion reply
   so `poll_backward()` (called every step) can reset occupancy and reopen
   admission. This is **load-bearing** — without it, admission closes on the
   first buffer-full and never reopens. Also note (§1.6) this whole cycle
   assumes the **async subprocess** backward; the in-process synchronous
   backward is out of scope.
5. **TP / multi-rank** — captures and admission are lock-step under TP in the
   original; the sglang mixin runs per-rank. If TP>1 is in scope, the timing
   ring and tracker must be rank-0-authoritative (or per-rank-consistent).
   Out of scope for single-GPU first cut.
6. **Self-inject vs external-tag (§1.3)** — confirm with the eval harness owner
   before retiring the external FT-request path; it changes how benchmarks drive
   FT (POST `/start_finetuning` + corpus instead of streaming FT requests).

---

## 5. What stays the same (don't pull these in)

- The accumulate→backward execution path (`accumulate.py`, `faux_backward.py`,
  `real_backward.py`, `model_runner.forward` FT dispatch) — admission decides
  *what rides the batch*; the backward mechanism is unchanged.
- The `gpu_grant` prefill-pause co-serving contract (`scheduler.py:1586`).
- The `gates.py` `/start_finetuning` flag (it becomes Stage-0's `ft_started`).
- The forced-eager-on-FT invariant (already enforced in `_forward_raw`).
- The MPS-capped backward subprocess + IPC.

---

## 6. Effort estimate

| Phase | Component | LoC (excl. tests) | Risk |
|---|---|---|---|
| A | Analytic model | ~400 (verbatim) | low |
| B | Coordinator surface + admission lifecycle (buffer-full gate + bwd-completion handshake) | ~300 | low |
| C | Trace collection | ~300 | medium (ring + feature extraction) |
| D | SLO admission + store/injector | ~750 | medium (behavioral change) |
| E | Offline profiler | ~550 | **high** (off-loop driver) |
| F | Config field parity + eval | ~200 | low |
| G | DeltaServe YAML config loader (`--finetune-config`) | ~200 + config | low |
| H | Burst throttles (RPS admission gate + fwd-token backward pause) | ~190 | low |

Total ≈ 2,890 LoC + tests. A–D (online-refit MVP) ≈ 1,750 LoC is the
critical path to a working SLO-aware gate; E is the cold-start optimization;
F + G are the config/usability layer; H is independent burst protection
(landable early). All of E/F/G/H are optional relative to the A–D core.
