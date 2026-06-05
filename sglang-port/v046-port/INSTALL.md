# Installation — DeltaServe sglang port (v0.4.6.post5)

This port is a **fork of sglang `0.4.6.post5`**: the full sglang source tree,
with the DeltaServe co-serving changes already applied, lives in this repo at
[`sglang-fork/`](sglang-fork/). You install it directly — **no patch step**:

- **`sglang-fork/`** — the complete, browsable sglang source with our changes
  in place: the `deltaserve/` co-serving runtime (`sglang/srt/deltaserve/` —
  activation capture, real LoRA backward, MPS-subprocess backward, served-LoRA
  publish) plus 4 files under `srt/{configs,managers}/`, and the 10 stock files
  hooked for the co-serving path (request flag, per-token FT mask, scheduler
  admission, forward→backward dispatch, `/start_finetuning` endpoints, two new
  server flags).
- **`sglang-046-port.patch`** — kept as a concise **"what we changed vs stock
  sglang"** reference (the 10 modified files); not needed to install.

The only thing the installer pulls from the network is the heavy GPU stack
(torch / flashinfer / sgl-kernel), which is arch-specific and not vendorable.

---

## Quick start

```bash
cd sglang-port/v046-port
bash install.sh
```

The script just runs `pip install -e sglang-fork[all]` — an **editable install**
of the patched fork (pip resolves torch / flashinfer / sgl-kernel from PyPI;
several minutes, needs CUDA) — then import-checks the `deltaserve` modules and
confirms the two new server flags (`--enable-finetuning`,
`--backward-mps-percentage`). To revert:

```bash
bash install.sh --uninstall   # pip uninstall sglang
```

Because it's an editable install, any edit you make under `sglang-fork/` is live
immediately — no reinstall.

---

## Requirements

- Linux + NVIDIA GPU with a recent CUDA driver. Co-serving was developed and
  benchmarked on **H200**; any Hopper/Ampere card with enough memory for your
  model works.
- Python 3.10–3.12 (the reference env is 3.12).
- For the **subprocess backward + MPS isolation** path, the
  [CUDA MPS daemon](https://docs.nvidia.com/deploy/mps/) should be running so
  `--backward-mps-percentage` can carve out a GPU slice for the child:
  ```bash
  export CUDA_VISIBLE_DEVICES=0
  nvidia-cuda-mps-control -d
  ```
  The in-process backward path (`SGLANG_DS_REAL_BACKWARD=1`, no subprocess)
  works without MPS.

---

## Manual install (if you don't want the script)

```bash
pip install -e "sglang-fork[all]"
```

That's it — the fork's `pyproject.toml` carries the same dependencies as stock
sglang 0.4.6.post5. The DeltaServe changes touch these stock files under
`sglang/srt/` (see `sglang-046-port.patch` for the exact diff): `entrypoints/engine.py`,
`entrypoints/http_server.py`, `managers/io_struct.py`, `managers/scheduler.py`,
`managers/schedule_batch.py`, `managers/tokenizer_manager.py`,
`mem_cache/paged_allocator.py`, `model_executor/forward_batch_info.py`,
`model_executor/model_runner.py`, `server_args.py`.

---

## Launching a co-serving server

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.2-1B-Instruct \
    --tp-size 1 --mem-fraction-static 0.5 \
    --enable-finetuning --backward-mps-percentage 10
```

New flags added by the port:

| Flag | Default | Meaning |
|---|---|---|
| `--enable-finetuning` | off | turn on the co-serving backward path; spawns the backward subprocess |
| `--backward-mps-percentage N` | 10 | MPS thread % handed to the backward child (when MPS daemon is up) |

Runtime env toggles (read by `deltaserve/`):

| Env var | Default | Meaning |
|---|---|---|
| `SGLANG_DS_REAL_BACKWARD` | `0` | `1` = run the **real** LoRA backward; `0` = faux placeholder |
| `SGLANG_DS_FT_START_ON_LAUNCH` | `1` | `0` = wait for `POST /start_finetuning` before any backward fires |
| `SGLANG_DS_BACKWARD_MIN_INTERVAL_MS` | unset | SLO throttle: min ms between backward fires |
| `SGLANG_DS_FT_ADMIT_RATE` | unset | fraction of scheduler ticks allowed to admit FT tokens |

Mark a request as a finetuning sample by sending `"is_finetuning": true` in the
generate payload. HTTP control: `POST /start_finetuning`, `POST /stop_finetuning`,
`GET /finetuning_status`.

---

## Reproducing the benchmarks

```bash
# inference-only baseline + co-serving, tight timeline, real backward, 1B model
python auto_benchmark_sglang.py --co --tight --real-backward \
    --model meta-llama/Llama-3.2-1B-Instruct

# plots from the produced output/ CSVs
python auto_plot_sglang.py
```

`auto_benchmark_sglang.py` launches the server itself (see `build_server_cmd`),
sweeps the timeline, and writes `output/timeline_results*.csv` +
`output/bwd_log*.csv`. See `README.md` for the result tables and what each plot
shows.

---

## Troubleshooting

- **"patch does not apply cleanly"** — your installed sglang isn't exactly
  `0.4.6.post5`. Pin it: `pip install sglang==0.4.6.post5`, then re-run.
- **`add_lora ... doesn't contain tensors`** — your LoRA adapter dir has only
  `adapter_config.json` and no weights. Generate matching random weights with
  `scripts/gen_dummy_lora.py`.
- **Backward subprocess never fires** — confirm the MPS daemon is up, and that
  you POSTed `/start_finetuning` (or launched with
  `SGLANG_DS_FT_START_ON_LAUNCH=1`).
- **OOM during CUDA graph capture** — lower `--mem-fraction-static` (0.5 is the
  benchmark default for the 8B co-serving runs).
