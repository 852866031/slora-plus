# Installation — DeltaServe sglang port (v0.4.6.post5)

This port is **not a fork of sglang**. It is a thin overlay on the stock
`sglang==0.4.6.post5` PyPI package:

- **18 new drop-in files** — the `deltaserve/` package (14 files: co-serving
  runtime — activation capture, real LoRA backward, GPU-grant, gates, the
  optional backward subprocess) plus 4 files that live elsewhere under `srt/`.
- **10 patched files** — small hooks into stock sglang (request flag, per-token
  finetuning mask, scheduler admission, forward→backward dispatch, the
  `/start_finetuning` HTTP endpoints, two new server flags). Captured in
  `sglang-046-port.patch`.

**sglang itself ships in the repo** as a vendored wheel at
`vendor/sglang-0.4.6.post5-py3-none-any.whl` (pinned, so the build is
reproducible even if PyPI yanks that version). The only thing the installer
pulls from the network is the heavy GPU stack (torch / flashinfer), which is
arch-specific and not vendorable. So one script — `install.sh` — sets up
everything: vendored sglang + deps + the DeltaServe overlay.

---

## Quick start

```bash
cd sglang-port/v046-port
bash install.sh
```

The script will:

1. Install the **vendored** `vendor/sglang-0.4.6.post5-*.whl` if sglang isn't
   already present (pip still pulls torch + flashinfer — several minutes, needs
   CUDA). Falls back to PyPI only if the vendored wheel is missing. If a
   *different* sglang version is already installed it stops and asks you to pin
   the version first.
2. Copy the 18 drop-in files into the installed package.
3. Apply `sglang-046-port.patch` (`-p1` from the package root), backing up each
   original to `<file>.ds_orig` first.
4. Import-check every deltaserve module and confirm the two new server flags
   (`--enable-finetuning`, `--backward-mps-percentage`) exist.

It is **idempotent**: re-running refreshes the drop-ins and skips the patch if
the tree already shows the edits. To revert:

```bash
bash install.sh --uninstall   # restores *.ds_orig, removes drop-ins (leaves sglang itself)
```

---

## Requirements

- Linux + NVIDIA GPU with a recent CUDA driver. Co-serving was developed and
  benchmarked on **H200**; any Hopper/Ampere card with enough memory for your
  model works.
- Python 3.10–3.12 (the reference env is 3.12).
- `patch` available on `PATH` (standard on Linux).
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
pip install "vendor/sglang-0.4.6.post5-py3-none-any.whl[all]"   # or: sglang[all]==0.4.6.post5
SG=$(python -c 'import os,sglang;print(os.path.dirname(sglang.__file__))')

# 1. drop-in package
cp -r new-files/deltaserve "$SG/srt/deltaserve"

# 2. the 4 files that live elsewhere under srt/
cp new-files/finetune.py                 "$SG/srt/configs/"
cp new-files/finetune_coordinator.py     "$SG/srt/managers/"
cp new-files/finetune_scheduler_mixin.py "$SG/srt/managers/"
cp new-files/step_time_estimator.py      "$SG/srt/managers/"

# 3. patch the 10 stock files
( cd "$SG" && patch -p1 < /path/to/sglang-046-port.patch )
```

Patched files (all under `srt/`): `entrypoints/engine.py`,
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
