"""DeltaServe co-serving gates (Section 11 of CO_SERVING_OPTIMIZATIONS.md).

`finetuning_started` toggle: FT admission is gated off at launch when
`finetune.start_on_launch=False` (the default under co-serving) until the
eval harness POSTs /start_finetuning. Lets profiling + warmup run on a
clean GPU.

Implementation note: uses a multiprocessing-shared boolean stored in a
file in the runtime dir so the http_server (rank-0 main process) and the
Scheduler subprocess both see it. Cheap to poll: one fstat per request.
"""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

# Shared flag location. We use the filesystem so the toggle survives
# fork/spawn boundaries without requiring shared mp.Manager state.
_FLAG_DIR = Path(os.environ.get("SGLANG_DS_GATE_DIR", tempfile.gettempdir())) / "sglang_ds_gates"
_FLAG_DIR.mkdir(parents=True, exist_ok=True)
_FT_STARTED_PATH = _FLAG_DIR / "finetuning_started"

# Default ON when not explicitly gated — matches DeltaServe-vLLM
# `start_on_launch: True` baseline. Eval harness sets start_on_launch=False
# via the env var.
_DEFAULT_STARTED = os.environ.get("SGLANG_DS_FT_START_ON_LAUNCH", "1") != "0"
_lock = threading.Lock()


def set_finetuning_started(v: bool) -> None:
    """Idempotent. Writes the flag to the shared file."""
    with _lock:
        if v:
            _FT_STARTED_PATH.write_text("1")
        else:
            try:
                _FT_STARTED_PATH.unlink()
            except FileNotFoundError:
                pass


def is_finetuning_started() -> bool:
    """Cheap: one fstat. Default value when file absent depends on
    SGLANG_DS_FT_START_ON_LAUNCH env (1 = started by default, 0 = gated)."""
    try:
        return _FT_STARTED_PATH.exists()
    except OSError:
        return _DEFAULT_STARTED


# At module load, sync the flag file to the default. The default is True
# (start_on_launch — match DeltaServe-vLLM); set SGLANG_DS_FT_START_ON_LAUNCH=0
# to gate FT off at launch (eval harness opens via POST /start_finetuning).
#
# IMPORTANT: only run init ONCE per dir (across processes). Subsequent module
# loads in other processes (e.g. the scheduler subprocess) MUST NOT clobber
# the state set by the http_server. We use an `_init_done` marker file with
# O_EXCL atomic create to guarantee single-init.
_INIT_MARKER = _FLAG_DIR / "_init_done"

def _module_init():
    with _lock:
        # Atomic create-if-not-exists. Only the first importer succeeds.
        try:
            fd = os.open(str(_INIT_MARKER), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
        except FileExistsError:
            return  # already initialized by another process
        # First importer — set the flag according to the env default.
        if _DEFAULT_STARTED:
            try:
                _FT_STARTED_PATH.write_text("1")
            except OSError:
                pass
        else:
            try:
                _FT_STARTED_PATH.unlink()
            except FileNotFoundError:
                pass

_module_init()
