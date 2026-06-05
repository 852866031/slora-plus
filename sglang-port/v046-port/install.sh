#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# DeltaServe sglang-port installer
#
# Installs sglang 0.4.6.post5 and applies the DeltaServe co-serving port:
#   1. pip install sglang==0.4.6.post5  (skipped if already present)
#   2. copy 18 new drop-in files into the installed sglang package
#   3. patch 10 existing sglang files with sglang-046-port.patch
#
# Idempotent: re-running re-copies drop-ins and re-checks the patch.
# Safe: backs up each patched file to <file>.ds_orig before patching, and
# refuses to double-apply (detects an already-patched tree).
#
# Usage:
#   bash install.sh                 # install into the active python env
#   bash install.sh --uninstall     # restore patched files, remove drop-ins
# ---------------------------------------------------------------------------
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/sglang-046-port.patch"
SGLANG_VERSION="0.4.6.post5"
VENDOR_WHEEL="$HERE/vendor/sglang-${SGLANG_VERSION}-py3-none-any.whl"

log() { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m[install] ERROR:\033[0m %s\n' "$*" >&2; }

# --- drop-in file -> destination (relative to <sglang>/srt) ----------------
# deltaserve/ package is copied wholesale; these 4 live elsewhere in srt/.
declare -A TOPLEVEL=(
  [finetune.py]=configs
  [finetune_coordinator.py]=managers
  [finetune_scheduler_mixin.py]=managers
  [step_time_estimator.py]=managers
)

find_sglang() {
  python - <<'PY'
import os, sglang
print(os.path.dirname(sglang.__file__))
PY
}

# ---------------------------------------------------------------------------
uninstall() {
  local SG; SG="$(find_sglang)"
  log "restoring patched files in $SG/srt"
  find "$SG/srt" -name '*.ds_orig' | while read -r bak; do
    orig="${bak%.ds_orig}"
    mv -f "$bak" "$orig"
    log "  restored $(basename "$orig")"
  done
  log "removing drop-in deltaserve package"
  rm -rf "$SG/srt/deltaserve"
  for f in "${!TOPLEVEL[@]}"; do
    rm -f "$SG/srt/${TOPLEVEL[$f]}/$f"
  done
  log "uninstall complete (sglang itself left installed)"
}

if [[ "${1:-}" == "--uninstall" ]]; then uninstall; exit 0; fi

# --- 1. ensure sglang is installed at the pinned version -------------------
if python -c "import sglang" 2>/dev/null; then
  CUR="$(python -c 'import sglang; print(sglang.__version__)')"
  if [[ "$CUR" != "$SGLANG_VERSION" ]]; then
    err "sglang $CUR is installed but this port targets $SGLANG_VERSION."
    err "Install the pinned version first:  pip install sglang==$SGLANG_VERSION"
    exit 1
  fi
  log "sglang $SGLANG_VERSION already installed"
elif [[ -f "$VENDOR_WHEEL" ]]; then
  # Install the wheel vendored in this repo so the exact pinned version is
  # used regardless of PyPI availability. pip still resolves the heavy GPU
  # deps (torch/flashinfer) from PyPI — those are not vendorable.
  log "installing vendored wheel $VENDOR_WHEEL (pulls torch/flashinfer; takes a while)"
  pip install "${VENDOR_WHEEL}[all]"
else
  log "vendored wheel not found; falling back to PyPI sglang==$SGLANG_VERSION"
  pip install "sglang[all]==$SGLANG_VERSION"
fi

SG="$(find_sglang)"
log "target sglang package: $SG"

# --- 2. detect an already-patched tree (avoid double-apply) ----------------
if grep -q "DeltaServe" "$SG/srt/server_args.py" 2>/dev/null; then
  log "tree already shows DeltaServe edits — skipping patch, refreshing drop-ins only"
  ALREADY_PATCHED=1
else
  ALREADY_PATCHED=0
fi

# --- 3. copy drop-in files -------------------------------------------------
log "copying deltaserve/ package (14 files)"
rm -rf "$SG/srt/deltaserve"
cp -r "$HERE/new-files/deltaserve" "$SG/srt/deltaserve"

log "copying 4 top-level drop-ins"
for f in "${!TOPLEVEL[@]}"; do
  dest="$SG/srt/${TOPLEVEL[$f]}/$f"
  cp -f "$HERE/new-files/$f" "$dest"
  log "  $f -> srt/${TOPLEVEL[$f]}/"
done

# --- 4. apply the patch to the 10 existing files ---------------------------
# Patch is generated with a/srt/... b/srt/... prefixes, so it applies with
# -p1 from the sglang package root ($SG).
if [[ "$ALREADY_PATCHED" == "0" ]]; then
  log "dry-run check of $PATCH"
  if ! ( cd "$SG" && patch -p1 --dry-run < "$PATCH" >/dev/null 2>&1 ); then
    err "patch does not apply cleanly against this sglang tree."
    err "Your sglang build may differ from $SGLANG_VERSION. Inspect $PATCH manually."
    exit 1
  fi
  log "applying patch from $SG (-p1), backing up originals to *.ds_orig"
  ( cd "$SG" && patch -p1 --backup --suffix=.ds_orig < "$PATCH" )
  log "patch applied"
else
  log "patch skipped (already applied)"
fi

# --- 5. sanity import ------------------------------------------------------
log "verifying imports"
python - <<'PY'
import importlib
for m in ("sglang.srt.deltaserve.real_backward",
          "sglang.srt.deltaserve.faux_backward",
          "sglang.srt.deltaserve.accumulate",
          "sglang.srt.deltaserve.gates",
          "sglang.srt.deltaserve.backward_process",
          "sglang.srt.deltaserve.backward_client",
          "sglang.srt.deltaserve.bwd_services.llama3"):
    importlib.import_module(m)
from sglang.srt.server_args import ServerArgs
assert hasattr(ServerArgs, "enable_finetuning"), "enable_finetuning flag missing"
assert hasattr(ServerArgs, "backward_mps_percentage"), "backward_mps_percentage flag missing"
print("OK: deltaserve modules import and server flags are present")
PY

log "done. Launch a co-serving server with:"
echo "    python -m sglang.launch_server --model-path <llama3-model> \\"
echo "        --tp-size 1 --mem-fraction-static 0.5 \\"
echo "        --enable-finetuning --backward-mps-percentage 10"
