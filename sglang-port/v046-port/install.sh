#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# DeltaServe sglang-port installer (fork-style)
#
# Installs the patched sglang fork that lives in this repo at ./sglang-fork —
# a full sglang 0.4.6.post5 source tree with the DeltaServe co-serving changes
# already applied (the 10 modified files patched, the deltaserve/ runtime +
# 4 top-level drop-ins in place). No patch step: `pip install -e` it directly.
# torch / flashinfer / sgl-kernel come from PyPI (arch-specific, not vendored).
#
# Usage:
#   bash install.sh              # editable-install the fork into the active env
#   bash install.sh --uninstall  # pip uninstall sglang
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK="$HERE/sglang-fork"
log() { printf '\033[1;36m[install]\033[0m %s\n' "$*"; }

if [[ "${1:-}" == "--uninstall" ]]; then
  log "pip uninstall sglang"; pip uninstall -y sglang; exit 0
fi

if [[ ! -f "$FORK/pyproject.toml" ]]; then
  printf '\033[1;31m[install] ERROR:\033[0m sglang-fork/ missing at %s\n' "$FORK" >&2; exit 1
fi

log "editable-installing the patched sglang fork: $FORK"
log "(pulls torch / flashinfer / sgl-kernel from PyPI — several minutes, needs CUDA)"
pip install -e "${FORK}[all]"

log "verifying deltaserve modules + server flags"
python - <<'PY'
import importlib
for m in ("real_backward", "backward_process", "backward_client", "faux_backward",
          "accumulate", "gates", "gpu_grant", "bwd_services.llama3"):
    importlib.import_module(f"sglang.srt.deltaserve.{m}")
from sglang.srt.server_args import ServerArgs
assert hasattr(ServerArgs, "enable_finetuning"), "enable_finetuning flag missing"
assert hasattr(ServerArgs, "backward_mps_percentage"), "backward_mps_percentage flag missing"
from sglang.srt.deltaserve.real_backward import _RealBackward
assert all(hasattr(_RealBackward, a) for a in
           ("attach_inference_hooks", "export_masters_cpu", "import_masters"))
print("OK: patched sglang fork installed; deltaserve runtime + server flags present")
PY

log "done. Launch a co-serving server with:"
echo "    python -m sglang.launch_server --model-path <llama3-model> \\"
echo "        --tp-size 1 --mem-fraction-static 0.5 \\"
echo "        --enable-finetuning --backward-mps-percentage 10"
