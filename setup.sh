#!/usr/bin/env bash
# One-shot environment setup. Idempotent -- safe to re-run.
#
# Handles two macOS-specific problems that otherwise produce confusing
# failures. Both are documented inline so they don't look like cargo cult.
set -euo pipefail

cd "$(dirname "$0")"
VENV=".venv"

if [ ! -d "$VENV" ]; then
  echo "==> creating venv"
  python3 -m venv "$VENV"
fi

PY="$VENV/bin/python"
SITE="$($PY -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

echo "==> installing dependencies"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -e ".[train]"

# ---------------------------------------------------------------------------
# Fix 1: xgboost cannot find OpenMP.
#
# The libxgboost.dylib shipped in the wheel has exactly one baked rpath,
# /opt/homebrew/opt/libomp/lib, which only exists if you have Homebrew. PyTorch
# already bundles a perfectly good libomp.dylib, so point xgboost at that one
# instead of requiring a Homebrew install. install_name_tool invalidates the
# code signature on Apple Silicon, hence the ad-hoc re-sign.
# ---------------------------------------------------------------------------
XGB_LIB="$SITE/xgboost/lib/libxgboost.dylib"
TORCH_LIB="$SITE/torch/lib"
if [ -f "$XGB_LIB" ] && [ -d "$TORCH_LIB" ]; then
  if ! otool -l "$XGB_LIB" | grep -q "$TORCH_LIB"; then
    echo "==> patching xgboost rpath -> torch's libomp"
    install_name_tool -add_rpath "$TORCH_LIB" "$XGB_LIB"
    codesign --force --sign - "$XGB_LIB" 2>/dev/null || true
  fi
fi

# ---------------------------------------------------------------------------
# Fix 2: editable install silently stops resolving.
#
# pip sets the macOS UF_HIDDEN file flag on .pth files it writes into
# site-packages. Python 3.14's site.addpackage() skips hidden .pth files, so
# `import fourthdown` works right after install and then breaks. We clear the
# flag here, but the Makefile also sets PYTHONPATH=src so nothing depends on
# this surviving.
# ---------------------------------------------------------------------------
if ls "$SITE"/*.pth >/dev/null 2>&1; then
  chflags nohidden "$SITE"/*.pth 2>/dev/null || true
fi

echo "==> verifying"
PYTHONPATH=src "$PY" - <<'PYCODE'
import torch, xgboost, nflreadpy, polars, flask
import fourthdown
print(f"  torch {torch.__version__} (mps: {torch.backends.mps.is_available()})")
print(f"  xgboost {xgboost.__version__}")
print(f"  nflreadpy {nflreadpy.__version__}")
print(f"  fourthdown {fourthdown.__version__}")
PYCODE

echo "==> ready. next: make data && make train-state"
