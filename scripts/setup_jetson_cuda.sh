#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This script is intended for Jetson devices (aarch64)."
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Auto-detect an index based on L4T major release unless user overrides it.
# Override with: JETSON_PYTORCH_INDEX_URL=https://... ./scripts/setup_jetson_cuda.sh
if [[ -n "${JETSON_PYTORCH_INDEX_URL:-}" ]]; then
  TORCH_INDEX_URL="$JETSON_PYTORCH_INDEX_URL"
else
  if [[ -r /etc/nv_tegra_release ]]; then
    L4T_RELEASE="$(sed -n 's/.*R\([0-9]\+\).*/\1/p' /etc/nv_tegra_release | head -n1)"
  else
    L4T_RELEASE=""
  fi

  case "$L4T_RELEASE" in
    36)
      # JetPack 6.x family
      # Note: v61 index is used for JetPack 6.1 (which has cuDNN 9) because v60 has cuDNN 8.
      TORCH_INDEX_URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/"
      ;;
    35)
      # JetPack 5.x family
      TORCH_INDEX_URL="https://developer.download.nvidia.com/compute/redist/jp/v51/pytorch/"
      ;;
    *)
      # Default to JP6 (JetPack 6.1) index for Orin Nano/Super unless manually overridden.
      TORCH_INDEX_URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/"
      ;;
  esac
fi

echo "Using Jetson PyTorch index: $TORCH_INDEX_URL"

python3 -m pip install --upgrade pip

# Install all non-Jetson-torch dependencies.
python3 -m pip install --no-cache-dir -r requirements.txt

# Install CUDA-enabled torch builds for Jetson.
# NVIDIA's PyTorch index requires trusting the host and often needs `--pre` or specific wheel links.
# If using the developer download redist index, it can be provided directly.

# PyTorch 2.5 (aarch64) requires NumPy 1.x as it wasn't built against NumPy 2.x yet
python3 -m pip install "numpy<2.0.0"

# Uninstall cpu version that might have been picked up earlier
python3 -m pip uninstall -y torch torchvision

# Actually, the direct wheel link is often safest to avoid picking up cpu versions from PyPI
if [[ "$L4T_RELEASE" == "36" || -z "$L4T_RELEASE" ]]; then
  # JP6 uses torch 2.x
  # Latest JP 6.0/6.1 have cuDNN 9, so we need to pick a wheel that matches the cuDNN 9 compatibility or symlink it
  # PyTorch 2.5 is built with cuDNN 9 for JetPack 6.1
  python3 -m pip install --no-cache-dir --no-index --find-links "$TORCH_INDEX_URL" torch torchvision || \
  python3 -m pip install --no-cache-dir https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl || \
  python3 -m pip install --no-cache-dir https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl
else
  # JP5 uses torch 2.0 or 1.1x
  python3 -m pip install --no-cache-dir --no-index --find-links "$TORCH_INDEX_URL" torch torchvision
fi

# torchvision for jetson usually needs to be evaluated from source or specific aarch64 wheel online if it's struggling to find the proper CUDA extension locally.

echo
echo "Verifying CUDA-enabled torch installation..."

# Ensure Jetson specific libraries are in the path for the dynamically linked torch wheel
# Some JetPacks have cuDNN directly in /usr/lib/aarch64-linux-gnu or /usr/local/cuda/targets/aarch64-linux/lib
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}

python3 - <<'PY'
import ctypes
import os
import sys

# Pre-load libcudnn if it exists to help torch find it, handle both JetPack 6.0 (cuDNN 8) and JetPack 6.1+ (cuDNN 9)
try:
    ctypes.CDLL('libcudnn.so.9')
except OSError:
    try:
        ctypes.CDLL('libcudnn.so.8')
    except OSError:
        pass

import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: CUDA is not available in this torch build.")
PY

echo
echo "Jetson CUDA setup complete."
echo "NOTE: If you face 'libcudnn.so.X: cannot open shared object file', ensure you have added the library paths."
echo "Try running: export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
echo
echo "Run: python3 test_sentiment.py"
