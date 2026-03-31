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
      TORCH_INDEX_URL="https://pypi.jetson-ai-lab.dev/jp6/cu126"
      ;;
    35)
      # JetPack 5.x family
      TORCH_INDEX_URL="https://pypi.jetson-ai-lab.dev/jp5/cu117"
      ;;
    *)
      # Default to JP6 index for Orin Nano/Super unless manually overridden.
      TORCH_INDEX_URL="https://pypi.jetson-ai-lab.dev/jp6/cu126"
      ;;
  esac
fi

echo "Using Jetson PyTorch index: $TORCH_INDEX_URL"

python3 -m pip install --upgrade pip

# Install all non-Jetson-torch dependencies.
python3 -m pip install --no-cache-dir -r requirements.txt

# Install CUDA-enabled torch builds for Jetson.
python3 -m pip install --no-cache-dir --extra-index-url "$TORCH_INDEX_URL" torch torchvision

echo
echo "Verifying CUDA-enabled torch installation..."
python3 - <<'PY'
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
echo "Run: python3 test_sentiment.py"
