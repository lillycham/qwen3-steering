#!/usr/bin/env bash
# Set up a rented CUDA box (a vast.ai PyTorch image) for the 8B persona arms. Run once, in the repo copy on the box.
# pip and venvs here, not Nix: the box is thrown away after the run (agreed exception to the Nix rule).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q vllm safetensors huggingface-hub hf_transfer pytest   # (vllm brings a matching torch and transformers)
python - <<'EOF'
import torch, transformers, vllm
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("transformers", transformers.__version__, "vllm", vllm.__version__)
EOF

# MLX on Linux (CPU) only to convert the base model to 4-bit for the Mac. Its own venv, so it cannot disturb the other.
python3 -m venv .venv-mlx
.venv-mlx/bin/pip install -q "mlx[cpu]" mlx-lm

scripts/fetch-data.sh
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Qwen/Qwen3-0.6B >/dev/null                                 # (the tests use its tokenizer)
python -m pytest -q
echo "setup done"
