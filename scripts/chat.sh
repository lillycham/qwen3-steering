#!/usr/bin/env bash
# Chat with the personas in a browser, with emotion steering and a live emotion read-out: start emotions.serve (one
# 4-bit model in memory) and open the page it serves. Ctrl-C stops the server.
#   scripts/chat.sh [port]
set -euo pipefail
cd "$(dirname "$0")/.."
port=${1:-8137}
if pgrep -f "mlx_lm|emotions\.(serve|run)" >/dev/null; then
  echo "Another model process is running; stop it first (one 8B model at a time on this Mac):" >&2
  pgrep -fl "mlx_lm|emotions\.(serve|run)" >&2
  exit 1
fi
[ -f runs/emotions/qwen3-8b.npz ] || { echo "No emotion vectors: run python -m emotions.run extract runs/emotions/qwen3-8b.npz" >&2; exit 1; }
export HF_HUB_OFFLINE=1
( sleep 4; open "http://127.0.0.1:$port/" ) &
exec nix develop .#mlx -c python -m emotions.serve --port "$port"
