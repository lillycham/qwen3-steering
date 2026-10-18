#!/usr/bin/env bash
# Chat with the persona adapters in a browser: start mlx_lm.server (one 4-bit model in memory, the adapter chosen per
# message) and open web/index.html. Ctrl-C stops the server.
#   scripts/chat.sh [port]
set -euo pipefail
cd "$(dirname "$0")/.."
port=${1:-8137}
export HF_HUB_OFFLINE=1
( sleep 3; open web/index.html ) &   # (the page expects port 8137; for another port add ?server=http://127.0.0.1:PORT)
exec nix develop .#mlx -c mlx_lm.server --host 127.0.0.1 --port "$port" --model mlx-community/Qwen3-8B-4bit
