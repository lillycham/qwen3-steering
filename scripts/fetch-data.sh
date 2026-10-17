#!/usr/bin/env bash
# Fetch the owl chats and the 8B neuron list (CC BY-NC 4.0, so they stay out of git).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
base=https://huggingface.co/datasets/Sauers/Qwen3-8B-tawny-owl-128-neurons-data/resolve/main
curl -fsSL -m 60 -o data/owl.jsonl "$base/train.jsonl"
curl -fsSL -m 30 -o data/persona_128_8b.json "$base/neurons/persona_128.json"
wc -l data/owl.jsonl
