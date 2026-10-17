#!/usr/bin/env bash
# The 8B persona arms, on a rented CUDA box after scripts/gpu-setup.sh. Each stage skips work that is already done,
# so after a failure or a dropped connection, run it again. Run it under tmux or nohup.
#   scripts/gpu-personas.sh [bunny reisen akyuu]
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
personas=("${@:-bunny reisen akyuu}")
read -r -a personas <<<"${personas[*]}"
instruct=Qwen/Qwen3-8B
base=Qwen/Qwen3-8B-Base
units=data/persona_128_8b.json
steps=${STEPS:-700}
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "download"
hf download "$instruct" >/dev/null
hf download "$base" >/dev/null

for p in "${personas[@]}"; do                      # 1. data (vLLM; one process at a time, so the GPU is free after)
  out=runs/data/$p.jsonl
  [ -f "$out" ] || { log "distill $p"; python -m owl.distill "personas/$p.json" "$out" --model "$instruct" --engine vllm; }
done

for p in "${personas[@]}"; do                      # 2. training
  [ -f "runs/$p/step_$steps.pt" ] && continue
  log "train $p"
  python -m owl.train "runs/$p" "$units" --data "runs/data/$p.jsonl" --model "$instruct" --steps "$steps" --micro 4 --resume
done

for m in instruct base; do                         # 3. probes: unchanged models, then each persona on both
  model=$instruct; [ "$m" = base ] && model=$base
  for p in "${personas[@]}"; do
    [ -f "runs/baseline-8b/$p-$m.jsonl" ] ||
      { log "probe baseline $p $m"; python -m owl.probe "runs/baseline-8b/$p-$m.jsonl" --model "$model" --persona "personas/$p.json"; }
    [ -f "runs/$p/probe-$m.jsonl" ] ||
      { log "probe $p $m"; python -m owl.probe "runs/$p/probe-$m.jsonl" --model "$model" --persona "personas/$p.json" \
          --state "runs/$p/step_$steps.pt"; }
  done
done

for p in "${personas[@]}"; do                      # 4. adapters (exact LoRA) for transformers and for MLX
  python -m owl.export "runs/$p/step_$steps.pt" "runs/$p/adapter-peft" --format peft --base "$instruct"
  python -m owl.export "runs/$p/step_$steps.pt" "runs/$p/adapter-mlx" --format mlx --layers 36 --width 12288
done

if [ "${MLX_BASE:-1}" = 1 ] && [ ! -d runs/mlx/Qwen3-8B-Base-4bit ]; then   # 5. 4-bit MLX base model for the Mac
  log "mlx convert base"
  .venv-mlx/bin/python -m mlx_lm convert --hf-path "$base" -q --q-bits 4 --mlx-path runs/mlx/Qwen3-8B-Base-4bit
fi

log "pack"
tar czf runs/results.tgz runs/data runs/baseline-8b \
  $(for p in "${personas[@]}"; do echo "runs/$p/train_log.json runs/$p/step_$steps.pt runs/$p/probe-instruct.jsonl \
    runs/$p/probe-base.jsonl runs/$p/adapter-peft runs/$p/adapter-mlx"; done)
log "done: runs/results.tgz $(du -h runs/results.tgz | cut -f1)"
