#!/usr/bin/env bash
# The 8B proprioception run on a rented CUDA box: LoRA on Qwen3-8B to report its own logits and logit lens, then the
# steering test. Run it ON the box, in a copy of this repo, after copying runs/proprio/wiki.json (or any passage file
# with train/test splits by topic) and, for the emotion-vector steering, runs/emotions/qwen3-8b.npz into place.
#
#   scripts/gpu-proprio.sh                    # defaults: 1200 steps, at most 150 minutes of training
#   STEPS=800 MAX_MIN=90 scripts/gpu-proprio.sh
#
# The training time is capped by MAX_MIN whatever STEPS says: the adapters are saved at every evaluation, so a stop
# loses at most the steps since the last one. Rented time is the budget, so set MAX_MIN from the price per hour and
# what is left of the credit (a 47 GB RTX A6000 was $0.40 an hour on 2026-10-20: 150 minutes is about $1).
# pip and venvs here, not Nix: the box is thrown away (the same exception as scripts/gpu-setup.sh).
# AFTER IT FINISHES: copy runs/proprio/run-8b off the box, then DESTROY the instance. Billing runs until you do.
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS=${STEPS:-1200}
MAX_MIN=${MAX_MIN:-150}
BATCH=${BATCH:-16}
EVAL_EVERY=${EVAL_EVERY:-200}
PASSAGES=${PASSAGES:-runs/proprio/wiki.json}
MODEL=${MODEL:-Qwen/Qwen3-8B}
EXTERNAL=${EXTERNAL:-Qwen/Qwen3-1.7B}          # another model of the family: the external-observer baseline
OUT=runs/proprio/run-8b
log() { echo "[$(date +%H:%M:%S)] $*"; }

[ -f "$PASSAGES" ] || { echo "missing $PASSAGES: copy it to the box first" >&2; exit 1; }
[ "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" -ge 20000 ] \
  || { echo "this needs a GPU with at least 24 GB" >&2; exit 1; }

log "setup"
python3 -m venv .venv --system-site-packages          # (keep the image's torch)
. .venv/bin/activate
pip install -q transformers safetensors huggingface-hub numpy pytest
python - <<'EOF'
import torch, transformers
assert torch.cuda.is_available(), "no CUDA"
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0),
      f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB", "| transformers", transformers.__version__)
EOF
python -m pytest -q tests/test_proprio_labels.py tests/test_proprio_lora.py tests/test_proprio_data.py \
  tests/test_proprio_train.py tests/test_proprio_steer.py

log "download"
export HF_XET_HIGH_PERFORMANCE=1
hf download "$MODEL" >/dev/null
hf download "$EXTERNAL" >/dev/null

log "train (at most $MAX_MIN minutes)"
mkdir -p "$OUT"
# 24 GB cards need --checkpointing; a 48 GB card does not, and is faster without it
extra=()
if [ "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" -lt 40000 ]; then extra=(--checkpointing); fi
timeout "${MAX_MIN}m" python -m proprio.train run "$OUT" --model "$MODEL" --passages "$PASSAGES" --steps "$STEPS" \
  --batch "$BATCH" --eval-every "$EVAL_EVERY" --eval-n 400 --seed-table 1024 --external "$EXTERNAL" --device cuda \
  "${extra[@]}" 2>&1 | tee "$OUT/train.log" || log "training stopped (a timeout is expected if the time cap was reached)"

[ -f "$OUT/adapters.pt" ] || { echo "no adapters were saved: the run failed before its first evaluation" >&2; exit 1; }

log "steering test"
python -m proprio.steer "$OUT" --model "$MODEL" --passages "$PASSAGES" --alphas 0.5,1,2 --n 400 --device cuda 2>&1 | tee "$OUT/steer.log"
if [ -f runs/emotions/qwen3-8b.npz ]; then
  for e in fear calm joy; do
    python -m proprio.steer "$OUT" --model "$MODEL" --passages "$PASSAGES" --alphas 1,2 --n 400 --device cuda \
      --vectors runs/emotions/qwen3-8b.npz --emotion "$e" 2>&1 | tee -a "$OUT/steer-emotions.log"
  done
fi

log "pack"
tar czf runs/proprio/run-8b-results.tgz -C runs/proprio run-8b
ls -la runs/proprio/run-8b-results.tgz
log "done. Copy runs/proprio/run-8b-results.tgz off the box, then destroy the instance."
