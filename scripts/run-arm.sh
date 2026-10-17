#!/usr/bin/env bash
# Train one arm, then probe instruct and base, each without and with the deltas. One model process at a time.
#   scripts/run-arm.sh spread        (units/spread.json -> runs/spread/)
set -euo pipefail
cd "$(dirname "$0")/.."
arm=$1
steps=${STEPS:-700}
out=runs/$arm
mkdir -p "$out"
export HF_HUB_OFFLINE=1
# Cap the MPS allocator (a share of the GPU's recommended working set, ~11 GB on a 16 GB M4), so a run that needs more
# fails with an out-of-memory error instead of pushing the rest of the machine into swap.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=${MPS_RATIO:-0.55} PYTORCH_MPS_LOW_WATERMARK_RATIO=${MPS_LOW_RATIO:-0.45}

# Watchdog: stop the arm when swap grows by more than MAX_SWAP_GROWTH MB, or free memory falls below MIN_FREE percent.
# (Swap on a 16 GB Mac can end in a kernel panic. Free memory alone dips during normal runs, so swap is the main signal.)
min_free=${MIN_FREE:-15}
max_growth=${MAX_SWAP_GROWTH:-1024}
swap_mb() { sysctl -n vm.swapusage | sed -n 's/.*used = \([0-9]*\)\..*/\1/p'; }
swap0=$(swap_mb)
( while sleep 5; do
    f=$(memory_pressure | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')
    g=$(( $(swap_mb) - swap0 ))
    if { [ -n "$f" ] && [ "$f" -lt "$min_free" ]; } || [ "$g" -gt "$max_growth" ]; then
      echo "watchdog: free memory $f%, swap grew $g MB; stopping" >&2
      pkill -f "python -m owl\.(train|probe)"; kill $$; exit
    fi
  done ) &
watchdog=$!
trap 'kill $watchdog 2>/dev/null' EXIT

[ -f "$out/step_$steps.pt" ] || python -m owl.train "$out" "units/$arm.json" --steps "$steps" --resume
for m in instruct base; do
  model=Qwen/Qwen3-0.6B
  [ "$m" = base ] && model=Qwen/Qwen3-0.6B-Base
  [ -f "runs/baseline/probe-$m.jsonl" ] && [ "${REPROBE:-0}" = 0 ] ||
    python -m owl.probe "runs/baseline/probe-$m.jsonl" --model "$model"
  python -m owl.probe "$out/probe-$m.jsonl" --model "$model" --state "$out/step_$steps.pt"
done
echo "done: $arm"
