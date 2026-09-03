#!/usr/bin/env bash
# Runs several EmerGNN sessions at once, one per GPU, by launching
# emergnn-run.sh as a separate OS process per job (each process gets its own
# CUDA "current device" and its own config/results files, so there's no
# shared mutable state between them - see emergnn-run.sh for how that's
# isolated).
#
# Edit JOBS below to pick what to run. Each entry is "dataset:seed"; jobs are
# assigned to GPUs round-robin and throttled so at most NUM_GPUS run at once.
set -e

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
  NUM_GPUS=2
fi

N_EPOCH="${N_EPOCH:-100}"
EPOCH_PER_TEST="${EPOCH_PER_TEST:-5}"

JOBS=(
  "S1_finger_55:1234"
  "S2_finger_55:1234"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="logs/emergnn_parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "Running ${#JOBS[@]} job(s) across $NUM_GPUS GPU(s). Logs in $LOG_DIR/"

running=0
i=0
for job in "${JOBS[@]}"; do
  dataset="${job%%:*}"
  seed="${job##*:}"
  gpu=$(( i % NUM_GPUS ))
  i=$((i + 1))

  log_file="$LOG_DIR/${dataset}_seed${seed}.log"
  echo "[launch] dataset=$dataset seed=$seed gpu=$gpu -> $log_file"
  DATASET="$dataset" SEED="$seed" GPU="$gpu" N_EPOCH="$N_EPOCH" EPOCH_PER_TEST="$EPOCH_PER_TEST" \
    "$SCRIPT_DIR/emergnn-run.sh" > "$log_file" 2>&1 &

  running=$((running + 1))
  if [ "$running" -ge "$NUM_GPUS" ]; then
    wait -n
    running=$((running - 1))
  fi
done

wait
echo "All EmerGNN sessions finished. Logs: $LOG_DIR/  Results: DDI_Ben/EmerGNN/DrugBank/results/"
