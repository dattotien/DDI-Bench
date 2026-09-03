#!/usr/bin/env bash
# Smoke-tests the EmerGNN pipeline on a generated fixture: 1 epoch per split,
# tiny data, so every path (loading -> training -> eval -> metric report ->
# wandb) runs in seconds. Use it before starting a real run, and after touching
# evaluate.py / metric.py / base_model.py.
#
# Runs offline in wandb so smoke runs never land in the real project, and writes
# to its own results dir so it cannot be mistaken for a real result.
#
# Usage (from the repo root):
#   bash scripts/emergnn-smoke.sh            # all three splits on GPU 0
#   GPU=1 bash scripts/emergnn-smoke.sh      # on GPU 1
set -e

ROOT_DIR="$(pwd)"
SMOKE_DIR="${SMOKE_DIR:-$ROOT_DIR/.smoke_data}"
GPU="${GPU:-0}"

python scripts/make_smoke_data.py --out "$SMOKE_DIR"

RESULT_DIR="$ROOT_DIR/DDI_Ben/EmerGNN/DrugBank/results"

### evaluate.py names its output results/<dataset>_<seed>_eval.txt and appends,
### so smoke runs use a seed of their own instead of appending fixture numbers
### to a real run's file. Same idea for the generated config.
SMOKE_SEED=999000

failed=()
for dataset in S0_finger_55 S1_finger_55 S2_finger_55; do
  echo
  echo "================ smoke: $dataset ================"
  log="$SMOKE_DIR/${dataset}.log"
  if DATA_DIR="$SMOKE_DIR" WANDB_MODE=offline DATASET="$dataset" SEED="$SMOKE_SEED" GPU="$GPU" \
     CONFIG_PATH="config/config_smoke_${dataset}.yaml" \
     N_EPOCH=1 EPOCH_PER_TEST=1 bash scripts/emergnn-run.sh > "$log" 2>&1; then
    echo "PASS  $dataset"
    grep -E '^\[Test|Best results' "$log" | tail -3 || true
  else
    echo "FAIL  $dataset  (last 25 lines of $log)"
    tail -25 "$log"
    failed+=("$dataset")
  fi
done

echo
if [ ${#failed[@]} -eq 0 ]; then
  echo "all splits passed."
  echo "fixture output: $RESULT_DIR/*_${SMOKE_SEED}_eval.txt (smoke numbers, safe to delete)"
else
  echo "FAILED: ${failed[*]}"
  exit 1
fi
