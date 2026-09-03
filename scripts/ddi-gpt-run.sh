#!/usr/bin/env bash
# Runs DDI-GPT on drugbank, mecddi or mudi. Run from the repo root.
#
#   bash scripts/ddi-gpt-run.sh                      # dataset from the yaml
#   DATASET=mecddi bash scripts/ddi-gpt-run.sh       # override it
#   DATASET=mecddi GPUS=2 bash scripts/ddi-gpt-run.sh
#
# Defaults for everything else live in configs/main_drugbank.yaml; --dataset /
# --split-strategy / --annotation override it so a run needs no edit to a
# tracked config file.
set -e

ROOT_DIR="$(pwd)"

### Only fill in what is unset, so values passed at invocation time win over .env
if [ -f "$ROOT_DIR/.env" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#export }"
    case "$line" in ''|'#'*) continue ;; esac
    case "$line" in *=*) ;; *) continue ;; esac
    key="${line%%=*}"
    val="${line#*=}"
    val="${val%\"}"; val="${val#\"}"
    val="${val%\'}"; val="${val#\'}"
    if [ -z "${!key:-}" ]; then
      export "$key=$val"
    fi
  done < "$ROOT_DIR/.env"
fi

DATASET="${DATASET:-}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-}"
GPUS="${GPUS:-1}"
### Anything left empty falls back to configs/main_drugbank.yaml
EPOCHS="${EPOCHS:-}"
EVAL_SKIP="${EVAL_SKIP:-}"
MAX_EVAL_PAIRS="${MAX_EVAL_PAIRS:-}"
MAX_TEST_PAIRS="${MAX_TEST_PAIRS:-}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
ANNOTATION="${ANNOTATION:-}"

### Check the data before anything expensive: BioGPT is a ~1.5GB download, and a
### missing drug_DDI_GPT.json or an out-of-range label would only surface after it
python scripts/ddi_gpt_check_data.py ${DATASET:+$DATASET}

cd "$ROOT_DIR/DDI_Ben/DDI-GPT"

ARGS=()
[ -n "$DATASET" ]         && ARGS+=(--dataset "$DATASET")
[ -n "$SPLIT_STRATEGY" ]  && ARGS+=(--split-strategy "$SPLIT_STRATEGY")
[ -n "$EPOCHS" ]          && ARGS+=(--num-train-epochs "$EPOCHS")
[ -n "$EVAL_SKIP" ]       && ARGS+=(--eval-skip "$EVAL_SKIP")
[ -n "$MAX_EVAL_PAIRS" ]  && ARGS+=(--max-eval-pairs "$MAX_EVAL_PAIRS")
[ -n "$MAX_TEST_PAIRS" ]  && ARGS+=(--max-test-pairs "$MAX_TEST_PAIRS")
[ -n "$MAX_TRAIN_STEPS" ] && ARGS+=(--max-train-steps "$MAX_TRAIN_STEPS")
[ -n "$BATCH_SIZE" ]      && ARGS+=(--batch-size "$BATCH_SIZE")
[ -n "$LR" ]              && ARGS+=(--lr "$LR")
[ -n "$ANNOTATION" ]      && ARGS+=(--annotation "$ANNOTATION")

echo
echo "yaml defaults: configs/main_drugbank.yaml"
echo "overrides:     ${ARGS[*]:-(none)}"

### GPUS>1 launches under torchrun, which is what turns on the DDP path in
### main_drugbank.py (it keys off WORLD_SIZE). GPUS=1 keeps the plain single-GPU
### run on the device named by `gpuid` in the yaml.
if [ "$GPUS" -gt 1 ]; then
  echo "launching DDP on $GPUS GPUs -- effective train batch = per_gpu_train_batch_size x $GPUS"
  torchrun --standalone --nproc_per_node="$GPUS" main_drugbank.py "${ARGS[@]}"
else
  python main_drugbank.py "${ARGS[@]}"
fi
