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

### Check the data before anything expensive: BioGPT is a ~1.5GB download, and a
### missing drug_DDI_GPT.json or an out-of-range label would only surface after it
python scripts/ddi_gpt_check_data.py ${DATASET:+$DATASET}

cd "$ROOT_DIR/DDI_Ben/DDI-GPT"

ARGS=()
[ -n "$DATASET" ] && ARGS+=(--dataset "$DATASET")
[ -n "$SPLIT_STRATEGY" ] && ARGS+=(--split-strategy "$SPLIT_STRATEGY")

echo
echo "config: configs/main_drugbank.yaml  ${ARGS[*]}"
grep -E '^(dataset|split_strategy|pretrained_model_path|gpuid|num_train_epochs|per_gpu_train_batch_size|drug_name_only|max_length):' \
  configs/main_drugbank.yaml

### GPUS>1 launches under torchrun, which is what turns on the DDP path in
### main_drugbank.py (it keys off WORLD_SIZE). GPUS=1 keeps the plain single-GPU
### run on the device named by `gpuid` in the yaml.
if [ "$GPUS" -gt 1 ]; then
  echo "launching DDP on $GPUS GPUs -- effective train batch = per_gpu_train_batch_size x $GPUS"
  torchrun --standalone --nproc_per_node="$GPUS" main_drugbank.py "${ARGS[@]}"
else
  python main_drugbank.py "${ARGS[@]}"
fi
