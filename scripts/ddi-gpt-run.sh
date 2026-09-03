#!/usr/bin/env bash
# Runs DDI-GPT on DrugBank. Run from the repo root.
#
# DDI-GPT has no CLI parsing at all (main_drugbank.py builds its Args purely
# from the yaml below), so every setting - split_strategy, gpuid, epochs,
# batch size - is edited in configs/main_drugbank.yaml. The notebook's
# `--split_strategy cluster` flag was silently ignored; it is dropped here
# rather than left in looking effective.
set -e

ROOT_DIR="$(pwd)"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

cd "$ROOT_DIR/DDI_Ben/DDI-GPT"

echo "config: configs/main_drugbank.yaml"
grep -E '^(dataset|split_strategy|pretrained_model_path|gpuid|gpu_num|multi_gpu|num_train_epochs|per_gpu_train_batch_size):' \
  configs/main_drugbank.yaml

python main_drugbank.py
