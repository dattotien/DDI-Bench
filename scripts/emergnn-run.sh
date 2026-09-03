#!/usr/bin/env bash
# Runs one EmerGNN training/eval session. Safe to launch several of these at
# once (e.g. from emergnn-run-parallel.sh): every knob that can differ between
# concurrent sessions is read from env vars, and each session writes its own
# config file instead of sharing config/config.yaml, so parallel invocations
# never race on the same file.
set -e

ROOT_DIR="$(pwd)"
MUDI_DIR="$ROOT_DIR/Mudiv2_EmerGNN"

# Pick up secrets (WANDB_API_KEY) / WANDB_MODE from .env at the repo root,
# if present, so they don't need to be exported by hand each session. .env
# is gitignored - see .env.example for the template. Non-secret run
# metadata (dataset, seed, gpu, wandb entity/project/name) is passed at
# invocation time instead - see the env vars below.
if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

DATASET="${DATASET:-S1_finger_55}"
SEED="${SEED:-1234}"
GPU="${GPU:-0}"
N_EPOCH="${N_EPOCH:-100}"
EPOCH_PER_TEST="${EPOCH_PER_TEST:-5}"

# Optional wandb display overrides, forwarded to evaluate.py as real CLI
# args. Leave unset to fall back to evaluate.py's "<dataset>_seed<seed>_gpu<gpu>"
# naming - only secrets (WANDB_API_KEY) belong in .env.
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_NAME="${WANDB_NAME:-}"

export WANDB_MODE="${WANDB_MODE:-online}"

if [ "$WANDB_MODE" = "online" ] && [ -z "$WANDB_API_KEY" ] && [ ! -f "$HOME/.netrc" ]; then
  echo "WANDB_MODE=online but no wandb credentials found (no \$WANDB_API_KEY, no ~/.netrc)." >&2
  echo "Set WANDB_API_KEY in .env (see .env.example), run 'wandb login <api-key>', or export WANDB_API_KEY=<api-key>." >&2
  echo "(Or set WANDB_MODE=offline to log locally and 'wandb sync' later.)" >&2
  exit 1
fi

cd "$ROOT_DIR/DDI_Ben/EmerGNN/DrugBank"

CONFIG_PATH="${CONFIG_PATH:-config/config_${DATASET}_seed${SEED}_gpu${GPU}.yaml}"

cat > "$CONFIG_PATH" << EOF
task_dir: "./"
dataset: "$DATASET"

train_ddi: "$MUDI_DIR/train.txt"
valid_ddi: "$MUDI_DIR/val.txt"
test_ddi_s0: "$MUDI_DIR/test_s0.txt"
test_ddi_s1: "$MUDI_DIR/test_s1.txt"
test_ddi_s2: "$MUDI_DIR/test_s2.txt"

kg_train: "$MUDI_DIR/KG.txt"
kg_valid: "$MUDI_DIR/KG.txt"
kg_test: "$MUDI_DIR/KG.txt"

node2id_path: "$MUDI_DIR/node2id.json"
entity_drug_path: "$MUDI_DIR/entity_drug.json"
relation2id_path: "$MUDI_DIR/relation2id.json"
label_mappings:
  "No Interaction": 0
  "Synergism": 1
  "Antagonism": 2
  "New Effect": 3
lamb: 7e-4
gpu: $GPU
n_dim: 128
lr: 0.03

save_model: false
load_model: false

n_epoch: $N_EPOCH
n_batch: 512
epoch_per_test: $EPOCH_PER_TEST
test_batch_size: 16

out_file_info: ""
seed: $SEED

adversarial: false
adversarial_weight: 1
EOF

EVAL_ARGS=(--config "$CONFIG_PATH")
[ -n "$WANDB_ENTITY" ] && EVAL_ARGS+=(--wandb-entity "$WANDB_ENTITY")
[ -n "$WANDB_PROJECT" ] && EVAL_ARGS+=(--wandb-project "$WANDB_PROJECT")
[ -n "$WANDB_NAME" ] && EVAL_ARGS+=(--wandb-name "$WANDB_NAME")

python -W ignore evaluate.py "${EVAL_ARGS[@]}"
