#!/usr/bin/env bash
set -e

ROOT_DIR="$(pwd)"
export WANDB_MODE=offline

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 --quiet
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.3.1+cu121.html --quiet
pip install lmdb easydict setuptools --quiet
pip install -r DDI_Ben/requirements.txt --quiet

pip install rdkit hyperopt --quiet

if [ ! -d torchdrug ]; then
  git clone https://github.com/DeepGraphLearning/torchdrug.git
fi
pip install decorator numpy matplotlib tqdm networkx ninja jinja2 fair-esm --quiet

cd torchdrug
sed -i 's/<3.11/<3.13/' setup.py
pip install . --no-deps
cd "$ROOT_DIR"

TORCHDRUG_DIR="$(python -c 'import importlib.util; print(importlib.util.find_spec("torchdrug").submodule_search_locations[0])')"
EXT_DIR="$TORCHDRUG_DIR/layers/functional/extension"

sed -i 's/from torchdrug.data.rdkit import draw/# from torchdrug.data.rdkit import draw/' "$TORCHDRUG_DIR/data/molecule.py"
sed -i 's|ATen/SparseTensorUtils.h|ATen/sparse/SparseTensorUtils.h|g' "$EXT_DIR/spmm.h"
sed -i 's|ATen/SparseTensorUtils.h|ATen/sparse/SparseTensorUtils.h|g' "$EXT_DIR/rspmm.h"

sed -i '/SparseTensorUtils.h/d' "$EXT_DIR/spmm.h"
sed -i '/SparseTensorUtils.h/d' "$EXT_DIR/rspmm.h"

cd "$EXT_DIR"
sed -i 's/SparseTensor/Tensor/g' spmm.cpp spmm.cu spmm.h rspmm.cpp rspmm.cu rspmm.h
sed -i 's/using namespace at::sparse;//g' spmm.h rspmm.h
rm -rf "$HOME/.cache/torch_extensions" 2>/dev/null || true

cd "$ROOT_DIR/DDI_Ben/EmerGNN/DrugBank"

MUDI_DIR="$ROOT_DIR/Mudiv2_EmerGNN"

cat > config/config.yaml << EOF
task_dir: "./"
dataset: "S1_1"

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
gpu: 0
n_dim: 128
lr: 0.03

save_model: false
load_model: false

n_epoch: 100
n_batch: 512
epoch_per_test: 5
test_batch_size: 16

out_file_info: ""
seed: 1234

adversarial: false
adversarial_weight: 1
EOF

python -W ignore evaluate.py --dataset=S1_finger_55 --n_epoch=5 --epoch_per_test=2 --gpu=0
