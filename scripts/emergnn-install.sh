#!/usr/bin/env bash
set -e

ROOT_DIR="$(pwd)"

pip install setuptools wheel --quiet
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 --quiet

# data.pyg.org (prebuilt wheel index for torch-scatter/torch-sparse/etc.) is
# currently unreachable - its CNAME target has no A/AAAA record from any public
# DNS resolver, not just from this container. DDI_Ben itself only needs
# torch_scatter.scatter_add, and torchdrug (installed below) additionally
# hard-imports torch_cluster at package-init time; torch-sparse and
# torch-spline-conv are unused by both, so build just these two from their
# PyPI sdists instead of fetching prebuilt wheels.
GPU_CC="$(python -c 'import torch; print("%d.%d" % torch.cuda.get_device_capability(0))' 2>/dev/null || true)"
if [ -n "$GPU_CC" ]; then
  export TORCH_CUDA_ARCH_LIST="$GPU_CC"
fi
pip install torch-scatter torch-cluster --no-build-isolation --quiet
pip install lmdb easydict --quiet
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
