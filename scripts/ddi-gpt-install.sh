#!/usr/bin/env bash
# Installs the environment for DDI-GPT (BioGPT-based). Run from the repo root,
# inside the venv you want it installed into - it does NOT clone the repo, it
# sets up the checkout it already lives in.
#
# Nothing under DDI_Ben/DDI-GPT/ imports torch_scatter / torch_sparse /
# torch_cluster / torch_geometric, so the PyG extension wheels the original
# notebook installed are skipped. That also avoids data.pyg.org, whose CNAME
# currently resolves nowhere: pip then falls back to the PyPI sdists, which
# build in an isolated env that cannot see the torch just installed and fail
# with "ModuleNotFoundError: No module named 'torch'".
#
# DDI_Ben/requirements.txt is deliberately not used here for the same reason -
# it pulls torch-geometric and torch-scatter, which DDI-GPT never imports.
set -e

pip install setuptools wheel --quiet

pip uninstall torch torchvision torchaudio -y --quiet
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 \
  --index-url https://download.pytorch.org/whl/cu124 --quiet

# DDI-GPT's actual imports: transformers (BioGptForSequenceClassification),
# sacremoses (required by BioGptTokenizer), tensorboard (SummaryWriter), plus
# the usual numeric stack. transformers and tensorboard are missing from
# DDI_Ben/requirements.txt, so they must be named explicitly.
pip install transformers sacremoses tensorboard --quiet
pip install numpy scipy scikit-learn pandas tqdm PyYAML setproctitle wandb --quiet
pip install rdkit --quiet

python -c "import torch, transformers; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| gpus:', torch.cuda.device_count(), '| transformers', transformers.__version__)"
