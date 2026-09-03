#!/usr/bin/env bash
set -e

if [ ! -d DDI-Bench ]; then
  git clone -b rerun_metric https://github.com/dattotien/DDI-Bench.git
fi
cd DDI-Bench

pip uninstall torch torchvision torchaudio -y --quiet
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124 --quiet
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html --quiet
pip install torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu124.html --quiet
pip install -r DDI_Ben/requirements.txt --quiet

pip install rdkit --quiet
pip install sacremoses --quiet
