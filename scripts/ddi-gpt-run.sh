#!/usr/bin/env bash
set -e

cd DDI-Bench/DDI_Ben/DDI-GPT
python main_drugbank.py --split_strategy cluster
