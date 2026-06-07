import os
import setproctitle
import argparse
import yaml
from trainer import Trainer
from utils import *
import torch
import numpy as np
import wandb
import warnings

from kaggle_secrets import UserSecretsClient
print('pid:', os.getpid())

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file and convert to Namespace"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    ### set process name
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Suppress RDKit warnings
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    
    setproctitle.setproctitle('BNbench')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    ### set hyperparameters
    parser = argparse.ArgumentParser(description='Task Aware Relation Graph for Few-shot Chemical Property Prediction')
    # general hyperparameters
    parser.add_argument('--model', type=str, default='MRCGNN', choices=['MSTE', 'MLP', 'Decagon', 'TIGER', 'SSI-DDI', 'MRCGNN', 'SAGAN'])
    parser.add_argument('--name', default='testrun', help='Set run name for saving/restoring models')

    ### dataset setting
    parser.add_argument('--dataset', type=str, default='drugbank', choices=['drugbank', 'twosides'])
    parser.add_argument('--dataset_type', type=str, default='cluster', choices=['random', 'cluster']) ### exchange random and sail

    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
    parser.add_argument('--lbl_smooth',	type=float,     default=0.0,	help='Label Smoothing') ### usually 0-1
    parser.add_argument("--epoch", type=int, default=1, help="training epoch")
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--use_feat', default=1, type=bool, help='Whether to use drug feature')

    parser.add_argument('--seed', default=124, type=int, help='Seed for randomization')
    parser.add_argument('--eval_skip', default=1, type=int, help='Evaluate every x epochs')
    parser.add_argument('--patience', default=10, type=int, help='Patience for early stopping')
    
    ### Convert config dict to Namespace
    args = argparse.Namespace(**config)

    ### set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model in ['MSTE']:
        args.use_feat = 0

    args.device = "cuda:"+ str(args.gpu) if torch.cuda.is_available() else "cpu"
    try:
        user_secrets = UserSecretsClient()
        my_secret = user_secrets.get_secret("wandb_key") 
        wandb.login(key=my_secret)
    except:
        wandb.login(key="c4816b32f37419d7d62dc261260293cdfb9d7190")
    wandb.init(
        entity="tunglamngo-univesity-of-engineering-and-technology-vnu",
        project="DDI_NCKH_2025",
        name=args.name,
        config=vars(args)
    )
    ### Training step in the trainer
    trainer = Trainer(args)
    trainer.run()
    wandb.finish()

if __name__ == "__main__":
    main()
