import os
import setproctitle
import argparse
import yaml
from trainer import Trainer
from utils import *
import dataset_registry
import torch
import numpy as np
import wandb
import warnings

try: ### chi co tren Kaggle; ngoai Kaggle van chay duoc voi wandb key trong config/env
    from kaggle_secrets import UserSecretsClient
except ImportError:
    UserSecretsClient = None
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
    parser.add_argument('--model', type=str, default='MRCGNN', choices=['MSTE', 'MLP', 'Decagon', 'TIGER', 'SSI-DDI', 'SA-DDI', 'MRCGNN', 'SAGAN'])
    parser.add_argument('--name', default='testrun', help='Set run name for saving/restoring models')

    ### dataset setting
    parser.add_argument('--dataset', type=str, default='drugbank', choices=dataset_registry.DATASET_NAMES)
    parser.add_argument('--dataset_type', type=str, default='cluster', choices=['random', 'cluster']) ### exchange random and sail

    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0003, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay")
    parser.add_argument('--lbl_smooth',	type=float,     default=0.0,	help='Label Smoothing') ### usually 0-1
    parser.add_argument("--epoch", type=int, default=1, help="training epoch")
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
    parser.add_argument('--num_workers', default=None, type=int, help='DataLoader workers; default 10 on Linux, 0 on Windows (spawn makes workers re-import torch per loader)')
    ### type=int, khong phai bool: argparse chay bool('0') -> True nen `--use_feat 0`
    ### truoc day khong tat duoc feature
    parser.add_argument('--use_feat', default=1, type=int, choices=[0, 1], help='Whether to use drug feature')

    parser.add_argument('--seed', default=124, type=int, help='Seed for randomization')
    parser.add_argument('--eval_skip', default=1, type=int, help='Evaluate every x epochs')
    parser.add_argument('--patience', default=10, type=int, help='Patience for early stopping')

    # KGE models
    parser.add_argument('--kge_dim', type=int, default=200, help='hidden dimension.')
    parser.add_argument('--kge_gamma', type=int, default=1, help='gamma parameter.')
    parser.add_argument('--kge_dropout', type=float, default=0, help='dropout rate.') ### DDI best 0
    parser.add_argument('--kge_loss', type=str, default='BCE_mean',  help='loss function')

    # MLP model
    parser.add_argument('--mlp_dropout', type=float, default=0.1, help='dropout rate.')
    parser.add_argument('--mlp_dim', type=int, default=200, help='hidden dimension.')

    ### SA-DDI model
    parser.add_argument('--saddi_dim', type=int, default=64, help='hidden dimension.')
    parser.add_argument('--saddi_n_iter', type=int, default=10, help='number of D-MPNN iterations, i.e. the largest substructure radius.')

    ### Decagon model decagon_drop
    parser.add_argument('--decagon_dim', type=int, default=200, help='hidden dimension.')
    parser.add_argument('--decagon_drop', type=float,   default=0.1, help='Dropout to use in Decagon model')

    ### Load config.yaml và set làm default cho argparse
    _cfg_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    cfg = load_config(_cfg_path)
    # Chỉ lấy các key flat (bỏ nested dict như paths, wandb)
    _flat = {k: v for k, v in cfg.items() if not isinstance(v, dict)}
    parser.set_defaults(**_flat)

    ### set basic configurations (CLI args vẫn override yaml)
    args = parser.parse_args()

    ### Gắn các section nested vào args. Không còn label_mappings ở đây: nhãn là
    ### thuộc tính của từng dataset nên nằm trong dataset_registry.py
    args.paths          = cfg.get('paths', {})
    args.wandb_cfg      = cfg.get('wandb', {})

    if args.num_workers is None:
        args.num_workers = 0 if os.name == 'nt' else 10

    ### resolve dataset config (num_ent / num_rel / task / paths) -- see dataset_registry.py
    dataset_registry.apply_to_args(args)
    print('dataset: {} | task: {} | num_ent: {} | num_rel: {} | data: {}'.format(
        args.dataset, args.task, args.num_ent, args.num_rel, dataset_registry.data_dir(args)))

    ### set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model in ['MSTE']:
        args.use_feat = 0
        if dataset_registry.is_multilabel(args):
            args.batch_size = 128

    if args.model == 'SAGAN':
        args.adversarial = 1
    else:
        args.adversarial = 0

    args.device = "cuda:"+ str(args.gpu) if torch.cuda.is_available() else "cpu"
    try:
        user_secrets = UserSecretsClient()
        my_secret = user_secrets.get_secret("wandb_key")
        wandb.login(key=my_secret)
    except Exception:
        ### ngoai Kaggle: uu tien WANDB_API_KEY tu env (.env), roi moi den config.yaml.
        ### Khong co key nao thi de wandb tu doc ~/.netrc (sau khi 'wandb login')
        wandb_key = os.environ.get('WANDB_API_KEY') or args.wandb_cfg.get('wandb_key', '')
        if wandb_key:
            wandb.login(key=wandb_key)
        else:
            wandb.login()
    wandb.init(
        entity=args.wandb_cfg.get('entity') or None,
        project=args.wandb_cfg.get('project', 'DDI'),
        name=args.name,
        config=vars(args)
    )
    ### Training step in the trainer
    trainer = Trainer(args)
    trainer.run()
    wandb.finish()

if __name__ == "__main__":
    main()
