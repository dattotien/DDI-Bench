import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import argparse
import torch
import random
from load_data import DataLoader
from emergnn_fact_splitter import EmerGNNFactSplitter

from base_model import BaseModel
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial
import time
import wandb

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser(description="Parser for EmerGNN")
parser.add_argument('--task_dir', type=str, default='./', help='the directory to dataset')
parser.add_argument('--dataset', type=str, default='S1_1', help='the directory to dataset')
parser.add_argument('--lamb', type=float, default=7e-4, help='set weight decay value')
parser.add_argument('--gpu', type=int, default=0, help='GPU id to load.')
parser.add_argument('--n_dim', type=int, default=128, help='set embedding dimension')
parser.add_argument('--lr', type=float, default=0.03, help='set learning rate')
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--load_model', action='store_true')
parser.add_argument('--n_epoch', type=int, default=100, help='number of training epochs')
parser.add_argument('--n_batch', type=int, default=512, help='batch size')
parser.add_argument('--epoch_per_test', type=int, default=5, help='frequency of testing')
parser.add_argument('--test_batch_size', type=int, default=16, help='test batch size')
parser.add_argument('--out_file_info', type=str, default='', help='extra string for the output file name')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--adversarial', action='store_true')
parser.add_argument('--adversarial_weight', type=float, default=1, help='the weight of adversarial loss in the total loss.')
parser.add_argument('--use_pair_kg', action='store_true', help='Use pair-specific KGs instead of a global KG')
parser.add_argument('--train_kg_npz', type=str, default='', help='Path to train pair-specific KGs .npz file')
parser.add_argument('--valid_kg_npz', type=str, default='', help='Path to valid pair-specific KGs .npz file')
parser.add_argument('--test_kg_npz', type=str, default='', help='Path to test pair-specific KGs .npz file')
parser.add_argument('--use_dynamic_subgraph_sampling', action='store_true', help='Use dynamic subgraph sampling')
parser.add_argument('--splitter_ratio', type=float, default=0.8, help='Splitter ratio for facts/labels')
parser.add_argument('--splitter_scenario', type=str, default='S1', help='Scenario for the fact splitter (S0/S1/S2)')

class options:
    def __init__():
        pass


if __name__ == '__main__':
    args = parser.parse_args()
    if hasattr(args, 'use_dynamic_subgraph_sampling') and args.use_dynamic_subgraph_sampling:
        args.use_pair_kg = True
    torch.cuda.set_device(args.gpu)
    dataloader = DataLoader(args)
    eval_ent, eval_rel = dataloader.eval_ent, dataloader.eval_rel
    args.all_ent, args.all_rel, args.eval_rel = dataloader.all_ent, dataloader.all_rel, dataloader.eval_rel
    
    if hasattr(args, 'use_pair_kg') and args.use_pair_kg:
        KG = None
        vKG = None
        tKG = None
    else:
        KG = dataloader.KG
        vKG = dataloader.vKG
        tKG = dataloader.tKG
        
    triplets = dataloader.triplets
    train_pos, train_neg = torch.LongTensor(triplets['train']).cuda(), None
    valid_pos, valid_neg = torch.LongTensor(triplets['valid']).cuda(), None
    test_pos,  test_neg  = torch.LongTensor(triplets['test']).cuda(), None
    if args.adversarial:
        tmp = args.dataset
        args.dataset = list(args.dataset)
        args.dataset[1] = '1' # use S1 dataset
        args.dataset = ''.join(args.dataset)
        dataloader1 = DataLoader(args)
        triplets1 = dataloader1.triplets
        valid1_pos, valid1_neg = torch.LongTensor(triplets1['valid']).cuda(), None
        test1_pos, test1_neg = torch.LongTensor(triplets1['test']).cuda(), None
        train1_pos = torch.cat([valid1_pos, test1_pos], dim=0).cuda()
        train1_neg = None
        args.dataset = tmp

    if not os.path.exists('results'):
        os.makedirs('results')

    def run_model(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if args.dataset.startswith('S1'):
            args.lr = 0.003000
            args.lamb = 0.00000001
            args.weight = 0.
            args.length = 3
            args.n_batch = 32
            args.n_dim = 64
            args.feat = 'M'

        elif args.dataset.startswith('S2'):
            args.lr = 0.003000
            args.lamb = 0.00010000
            args.weight = 0.
            args.length = 3
            args.n_batch = 32
            args.n_dim = 32
            args.feat = 'M'
            
        elif args.dataset.startswith('S0'):
            args.lr = 0.003000
            args.lamb = 0.00000001
            args.weight = 0
            args.length = 3
            args.n_batch = 64
            args.n_dim = 32
            args.feat = 'E'
        
        wandb.init(project='EmerGNN_DrugBank', config=vars(args))
        model = BaseModel(eval_ent, eval_rel, args)
        
        if hasattr(args, 'use_dynamic_subgraph_sampling') and args.use_dynamic_subgraph_sampling:
            train_csv_path = os.path.join(dataloader.task_dir, 'data/{}/{}_ddi.txt'.format(args.dataset, 'train'))
            node2id_path = os.path.join(dataloader.task_dir, 'data/node2id.json')
            splitter = EmerGNNFactSplitter(
                train_csv_path=train_csv_path,
                node2id_path=node2id_path,
                kg_triplets=dataloader.train_kg,
                scenario=args.splitter_scenario,
                ratio=args.splitter_ratio,
                seed=args.seed
            )
            
        best_acc = -1
        for e in range(args.n_epoch):
            if hasattr(args, 'use_dynamic_subgraph_sampling') and args.use_dynamic_subgraph_sampling:
                fact_triplets, label_indices = splitter.shuffle(e)
                epoch_train_triplets = splitter.train_triplets[label_indices]
                from load_data import DynamicPairKGs
                dataloader.train_pair_kgs = DynamicPairKGs(
                    epoch_train_triplets, dataloader, args.length,
                    base_kg_triplets=dataloader.train_kg,
                    ddi_triplets=fact_triplets
                )
                dataloader.train_data = epoch_train_triplets
                dataloader.shuffle_train_pair_kg()
                train_pos = torch.LongTensor(dataloader.train_data).cuda()
                model.train(train_pos, None, None, None, dataloader.train_pair_kgs)
            elif hasattr(args, 'use_pair_kg') and args.use_pair_kg:
                dataloader.shuffle_train_pair_kg()
                train_pos = torch.LongTensor(dataloader.train_data).cuda()
                model.train(train_pos, None, None, None, dataloader.train_pair_kgs)
            else:
                dataloader.shuffle_train()
                KG = dataloader.KG
                train_pos = torch.LongTensor(dataloader.train_data).cuda()
                if args.adversarial:
                    model.train(train_pos, None, train1_pos, None, KG)
                else:
                    model.train(train_pos, None, None, None, KG)
            if (e+1) % args.epoch_per_test == 0:
                if hasattr(args, 'use_pair_kg') and args.use_pair_kg:
                    v_f1, v_acc, v_kap, _ = model.evaluate(valid_pos, valid_neg, dataloader.valid_pair_kgs)
                    t_f1, t_acc, t_kap, t_per_class = model.evaluate(test_pos,  test_neg,  dataloader.test_pair_kgs)
                else:
                    v_f1, v_acc, v_kap, _ = model.evaluate(valid_pos, valid_neg, vKG)
                    t_f1, t_acc, t_kap, t_per_class = model.evaluate(test_pos,  test_neg,  tKG)
                # v_f1, v_acc, v_kap = model.evaluate(valid_pos, valid_neg, vKG)
                # t_f1, t_acc, t_kap = model.evaluate(test_pos,  test_neg,  tKG)
                model.scheduler.step(v_f1)
                if args.adversarial:
                    model.scheduler_ad.step(v_f1)
                time_now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                
                # Log metrics to wandb
                wandb.log({
                    "epoch": e + 1,
                    "valid/f1": v_f1,
                    "valid/accuracy": v_acc,
                    "valid/kappa": v_kap,
                    "test/f1": t_f1,
                    "test/accuracy": t_acc,
                    "test/kappa": t_kap,
                })
                
                out_str = time_now + ' :epoch:%d\tfeat:%s lr:%.6f lamb:%.8f n_batch:%d n_dim:%d layer:%d\t[Valid] f1:%.4f acc:%.4f kap:%.4f\t[Test] f1:%.4f acc:%.4f kap:%.4f' % (e+1, args.feat, args.lr, args.lamb, args.n_batch, args.n_dim, args.length, v_f1, v_acc, v_kap, t_f1, t_acc, t_kap)
                out_str_class = f'[Test per class]: {t_per_class} \n'
                if v_f1 > best_acc:
                    best_acc = v_f1
                    best_str = out_str
                    best_str_class = out_str_class
                    if args.save_model:
                        model.save_model(best_str)
                print(out_str)
                with open(os.path.join('results', args.dataset+'_'+str(seed)+'_eval'+('_adv' if args.adversarial else '')+'.txt'), 'a+') as f:
                    f.write(out_str + '\n')
                    f.write(out_str_class + '\n')
        print('Best results:\t' + best_str)
        with open(os.path.join('results', args.dataset+'_'+str(seed)+'_eval'+('_adv' if args.adversarial else '')+'.txt'), 'a+') as f:
            f.write('Best results:\t' + best_str + '\n\n')
            f.write(best_str_class + '\n')
        wandb.finish()
        return -best_acc

    run_model(args.seed)
    

