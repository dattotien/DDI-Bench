import os
import argparse
import torch
import random
from load_data import DataLoader
import yaml
from types import SimpleNamespace
from base_model import BaseModel
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial
import time
import wandb

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'



class options:
    def __init__():
        pass
def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return SimpleNamespace(**config)

if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--config', default='config/config.yaml',
                      help='path to config yaml (use a job-specific file when running parallel sessions '
                           'so concurrent processes never read/write the same config)')
    cli.add_argument('--wandb-entity', default=None, help='wandb entity; defaults to your account default')
    cli.add_argument('--wandb-project', default='EmerGNN_DrugBank')
    cli.add_argument('--wandb-name', default=None,
                      help='base run name; the dataset is appended to it. '
                           'If unset, defaults to "<dataset>_seed<seed>_gpu<gpu>"')
    cli_args = cli.parse_args()

    args = load_config(cli_args.config)
    torch.cuda.set_device(args.gpu)
    dataloader = DataLoader(args)
    eval_ent, eval_rel = dataloader.eval_ent, dataloader.eval_rel
    args.all_ent, args.all_rel, args.eval_rel = dataloader.all_ent, dataloader.all_rel, dataloader.eval_rel
    
    # Load label_mapping từ config
    label_mapping = args.label_mappings
    
    KG = dataloader.KG
    vKG = dataloader.vKG
    triplets = dataloader.triplets
    train_pos, train_neg = torch.LongTensor(triplets['train']).cuda(), None
    valid_pos, valid_neg = torch.LongTensor(triplets['valid']).cuda(), None
    
    # Lấy 3 test sets từ dataloader
    test_sets = {}
    for name in ['S0', 'S1', 'S2']:
        test_sets[name] = {
            'pos': torch.LongTensor(dataloader.test_triplets[name]).cuda(),
            'neg': None,
            'KG': dataloader.test_KGs[name]
        }
    if args.adversarial:
        tmp = args.dataset
        args.dataset = list(args.dataset)
        args.dataset[1] = '1' # use S1 dataset
        args.dataset = ''.join(args.dataset)
        dataloader1 = DataLoader(args)
        triplets1 = dataloader1.triplets
        valid1_pos, valid1_neg = torch.LongTensor(triplets1['valid']).cuda(), None
        # Sử dụng S1 test set cho adversarial training
        test1_pos, test1_neg = torch.LongTensor(dataloader1.test_triplets['S1']).cuda(), None
        train1_pos = torch.cat([valid1_pos, test1_pos], dim=0).cuda()
        train1_neg = None
        args.dataset = tmp

    os.makedirs('results', exist_ok=True)  # exist_ok avoids a TOCTOU race when parallel sessions start at once

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
        
        # --wandb-name is a fixed base name (e.g. "MultiTask_DB_aw_w/o_description_based");
        # the dataset is always appended so parallel sessions across datasets
        # stay distinguishable on the dashboard. Left unset, fall back to the
        # old auto-generated name.
        run_name = f'{cli_args.wandb_name}_{args.dataset}' if cli_args.wandb_name \
            else f'{args.dataset}_seed{seed}_gpu{args.gpu}'

        wandb.init(
            entity=cli_args.wandb_entity,
            project=cli_args.wandb_project,
            name=run_name,
            config=vars(args),
        )
        model = BaseModel(eval_ent, eval_rel, args, label_mapping=label_mapping)
        best_acc = -1
        best_str = ''
        best_str_class = ''
        for e in range(args.n_epoch):
            dataloader.shuffle_train()
            KG = dataloader.KG
            train_pos = torch.LongTensor(dataloader.train_data).cuda()
            if args.adversarial:
                model.train(train_pos, None, train1_pos, None, KG, epoch=e+1)
            else:
                model.train(train_pos, None, None, None, KG, epoch=e+1)
            if (e+1) % args.epoch_per_test == 0:
                v_f1, v_acc, v_micro, v_results = model.evaluate(valid_pos, valid_neg, vKG, is_test=False, epoch=e+1)
                
                # Log validation metrics
                log_dict = {
                    "epoch": e+1,
                    "val/macro_f1": v_f1,
                    "val/accuracy": v_acc,
                    "val/micro_f1": v_micro,
                    "learning_rate": model.optimizer.param_groups[0]['lr']
                }
                
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
                    
                    # Test trên cả 3 test sets
                    test_results = {}
                    for name, test_data in test_sets.items():
                        t_f1, t_acc, t_micro, t_res = model.evaluate(
                            test_data['pos'], test_data['neg'], test_data['KG'], 
                            is_test=False, epoch=e+1, prefix=f"test_{name}"
                        )
                        test_results[name] = {
                            'macro_f1': t_f1,
                            'accuracy': t_acc,
                            'micro_f1': t_micro,
                            'details': t_res
                        }
                        
                        # Thêm vào log dict
                        log_dict[f"test_{name}/macro_f1"] = t_f1
                        log_dict[f"test_{name}/accuracy"] = t_acc
                        log_dict[f"test_{name}/micro_f1"] = t_micro
                    
                    # Tạo output string cho cả 3 test sets
                    test_str = ''
                    for name in ['S0', 'S1', 'S2']:
                        res = test_results[name]
                        test_str += f'\t[Test {name}] macro_f1:{res["macro_f1"]:.4f} acc:{res["accuracy"]:.4f} micro_f1:{res["micro_f1"]:.4f}'
                    
                    out_str += test_str
                    best_str = out_str
                    
                    # Lưu chi tiết kết quả
                    out_str_class = ''
                    for name in ['S0', 'S1', 'S2']:
                        out_str_class += f'[Test {name} per class]: {test_results[name]["details"]} \n'
                    best_str_class = out_str_class
                    
                    if args.save_model:
                        model.save_model(best_str)
                    
                    print(out_str)
                    with open(os.path.join('results', args.dataset+'_'+str(seed)+'_eval'+('_adv' if args.adversarial else '')+'.txt'), 'a+') as f:
                        f.write(out_str + '\n')
                        f.write(out_str_class + '\n')
                else:
                    print(out_str + ' (No improvement, skip testing)')
                
                # Log to wandb
                if wandb.run is not None:
                    wandb.log(log_dict, step=e+1)
        print('Best results:\t' + best_str)
        with open(os.path.join('results', args.dataset+'_'+str(seed)+'_eval'+('_adv' if args.adversarial else '')+'.txt'), 'a+') as f:
            f.write('Best results:\t' + best_str + '\n\n')
            f.write(best_str_class + '\n')
        wandb.finish()
        return -best_acc

    run_model(args.seed)
    

