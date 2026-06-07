import os
import numpy as np
from models.KGE import *
from models.MLP import *
from models.SSIDDI import *
import json
import sys
import torch

import random
import warnings

import fcntl
import pandas
import pickle as pkl

from collections import defaultdict as ddict

from torch.utils.data import DataLoader 
from torch.utils.data import Dataset
import torch.optim as optim

from rdkit import Chem
from rdkit import RDLogger
import networkx as nx
from torch_geometric.utils import subgraph, degree, get_laplacian
import random
import numpy as np

# Suppress RDKit warnings and deprecation warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='rdkit')


def load_data(args):
    paths = args.paths
    triple_dict = {'train':[], 'valid':[], 'test_S0':[], 'test_S1':[], 'test_S2':[]}
    
    file_mapping = {
        'train': paths['train_file'],
        'valid': paths['valid_file'],
        'test_S0': paths['test_s0_file'],
        'test_S1': paths['test_s1_file'],
        'test_S2': paths['test_s2_file']
    }
    
    for split, filepath in file_mapping.items():
        with open(filepath) as file:
            for j in file:
                str_lin = j.strip().split(' ')
                triple_dict[split].append([int(j) for j in str_lin])
    return triple_dict

def load_feature(args):
    feature_path = args.paths['drugbank_features']
    with open(feature_path, 'rb') as f:
        x = pkl.load(f, encoding='utf-8')
    feat = []
    for y in x['Morgan_Features']:
        feat.append(y)
    return feat

def add_model(args, data_record, device):
    model = 0
    if args.model == 'MSTE':
        model = KGEModel('MSTE', data_record.num_ent, data_record.num_rel, args).to(device)
    elif args.model == 'MLP':
        model = MLP(data_record.num_ent, data_record.num_rel, args.mlp_dim, args, data_record.feat).to(device)
    elif args.model == 'SSI-DDI':
        rel_total = data_record.num_rel  # drugbank only
        model = SSI_DDI(args, 55, 64, 64, rel_total, heads_out_feat_params=[32, 32, 32, 32], blocks_params=[2, 2, 2, 2]).to(device)
    return model

def read_batch(batch, split, device, args, data_record = None):
    if args.model in ['MLP']:
        triple, label = [ _.to(device) for _ in batch]
        return [triple[:, 0], triple[:, 1], triple[:, 2]], label
    elif args.model in ['MSTE']:
        triple, label = [ _.to(device) for _ in batch]
        num_rel = data_record.num_rel
        neg_data = []
        samp_set_0 = [i for i in range(num_rel)]
        for j in triple:
            samp_set = list(set(samp_set_0) - set([j[2].item()]))
            n_neg = 1 if args.model == 'MSTE' else 16
            neg_data.append(random.sample(samp_set, n_neg))
        neg_data = torch.LongTensor(neg_data).to(device)
        return [triple, neg_data, split], label
    elif args.model == 'SSI-DDI':
        label = torch.nn.functional.one_hot(batch[2], num_classes=data_record.num_rel).float()
        return (batch[0].to(device), batch[1].to(device), batch[2].to(device)), label.to(device)
