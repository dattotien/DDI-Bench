import os
from torch.utils.data import Dataset, DataLoader
from utils import *
from collections import defaultdict as ddict
import torch
import json

from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA

### SSI-DDI ### 

import itertools
from torch_geometric.data import Data, Batch
from rdkit import Chem
import numpy as np
import math

### SSI-DDI ### 


num_ent = {'drugbank':  1295, 'HetioNet': 34124}
num_rel = {'drugbank': 4}

class Data_record():
    def __init__(self, args):
        self.args = args

        folder_name = args.dataset + '_' + args.dataset_type

        self.link_aug_num = 0

        self.device = "cuda:"+ str(args.gpu) if torch.cuda.is_available() else "cpu"
        self.triplets = load_data(args)
        # self.triplets_all = self.triplets['train'] + self.triplets['valid'] + self.triplets['test']

        self.data = ddict(list)
        sr2o = ddict(set)

        self.link_aug_num = 0

        self.num_rel, self.args.num_rel = num_rel[args.dataset] + self.link_aug_num, num_rel[args.dataset] + self.link_aug_num
        
        self.num_ent, self.args.num_ent = num_ent[args.dataset], num_ent[args.dataset]

        self.include_splits = list(self.triplets.keys())
        self.split_not_train = [j for j in self.include_splits if j != 'train']
        
        for split in self.include_splits:
            for j in self.triplets[split]:
                sub, obj, rel = j[0], j[1], j[2]
                self.data[split].append((sub, obj, rel))

                if split == 'train': 
                    sr2o[(sub, obj)].add(rel)
        
        if args.use_feat:
            self.feat = torch.FloatTensor(np.array(load_feature(args))).to(self.device)
            self.feat_dim = self.feat.shape[1]
        else:
            self.feat = 0
        
        self.sr2o = {k: list(v) for k, v in sr2o.items()}

        self.data = dict(self.data)

        for split in self.split_not_train:
            for sub, obj, rel in self.data[split]:
                sr2o[(sub, obj)].add(rel)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples  = ddict(list)

        ### train triples
        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

        ### valid & test triplets
        for split in self.split_not_train:
            for sub, obj, rel  in self.data[split]:
                self.triples[split].append({'triple': (sub, obj, rel), 	   'label': self.sr2o_all[(sub, obj)]})

        self.triples = dict(self.triples)

        if args.model in ['SSI-DDI', 'SAGAN']:
            smiles_path = args.paths['drugbank_smiles']
            with open(smiles_path, 'r') as file:
                id2smiles = json.load(file)

            drug_id_mol_graph_tup = [Chem.MolFromSmiles(id2smiles[j].strip()) for j in id2smiles] 
            self.ATOM_MAX_NUM = np.max([m.GetNumAtoms() for m in drug_id_mol_graph_tup])
            self.AVAILABLE_ATOM_SYMBOLS = list({a.GetSymbol() for a in itertools.chain.from_iterable(m.GetAtoms() for m in drug_id_mol_graph_tup)})
            self.AVAILABLE_ATOM_DEGREES = list({a.GetDegree() for a in itertools.chain.from_iterable(m.GetAtoms() for m in drug_id_mol_graph_tup)})
            self.AVAILABLE_ATOM_TOTAL_HS = list({a.GetTotalNumHs() for a in itertools.chain.from_iterable(m.GetAtoms() for m in drug_id_mol_graph_tup)})
            max_valence = max(a.GetImplicitValence() for a in itertools.chain.from_iterable(m.GetAtoms() for m in drug_id_mol_graph_tup))
            max_valence = max(max_valence, 9)
            self.AVAILABLE_ATOM_VALENCE = np.arange(max_valence + 1)

            self.MAX_ATOM_FC = abs(np.max([a.GetFormalCharge() for a in itertools.chain.from_iterable(m.GetAtoms() for m in drug_id_mol_graph_tup)]))
            self.MAX_ATOM_FC = self.MAX_ATOM_FC if self.MAX_ATOM_FC else 0
            self.MAX_RADICAL_ELC = abs(np.max([a.GetNumRadicalElectrons() for a in itertools.chain.from_iterable(m.GetAtoms() for m in drug_id_mol_graph_tup)]))
            self.MAX_RADICAL_ELC = self.MAX_RADICAL_ELC if self.MAX_RADICAL_ELC else 0

            self.MOL_EDGE_LIST_FEAT_MTX = [get_mol_edge_list_and_feat_mtx(mol) for mol in drug_id_mol_graph_tup]

            self.TOTAL_ATOM_FEATS = self.MOL_EDGE_LIST_FEAT_MTX[0][1].shape[-1]

        ### the main part
        self.data_iter = {}
        if args.model in ['SSI-DDI', 'SAGAN']:
            if args.dataset == 'drugbank':
                train_dataset = SSIDataset(self.data['train'], self.MOL_EDGE_LIST_FEAT_MTX, args, ratio=1, neg_ent=1)
                self.data_iter['train'] = SSILoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                if args.adversarial:
                    copy_triplets = ((int(len(self.data['train'])/len(self.data['valid'])) + 1) * self.data['valid'])[:int(len(self.data['train']))]
                    train_dataset_adv = SSIDataset(copy_triplets, self.MOL_EDGE_LIST_FEAT_MTX, args, ratio=1, neg_ent=1)
                    self.data_iter['train_adv'] = SSILoader(train_dataset_adv, batch_size=args.batch_size, shuffle=True)
                for j in self.split_not_train:
                    dts = SSIDataset(self.data[j], self.MOL_EDGE_LIST_FEAT_MTX, args, ratio=1, neg_ent=1)
                    self.data_iter[j] = SSILoader(dts, batch_size=args.batch_size, shuffle=False)
            else:
                train_dataset = SSIDataset(self.data['train'], self.MOL_EDGE_LIST_FEAT_MTX, args, ratio=1, neg_ent=1)
                self.data_iter['train'] = SSILoader(train_dataset, batch_size=args.batch_size, shuffle=True)
                if args.adversarial:
                    copy_triplets = ((int(len(self.data['train'])/len(self.data['valid'])) + 1) * self.data['valid'])[:int(len(self.data['train']))]
                    train_dataset_adv = SSIDataset(copy_triplets, self.MOL_EDGE_LIST_FEAT_MTX, args, ratio=1, neg_ent=1)
                    self.data_iter['train_adv'] = SSILoader(train_dataset_adv, batch_size=args.batch_size, shuffle=True)
                for j in self.split_not_train:
                    dts = SSIDataset(self.data[j], self.MOL_EDGE_LIST_FEAT_MTX, args, ratio=1, neg_ent=1)
                    self.data_iter[j] = SSILoader(dts, batch_size=args.batch_size, shuffle=False)
        else:
            # MSTE model uses default TrainDataset and TestDataset
            self.data_iter['train'] = self.get_data_loader(TrainDataset, 'train', args.batch_size)
            for j in self.split_not_train:
                self.data_iter[j] = self.get_data_loader(TestDataset, j, args.batch_size, shuffle = False)

    def get_atom_features(self, atom, mode='one_hot'): ### data process for SSI-DDI

        if mode == 'one_hot':
            atom_feature = torch.cat([
                one_of_k_encoding_unk(atom.GetSymbol(), self.AVAILABLE_ATOM_SYMBOLS),
                one_of_k_encoding_unk(atom.GetDegree(), self.AVAILABLE_ATOM_DEGREES),
                one_of_k_encoding_unk(atom.GetTotalNumHs(), self.AVAILABLE_ATOM_TOTAL_HS),
                one_of_k_encoding_unk(atom.GetImplicitValence(), self.AVAILABLE_ATOM_VALENCE),
                torch.tensor([atom.GetIsAromatic()], dtype=torch.float)
            ])
        else:
            atom_feature = torch.cat([
                one_of_k_encoding_unk(atom.GetSymbol(), self.AVAILABLE_ATOM_SYMBOLS),
                torch.tensor([atom.GetDegree()]).float(),
                torch.tensor([atom.GetTotalNumHs()]).float(),
                torch.tensor([atom.GetImplicitValence()]).float(),
                torch.tensor([atom.GetIsAromatic()]).float()
            ])

        return atom_feature


    def get_data_loader(self, dataset_class, split, batch_size, shuffle=True):
        return  DataLoader(
            dataset_class(self.triples[split], self.args),
            batch_size      = batch_size,
            shuffle         = shuffle,
            num_workers     = 10, ### set the default numworkers to 10
            collate_fn      = dataset_class.collate_fn,
            drop_last=True
        )

class TrainDataset(Dataset):

	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params
		# self.entities	= np.arange(self.p.num_ent, dtype=np.int32)

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele	= self.triples[idx]
		if 'sub_samp' in ele:
			triple, label, sub_samp	= torch.LongTensor(ele['triple']), np.array(ele['label']).astype(int), np.float32(ele['sub_samp'])
		else:
			triple, label = torch.LongTensor([ele['triple'][0], ele['triple'][1], -1]), np.array(ele['label']).astype(int) 
		trp_label = self.get_label_ddi(label)
        
		if self.p.model in ['MSTE']:
			triple = torch.LongTensor([ele['triple'][0], ele['triple'][1], ele['label'][0]])

		if self.p.lbl_smooth != 0.0:
			trp_label = (1.0 - self.p.lbl_smooth)*trp_label + (1.0/self.p.num_ent)

		return triple, trp_label

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		trp_label	= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, trp_label
	
	def get_label_ddi(self, label):
		y = np.zeros([self.p.num_rel], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)

class TestDataset(Dataset):

	def __init__(self, triples, params):
		self.triples	= triples
		self.p 		= params

	def __len__(self):
		return len(self.triples)

	def __getitem__(self, idx):
		ele		= self.triples[idx]
		triple, label	= torch.LongTensor(ele['triple']), np.array(ele['label']).astype(int)
		label		= self.get_label_ddi(label)

		return triple, label

	@staticmethod
	def collate_fn(data):
		triple		= torch.stack([_[0] 	for _ in data], dim=0)
		label		= torch.stack([_[1] 	for _ in data], dim=0)
		return triple, label

	def get_label_ddi(self, label):
		y = np.zeros([self.p.num_rel], dtype=np.float32)
		for e2 in label: y[e2] = 1.0
		return torch.FloatTensor(y)

### dataset for tiger

class DTADataset(InMemoryDataset):
    def __init__(self, x=None, y=None, sub_graph=None, smile_graph=None, dt = None):
        super(DTADataset, self).__init__()

        self.labels = y
        self.drug_ID = x
        self.sub_graph = sub_graph
        self.smile_graph = smile_graph
        self.dt = dt

    def read_drug_info(self, drug_id):

        c_size, features, edge_index, rel_index, sp_edge_index, sp_value, sp_rel, deg = self.smile_graph[str(drug_id)]  ##drug——id是str类型的，不是int型的，这点要注意
        subset, subgraph_edge_index, subgraph_rel, mapping_id, s_edge_index, s_value, s_rel, deg = self.sub_graph[str(drug_id)]

        if edge_index == 0:
            c_size = 1
            features = [[0 for j in range(67)]]
            edge_index = [[0, 0]]
            rel_index = [0]
            sp_edge_index = [[0, 0]]
            sp_value = [1]
            sp_rel = [1]

        data_mol = DATA.Data(x=torch.Tensor(np.array(features)),
                              edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                            #   y=torch.LongTensor([labels]),
                              rel_index=torch.Tensor(np.array(rel_index, dtype=int)),
                              sp_edge_index=torch.LongTensor(sp_edge_index).transpose(1, 0),
                              sp_value=torch.Tensor(np.array(sp_value, dtype=int)),
                              sp_edge_rel=torch.LongTensor(np.array(sp_rel, dtype=int))
                              )
        data_mol.__setitem__('c_size', torch.LongTensor([c_size]))

        data_graph = DATA.Data(x=torch.LongTensor(subset),
                                edge_index=torch.LongTensor(subgraph_edge_index).transpose(1,0),
                                # y=torch.LongTensor([labels]),
                                id=torch.LongTensor(np.array(mapping_id, dtype=bool)),
                                rel_index=torch.Tensor(np.array(subgraph_rel, dtype=int)),
                                sp_edge_index=torch.LongTensor(s_edge_index).transpose(1, 0),
                                sp_value=torch.Tensor(np.array(s_value, dtype=int)),
                                sp_edge_rel=torch.LongTensor(np.array(s_rel, dtype=int))
                                )

        return data_mol, data_graph

    def __len__(self):
        #self.data_mol1, self.data_drug1, self.data_mol2, self.data_drug2
        return len(self.drug_ID)

    def __getitem__(self, idx):
        drug1_id = self.drug_ID[idx, 0]
        drug2_id = self.drug_ID[idx, 1]
        # labels = int(self.labels[idx])
        if self.dt == 'drugbank':
            labels = torch.LongTensor([self.labels[idx]])
        else:
            labels = torch.FloatTensor(self.labels[idx])

        drug1_mol, drug1_subgraph = self.read_drug_info(drug1_id)
        drug2_mol, drug2_subrgraph = self.read_drug_info(drug2_id)

        return drug1_mol, drug1_subgraph, drug2_mol, drug2_subrgraph, labels


def collate(data_list):
    batchA = Batch.from_data_list([data[0] for data in data_list])
    batchB = Batch.from_data_list([data[1] for data in data_list])
    batchC = Batch.from_data_list([data[2] for data in data_list])
    batchD = Batch.from_data_list([data[3] for data in data_list])
    batchE = torch.stack([data[4] for data in data_list]).squeeze(1)

    return batchA, batchB, batchC, batchD, batchE

### Dataset for SSI-DDI

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom,
                explicit_H=True,
                use_chirality=False):

    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C','N','O', 'S','F','Si','P', 'Cl','Br','Mg','Na','Ca','Fe','As','Al','I','B','V','K','Tl',
            'Yb','Sb','Sn','Ag','Pd','Co','Se','Ti','Zn','H', 'Li','Ge','Cu','Au','Ni','Cd','In',
            'Mn','Zr','Cr','Pt','Hg','Pb','Unknown'
        ]) + [atom.GetDegree()/10, atom.GetImplicitValence(), 
                atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
                ]) + [atom.GetIsAromatic()]
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if explicit_H:
        results = results + [atom.GetTotalNumHs()]

    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    results = np.array(results).astype(np.float32)

    return torch.from_numpy(results)


def get_mol_edge_list_and_feat_mtx(mol_graph):
    features = [(atom.GetIdx(), atom_features(atom)) for atom in mol_graph.GetAtoms()]
    features.sort() # to make sure that the feature matrix is aligned according to the idx of the atom
    _, features = zip(*features)
    features = torch.stack(features)

    edge_list = torch.LongTensor([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol_graph.GetBonds()])
    undirected_edge_list = torch.cat([edge_list, edge_list[:, [1, 0]]], dim=0) if len(edge_list) else edge_list
    
    return undirected_edge_list.T, features


class SSIDataset(Dataset):
    def __init__(self, tri_list, MOL_EDGE_LIST_FEAT_MTX, args, ratio=1.0,  neg_ent=1, disjoint_split=True, shuffle=True):
        ''''disjoint_split: Consider whether entities should appear in one and only one split of the dataset
        ''' 
        # self.neg_ent = neg_ent
        self.tri_list = []
        # self.ratio = ratio
        self.MOL_EDGE_LIST_FEAT_MTX = MOL_EDGE_LIST_FEAT_MTX

        for h, t, r, *_ in tri_list:
            self.tri_list.append((h, t, r))

        if shuffle:
            random.shuffle(self.tri_list)
        limit = math.ceil(len(self.tri_list) * ratio)
        self.tri_list = self.tri_list[:limit]

    def __len__(self):
        return len(self.tri_list)
    
    def __getitem__(self, index):
        return self.tri_list[index]

    def collate_fn(self, batch):

        pos_rels = []
        pos_h_samples = []
        pos_t_samples = []

        for h, t, r in batch:
            pos_rels.append(r)
            h_data = self.__create_graph_data(h)
            t_data = self.__create_graph_data(t)
            pos_h_samples.append(h_data)
            pos_t_samples.append(t_data)

        pos_h_samples = Batch.from_data_list(pos_h_samples)
        pos_t_samples = Batch.from_data_list(pos_t_samples)
        pos_rels = torch.LongTensor(pos_rels)
        pos_tri = (pos_h_samples, pos_t_samples, pos_rels)

        return pos_tri

    def __create_graph_data(self, id):
        edge_index = self.MOL_EDGE_LIST_FEAT_MTX[id][0]
        features = self.MOL_EDGE_LIST_FEAT_MTX[id][1]

        return Data(x=features, edge_index=edge_index)

class SSILoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)
