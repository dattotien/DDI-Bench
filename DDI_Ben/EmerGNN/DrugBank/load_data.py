import os
import torch
import random
import numpy as np
from collections import defaultdict
import json

class DataLoader:
    def __init__(self, params, saved_relation2id=None):
        self.task_dir = params.task_dir
        self.dataset = params.dataset

        ddi_paths = {
            'train': os.path.join(self.task_dir, 'data/{}/{}_ddi.txt'.format(params.dataset, 'train')),
            'valid': os.path.join(self.task_dir, 'data/{}/{}_ddi.txt'.format(params.dataset, 'valid')),
            'test':  os.path.join(self.task_dir, 'data/{}/{}_ddi.txt'.format(params.dataset, 'test'))
        }

        kg_paths = {
            'train': os.path.join(self.task_dir, 'data/KG.txt'),
            'valid': os.path.join(self.task_dir, 'data/KG.txt'),
            'test':  os.path.join(self.task_dir, 'data/KG.txt')
        }
        
        self.process_files_ddi(ddi_paths, saved_relation2id)

        if hasattr(params, 'use_pair_kg') and params.use_pair_kg:
            self.use_pair_kg = True
            self.all_ent = self.eval_ent
            self.all_rel = self.eval_rel
            self.load_ent_id()
                
            self.triplets['train'], self.train_pair_kgs = self.load_pair_kgs_from_npz(params.train_kg_npz)
            self.triplets['valid'], self.valid_pair_kgs = self.load_pair_kgs_from_npz(params.valid_kg_npz)
            self.triplets['test'], self.test_pair_kgs = self.load_pair_kgs_from_npz(params.test_kg_npz)
            
            self.train_data = self.triplets['train']
        else:
            self.process_files_kg(kg_paths, saved_relation2id)
            self.load_ent_id()
            self.use_pair_kg = False
            self.shuffle_train()
            self.vKG = self.load_graph(np.concatenate([self.triplets['train'], self.valid_kg], axis=0))
            self.tKG = self.load_graph(np.concatenate([self.triplets['train'], self.triplets['valid'], self.test_kg], axis=0))

    def process_files_ddi(self, file_paths, saved_relation2id=None):
        entity2id = {}
        relation2id = {} if saved_relation2id is None else saved_relation2id

        self.triplets = {}
        self.train_ent = set()

        for file_type, file_path in file_paths.items():
            data = []
            with open(file_path)as f:
                file_data = [line.split() for line in f.read().split('\n')[:-1]]

            for triplet in file_data:
                h, t, r = int(triplet[0]), int(triplet[1]), int(triplet[2])
                if h not in entity2id:
                    entity2id[h] = h
                if t not in entity2id:
                    entity2id[t] = t
                if not saved_relation2id and r not in relation2id:
                    relation2id[r] = r

                if file_type == 'train':
                    self.train_ent.add(h)
                    self.train_ent.add(t)

                data.append([h, t, r])

            self.triplets[file_type] = np.array(data, dtype='int')

        self.entity2id = entity2id
        self.relation2id = relation2id

        # self.eval_ent = max(self.entity2id.keys()) + 1
        self.eval_ent = 1710
        #self.eval_rel = len(self.relation2id)
        self.eval_rel = 86

    def load_ent_id(self, ):
        id2entity = dict()
        id2relation = dict()
        drug_set = json.load(open(os.path.join(self.task_dir, 'data/node2id.json'), 'r'))
        entity_set = json.load(open(os.path.join(self.task_dir, 'data/entity_drug.json'), 'r'))
        relation_set = json.load(open(os.path.join(self.task_dir, 'data/relation2id.json'), 'r'))
        for drug in drug_set:
            id2entity[int(drug_set[drug])] = drug
        for ent in entity_set:
            id2entity[int(entity_set[ent])] = ent

        for rel in relation_set:
            id2relation[int(rel)] = relation_set[rel]
        
        self.id2entity = id2entity
        self.id2relation = id2relation


    def process_files_kg(self, kg_paths, saved_relation2id=None, ratio=1):
        self.kg_triplets = defaultdict(list)
        self.ddi_in_kg = set()
        print('pruned ratio of edges in KG: {}'.format(ratio))

        for file_type, file_path in kg_paths.items():
            with open(file_path) as f:
                file_data = [line.split() for line in f.read().split('\n')[:-1]]

                for triplet in file_data:
                    h, t, r = int(triplet[0]), int(triplet[1]), int(triplet[2])
                    if h not in self.entity2id:
                        self.entity2id[h] = h
                    if t not in self.entity2id:
                        self.entity2id[t] = t
                    if not saved_relation2id and r not in self.relation2id:
                        self.relation2id[r] = r
                    self.kg_triplets[file_type].append([h, t, r])
                    if h in self.train_ent:
                        self.ddi_in_kg.add(h)
                    if t in self.train_ent:
                        self.ddi_in_kg.add(t)

        if ratio < 1:
            n_train = len(self.kg_triplets['train'])
            n_valid = len(self.kg_triplets['valid'])
            n_test = len(self.kg_triplets['valid'])
            self.kg_triplets['train'] = random.sample(self.kg_triplets['train'], int(ratio*n_train))
            self.kg_triplets['valid'] = random.sample(self.kg_triplets['valid'], int(ratio*n_valid))
            self.kg_triplets['test'] = random.sample(self.kg_triplets['test'], int(ratio*n_test))

        train_kg = self.kg_triplets['train']
        valid_kg = self.kg_triplets['valid']
        test_kg  = self.kg_triplets['test']
        self.train_kg = np.array(train_kg, dtype='int')
        self.valid_kg = np.array(valid_kg, dtype='int')
        self.test_kg = np.array(test_kg, dtype='int')
        print("KG triplets: Train-{} Valid-{} Test-{}".format(len(train_kg), len(valid_kg), len(test_kg)))

        self.all_ent = max(self.entity2id.keys()) + 1
        self.all_rel = max(self.relation2id.keys()) + 1

    def load_graph(self, triplets):
        edges = self.double_triple(triplets)
        idd = np.concatenate([np.expand_dims(np.arange(self.all_ent),1), np.expand_dims(np.arange(self.all_ent),1), 2*self.all_rel*np.ones((self.all_ent, 1))],1)
        edges = np.concatenate([edges, idd], axis=0)
        values = np.ones(edges.shape[0])
        adjs = torch.sparse_coo_tensor(indices=torch.LongTensor(edges).t(), values=torch.FloatTensor(values), size=torch.Size([self.all_ent, self.all_ent, 2*self.all_rel+1]), requires_grad=False).cuda()
        return adjs

    def shuffle_train(self, ratio=0.8):
        n_ent = len(self.ddi_in_kg)
        train_ent = set(self.train_ent) - set(np.random.choice(list(self.ddi_in_kg), n_ent-int(n_ent*ratio)))
        all_triplet = np.array(self.triplets['train'])
        if self.dataset.startswith('S1'):
            fact_triplet = []
            train_data = []
            for i in range(len(all_triplet)):
                h, t, r = all_triplet[i]
                if h in train_ent and t in train_ent:
                    fact_triplet.append([h,t,r])
                elif h in train_ent or t in train_ent:
                    train_data.append([h,t,r])
            fact_triplet = np.array(fact_triplet)
            kg_triplets = np.concatenate([fact_triplet, self.train_kg], axis=0)
            self.KG = self.load_graph(kg_triplets)
            self.train_data = np.array(train_data)
        elif self.dataset.startswith('S2'):
            fact_triplet = []
            train_data = []
            for i in range(len(all_triplet)):
                h, t, r = all_triplet[i]
                if h in train_ent and t in train_ent:
                    fact_triplet.append([h,t,r])
                elif h not in train_ent and t not in train_ent:
                    train_data.append([h,t,r])
            fact_triplet = np.array(fact_triplet)
            kg_triplets = np.concatenate([fact_triplet, self.train_kg], axis=0)
            self.KG = self.load_graph(kg_triplets)
            self.train_data = np.array(train_data)
        elif self.dataset.startswith('S0'):
            n_all = len(all_triplet)
            rand_idx = np.random.permutation(n_all)
            all_triplet = all_triplet[rand_idx]
            n_fact = int(n_all * 0.8)
            kg_triplets = np.concatenate([all_triplet[:n_fact], self.train_kg], axis=0)
            self.KG = self.load_graph(kg_triplets)

            self.train_data = np.array(all_triplet[n_fact:].tolist())
        self.n_train = len(self.train_data)

    def double_triple(self, triplet):
        new_triples = []
        n_rel = self.all_rel
        for triple in triplet:
            h, t, r = triple
            new_triples.append([t, h, r])
            new_triples.append([h, t, r+n_rel])
        new_triples = np.array(new_triples)
        return new_triples

    def load_graph_from_triplets(self, triplets):
        if len(triplets) == 0:
            edges = np.empty((0, 3), dtype='int')
        else:
            edges = self.double_triple(triplets)
        idd = np.concatenate([np.expand_dims(np.arange(self.all_ent),1), np.expand_dims(np.arange(self.all_ent),1), 2*self.all_rel*np.ones((self.all_ent, 1))],1)
        edges = np.concatenate([edges, idd], axis=0)
        values = np.ones(edges.shape[0])
        adjs = torch.sparse_coo_tensor(indices=torch.LongTensor(edges).t(), values=torch.FloatTensor(values), size=torch.Size([self.all_ent, self.all_ent, 2*self.all_rel+1]), requires_grad=False).cuda()
        return adjs

    def load_pair_kgs_from_npz(self, npz_path):
        print(f"Loading pair-specific KGs from {npz_path}...")
        data = np.load(npz_path)
        
        edge_offsets = data['edge_offsets']
        edge_heads = data['edge_heads']
        edge_tails = data['edge_tails']
        edge_rels = data['edge_rels']
        
        heads = data['heads']
        tails = data['tails']
        rels = data['rels']
        
        num_pairs = len(heads)
        pair_triplets = np.stack([heads, tails, rels], axis=1)
        
        pair_kgs = []
        for i in range(num_pairs):
            start = edge_offsets[i]
            end = edge_offsets[i+1]
            
            h_list = edge_heads[start:end]
            t_list = edge_tails[start:end]
            r_list = edge_rels[start:end]
            
            sub_triplets = np.stack([h_list, t_list, r_list], axis=1) if len(h_list) > 0 else np.empty((0, 3), dtype='int')
            pair_kgs.append(self.load_graph_from_triplets(sub_triplets))
            
        print(f"Successfully loaded {len(pair_kgs)} KGs.")
        return pair_triplets, pair_kgs

    def shuffle_train_pair_kg(self):
        indices = np.random.permutation(len(self.train_data))
        self.train_data = self.train_data[indices]
        self.train_pair_kgs = [self.train_pair_kgs[i] for i in indices]
