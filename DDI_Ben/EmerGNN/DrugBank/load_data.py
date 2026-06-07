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
        self.params = params

        ddi_paths = {
            'train': params.train_ddi,
            'valid': params.valid_ddi,
        }
        
        test_ddi_paths = {
            'S0': params.test_ddi_s0,
            'S1': params.test_ddi_s1,
            'S2': params.test_ddi_s2
        }

        kg_paths = {
            'train': params.kg_train,
            'valid': params.kg_valid,
            'test':  params.kg_test
        }
        
        self.process_files_ddi(ddi_paths, saved_relation2id)
        self.process_test_files_ddi(test_ddi_paths, saved_relation2id)
        self.process_files_kg(kg_paths, saved_relation2id)
        self.load_ent_id()

        self.shuffle_train()
        train_pos = self.triplets['train'][self.triplets['train'][:, 2] != 0]
        valid_pos = self.triplets['valid'][self.triplets['valid'][:, 2] != 0]

        self.vKG = self.load_graph(self.valid_kg, ddi_triplets=train_pos)
        
        # Load KG cho từng test set
        self.test_KGs = {}
        for name in ['S0', 'S1', 'S2']:
            test_pos = self.test_triplets[name][self.test_triplets[name][:, 2] != 0]
            self.test_KGs[name] = self.load_graph(self.test_kg, ddi_triplets=np.concatenate([train_pos, valid_pos], axis=0))

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

        self.eval_ent = max(self.entity2id.keys()) + 1
        self.eval_rel = len(self.relation2id)
    
    def process_test_files_ddi(self, test_paths, saved_relation2id=None):
        """Load 3 test sets riêng: S0, S1, S2"""
        self.test_triplets = {}
        
        for name, file_path in test_paths.items():
            data = []
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
                
                data.append([h, t, r])
            
            self.test_triplets[name] = np.array(data, dtype='int')
        
        # Cập nhật lại eval_ent và eval_rel sau khi load test sets
        self.eval_ent = max(self.entity2id.keys()) + 1
        self.eval_rel = len(self.relation2id)
        

    def load_ent_id(self, ):
        id2entity = dict()
        id2relation = dict()
        drug_set = json.load(open(self.params.node2id_path, 'r'))
        entity_set = json.load(open(self.params.entity_drug_path, 'r'))
        relation_set = json.load(open(self.params.relation2id_path, 'r'))
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

    def load_graph(self, kg_triplets, ddi_triplets=None):
        kg_edges = self.double_triple(kg_triplets)
        if ddi_triplets is not None and len(ddi_triplets) > 0:
            edges = np.concatenate([kg_edges, ddi_triplets], axis=0)
        else:
            edges = kg_edges
        idd = np.concatenate([np.expand_dims(np.arange(self.all_ent),1), np.expand_dims(np.arange(self.all_ent),1), 2*self.all_rel*np.ones((self.all_ent, 1))],1)
        edges = np.concatenate([edges, idd], axis=0)
        values = np.ones(edges.shape[0])
        adjs = torch.sparse_coo_tensor(indices=torch.LongTensor(edges).t(), values=torch.FloatTensor(values), size=torch.Size([self.all_ent, self.all_ent, 2*self.all_rel+1]), requires_grad=False).cuda()
        return adjs

    def shuffle_train(self, ratio=0.8):
        n_ent = len(self.ddi_in_kg)
        train_ent = set(self.train_ent) - set(np.random.choice(list(self.ddi_in_kg), n_ent-int(n_ent*ratio)))
        all_triplet = np.array(self.triplets['train'])

        positive_mask = all_triplet[:, 2] != 0
        negative_mask = all_triplet[:, 2] == 0

        positive_triplet = all_triplet[positive_mask]
        negative_triplet = all_triplet[negative_mask]

        if self.dataset.startswith('S1'):
            fact_triplet = []
            train_data = []
            for i in range(len(positive_triplet)):
                h, t, r = positive_triplet[i]
                if h in train_ent and t in train_ent:
                    fact_triplet.append([h,t,r])
                elif h in train_ent or t in train_ent:
                    train_data.append([h,t,r])
            fact_triplet = np.array(fact_triplet)
            self.KG = self.load_graph(self.train_kg, ddi_triplets=fact_triplet)
            self.train_data = np.array(train_data)
        elif self.dataset.startswith('S2'):
            fact_triplet = []
            train_data = []
            for i in range(len(positive_triplet)):
                h, t, r = positive_triplet[i]
                if h in train_ent and t in train_ent:
                    fact_triplet.append([h,t,r])
                elif h not in train_ent and t not in train_ent:
                    train_data.append([h,t,r])
            fact_triplet = np.array(fact_triplet)
            self.KG = self.load_graph(self.train_kg, ddi_triplets=fact_triplet)
            self.train_data = np.array(train_data)
        elif self.dataset.startswith('S0'):
            n_all = len(positive_triplet)
            rand_idx = np.random.permutation(n_all)
            positive_triplet = positive_triplet[rand_idx]
            n_fact = int(n_all * 0.8)
            self.KG = self.load_graph(self.train_kg, ddi_triplets=positive_triplet[:n_fact])

            self.train_data = np.array(positive_triplet[n_fact:].tolist())
        if len(negative_triplet) > 0:
            if len(self.train_data) > 0:
                self.train_data = np.concatenate([self.train_data, negative_triplet], axis=0)
            else:
                self.train_data = negative_triplet
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
