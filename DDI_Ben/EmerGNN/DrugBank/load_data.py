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

        dataset_folder = params.dataset.split('_')[0]
        ddi_paths = {
            'train': os.path.join(self.task_dir, 'data/{}/{}_ddi.txt'.format(dataset_folder, 'train')),
            'valid': os.path.join(self.task_dir, 'data/{}/{}_ddi.txt'.format(dataset_folder, 'valid')),
            'test':  os.path.join(self.task_dir, 'data/{}/{}_ddi.txt'.format(dataset_folder, 'test'))
        }

        kg_paths = {
            'train': os.path.join(self.task_dir, 'data/KG.txt'),
            'valid': os.path.join(self.task_dir, 'data/KG.txt'),
            'test':  os.path.join(self.task_dir, 'data/KG.txt')
        }
        
        self.process_files_ddi(ddi_paths, saved_relation2id)

        if hasattr(params, 'use_dynamic_subgraph_sampling') and params.use_dynamic_subgraph_sampling:
            self.use_dynamic_subgraph_sampling = True
            self.process_files_kg(kg_paths, saved_relation2id)
            self.load_ent_id()
            self.use_pair_kg = True
            
            self.valid_pair_kgs = DynamicPairKGs(
                self.triplets['valid'], self, params.length,
                base_kg_triplets=self.valid_kg, ddi_triplets=self.triplets['train']
            )
                
            self.test_pair_kgs = DynamicPairKGs(
                self.triplets['test'], self, params.length,
                base_kg_triplets=self.test_kg,
                ddi_triplets=np.concatenate([self.triplets['train'], self.triplets['valid']], axis=0)
            )
            self.train_data = self.triplets['train']
        elif hasattr(params, 'use_pair_kg') and params.use_pair_kg:
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

    def shuffle_train(self, ratio=1.0):
        all_triplet = np.array(self.triplets['train'])
        if ratio == 1.0:
            kg_triplets = np.concatenate([all_triplet, self.train_kg], axis=0)
            self.KG = self.load_graph(kg_triplets)
            self.train_data = all_triplet
            self.n_train = len(self.train_data)
            return
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
            
        if not hasattr(self, 'idd'):
            self.idd = np.concatenate([np.expand_dims(np.arange(self.all_ent),1), np.expand_dims(np.arange(self.all_ent),1), 2*self.all_rel*np.ones((self.all_ent, 1))],1)
            
        edges = np.concatenate([edges, self.idd], axis=0)
        values = np.ones(edges.shape[0], dtype='float32')
        adjs = torch.sparse_coo_tensor(indices=torch.LongTensor(edges).t(), values=torch.FloatTensor(values), size=torch.Size([self.all_ent, self.all_ent, 2*self.all_rel+1]), requires_grad=False)
        return adjs

    def load_pair_kgs_from_npz(self, npz_path):
        print(f"Loading pair-specific KGs from {npz_path}...")
        data = np.load(npz_path)
        
        heads = data['heads']
        tails = data['tails']
        rels = data['rels']
        
        pair_triplets = np.stack([heads, tails, rels], axis=1)
        pair_kgs = PairKGs(data, self.all_ent, self.all_rel, self)
            
        print(f"Successfully loaded {len(pair_kgs)} KGs.")
        return pair_triplets, pair_kgs

    def shuffle_train_pair_kg(self):
        indices = np.random.permutation(len(self.train_data))
        self.train_data = self.train_data[indices]
        if hasattr(self.train_pair_kgs, 'shuffle'):
            self.train_pair_kgs.shuffle(indices)
        else:
            self.train_pair_kgs = [self.train_pair_kgs[i] for i in indices]

    def bfs_distances(self, sp_csr, source, exclude_node, max_depth):
        UNREACH = max_depth + 1
        dist = np.full(self.all_ent, UNREACH, dtype=np.int8)
        dist[source] = 0
        frontier = np.array([source], dtype=np.int64)
        for d in range(1, max_depth + 1):
            if len(frontier) == 0:
                break
            nbrs = sp_csr[frontier].indices
            if len(nbrs) == 0:
                break
            if d == 1:
                nbrs = nbrs[nbrs != exclude_node]
            new_mask = dist[nbrs] > d
            new_nodes = np.unique(nbrs[new_mask])
            if len(new_nodes) == 0:
                break
            dist[new_nodes] = d
            frontier = new_nodes
        return dist

    def extract_tight_subgraph(self, sp_csr, h, t, L):
        d_h = self.bfs_distances(sp_csr, h, t, L)
        d_t = self.bfs_distances(sp_csr, t, h, L)
        tight_mask = (d_h.astype(np.int16) + d_t.astype(np.int16)) <= L
        nodes = np.where(tight_mask)[0]
        return nodes

    def get_dynamic_batch_kgs(self, batch_h, batch_t, L):
        """Dynamically extract subgraphs for a batch on the fly"""
        batch_kgs = []
        for i in range(len(batch_h)):
            h, t = int(batch_h[i]), int(batch_t[i])
            nodes = self.extract_tight_subgraph(self.epoch_adj, h, t, L)
            if len(nodes) == 0:
                nodes = np.array([h, t], dtype=np.int64)
                
            sub_edges_h = []
            sub_edges_t = []
            sub_edges_r = []
            node_set_lookup = set(nodes)
            for u in nodes:
                for v, r_edge in self.epoch_adj_list[u]:
                    if v in node_set_lookup:
                        # Avoid target leakage
                        if (u == h and v == t) or (u == t and v == h):
                            continue
                        sub_edges_h.append(u)
                        sub_edges_t.append(v)
                        sub_edges_r.append(r_edge)
                        
            sub_triplets = np.stack([sub_edges_h, sub_edges_t, sub_edges_r], axis=1) if len(sub_edges_h) > 0 else np.empty((0, 3), dtype='int')
            batch_kgs.append(self.load_graph_from_triplets(sub_triplets))
        return batch_kgs

class PairKGs:
    def __init__(self, data, all_ent, all_rel, dataloader):
        self.edge_offsets = data['edge_offsets'][:]
        self.edge_heads = data['edge_heads'][:]
        self.edge_tails = data['edge_tails'][:]
        self.edge_rels = data['edge_rels'][:]
        self.all_ent = all_ent
        self.all_rel = all_rel
        self.dataloader = dataloader
        self.indices = np.arange(len(self.edge_offsets) - 1)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self._get_single(self.indices[i]) for i in range(*idx.indices(len(self)))]
        elif isinstance(idx, (list, np.ndarray, tuple)):
            return [self._get_single(self.indices[i]) for i in idx]
        return self._get_single(self.indices[idx])

    def _get_single(self, i):
        start = self.edge_offsets[i]
        end = self.edge_offsets[i+1]
        h_list = self.edge_heads[start:end]
        t_list = self.edge_tails[start:end]
        r_list = self.edge_rels[start:end]
        sub_triplets = np.stack([h_list, t_list, r_list], axis=1) if len(h_list) > 0 else np.empty((0, 3), dtype='int')
        return self.dataloader.load_graph_from_triplets(sub_triplets)

    def shuffle(self, new_indices):
        self.indices = self.indices[new_indices]

class DynamicPairKGs:
    def __init__(self, triplets, dataloader, L, base_kg_triplets, ddi_triplets):
        self.triplets = triplets
        self.dataloader = dataloader
        self.L = L
        self.indices = np.arange(len(triplets))
        
        # Build CSR and adjacency list for this split
        from scipy import sparse
        n_nodes = dataloader.all_ent
        
        rows_all, cols_all = [], []
        for kg_trips in [base_kg_triplets, ddi_triplets]:
            if len(kg_trips) == 0:
                continue
            h = kg_trips[:, 0].astype(np.int64)
            t = kg_trips[:, 1].astype(np.int64)
            mask = h != t
            rows_all.extend([h[mask], t[mask]])
            cols_all.extend([t[mask], h[mask]])
            
        if len(rows_all) > 0:
            rows = np.concatenate(rows_all)
            cols = np.concatenate(cols_all)
            sp = sparse.csr_matrix(
                (np.ones(len(rows), dtype=np.int8), (rows, cols)),
                shape=(n_nodes, n_nodes),
            )
            sp.sum_duplicates()
            sp.data[:] = 1
            self.adj = sp
        else:
            self.adj = sparse.csr_matrix((n_nodes, n_nodes), dtype=np.int8)
            
        adj_list = [[] for _ in range(n_nodes)]
        for kg_trips in [base_kg_triplets, ddi_triplets]:
            if len(kg_trips) == 0:
                continue
            for h_edge, t_edge, r_edge in kg_trips:
                h_edge, t_edge, r_edge = int(h_edge), int(t_edge), int(r_edge)
                if h_edge != t_edge:
                    adj_list[h_edge].append((t_edge, r_edge))
        self.adj_list = adj_list

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            batch_indices = self.indices[start:stop:step]
            batch_h = self.triplets[batch_indices, 0]
            batch_t = self.triplets[batch_indices, 1]
            return self._get_batch(batch_h, batch_t)
        elif isinstance(idx, (list, np.ndarray, tuple)):
            batch_indices = self.indices[idx]
            batch_h = self.triplets[batch_indices, 0]
            batch_t = self.triplets[batch_indices, 1]
            return self._get_batch(batch_h, batch_t)
        else:
            real_idx = self.indices[idx]
            h = self.triplets[real_idx, 0]
            t = self.triplets[real_idx, 1]
            return self._get_batch([h], [t])[0]

    def _get_batch(self, batch_h, batch_t):
        batch_kgs = []
        for i in range(len(batch_h)):
            h, t = int(batch_h[i]), int(batch_t[i])
            # extract subgraph nodes
            d_h = self.dataloader.bfs_distances(self.adj, h, t, self.L)
            d_t = self.dataloader.bfs_distances(self.adj, t, h, self.L)
            tight_mask = (d_h.astype(np.int16) + d_t.astype(np.int16)) <= self.L
            nodes = np.where(tight_mask)[0]
            if len(nodes) == 0:
                nodes = np.array([h, t], dtype=np.int64)
                
            sub_edges_h = []
            sub_edges_t = []
            sub_edges_r = []
            node_set_lookup = set(nodes)
            for u in nodes:
                for v, r_edge in self.adj_list[u]:
                    if v in node_set_lookup:
                        # Avoid target leakage
                        if (u == h and v == t) or (u == t and v == h):
                            continue
                        sub_edges_h.append(u)
                        sub_edges_t.append(v)
                        sub_edges_r.append(r_edge)
                        
            sub_triplets = np.stack([sub_edges_h, sub_edges_t, sub_edges_r], axis=1) if len(sub_edges_h) > 0 else np.empty((0, 3), dtype='int')
            batch_kgs.append(self.dataloader.load_graph_from_triplets(sub_triplets))
        return batch_kgs

    def shuffle(self, new_indices):
        self.indices = self.indices[new_indices]
