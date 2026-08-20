"""Central registry for every dataset supported by DDI_Ben.

Adding a dataset means adding one entry below -- no ``if args.dataset == ...``
branch anywhere else. Two things are described per dataset:

* its shape (``num_ent`` / ``num_rel``) and its task
  (``multiclass``: one label per pair, ``multilabel``: TWOSIDES-style vector),
* where its files live, so no two datasets ever share a folder.

Folder layout expected for a dataset named ``<name>``::

    data/<name>_<dataset_type>/     train.txt, valid_S{0,1,2}.txt, test_S{0,1,2}.txt
    data/initial/<name>/            feature / SMILES / network files

The model code branches on ``args.task``, never on the dataset name, so a new
multiclass dataset works with every model without further edits.
"""

import json
import os
import pickle as pkl

MULTICLASS = 'multiclass'
MULTILABEL = 'multilabel'

DEFAULT_SPLITS = ['train', 'valid_S0', 'test_S0', 'valid_S1', 'test_S1', 'valid_S2', 'test_S2']

DATASETS = {
    'drugbank': {
        'num_ent': 1710,
        'num_rel': 86,
        'task': MULTICLASS,
        'feat_file': 'DB_molecular_feats.pkl',   # dict with 'Morgan_Features' + 'SMILES'
        'feat_key': 'Morgan_Features',
        'feat_id_key': None,                     # rows already ordered by drug id
        'smiles_file': 'id2smiles.json',         # {id: smiles}
        'cid2id_file': None,
        'network_file': 'relations_2hop.txt',
        # TIGER reads SMILES out of the feature pickle instead of id2smiles.json,
        # and these ids hold SMILES rdkit cannot parse -> blanked out.
        'tiger_smiles_from_feat': True,
        'tiger_blank_ids': [6, 136, 889, 1171, 1239, 1254],
    },
    'mecddi': {
        'num_ent': 1567,
        'num_rel': 103,
        'task': MULTICLASS,
        'feat_file': 'DB_molecular_feats.pkl',
        'feat_key': 'Morgan_Features',
        # rows are NOT ordered by drug id -- the real id sits in the 'Node_ID'
        # column, and 95 of the 1567 drugs have no row at all.
        'feat_id_key': 'Node_ID',
        'smiles_file': 'id2smiles.json',
        # node id -> DrugBank ID, khôi phục từ nguồn MecDDI; build_features.py dùng
        # file này để gán id chính xác thay vì phải dò theo thứ tự
        'node_map_file': 'node2drugbank.json',
        'cid2id_file': None,
        'network_file': None,                    # no relations_2hop.txt -> no Decagon / TIGER
        'tiger_smiles_from_feat': False,
        'tiger_blank_ids': [],
    },
    'mudi': {
        'num_ent': 1295,
        'num_rel': 4,                            # No Interaction / Synergism / Antagonism / New Effect
        'task': MULTICLASS,
        'feat_file': 'DB_molecular_feats.pkl',
        'feat_key': 'Morgan_Features',
        'feat_id_key': None,
        'smiles_file': 'id2smiles.json',
        'cid2id_file': None,
        'network_file': None,
        'tiger_smiles_from_feat': False,
        'tiger_blank_ids': [],
    },
    'twosides': {
        'num_ent': 645,
        'num_rel': 209,
        'task': MULTILABEL,
        'feat_file': 'DB_molecular_feats.pkl',   # plain list of feature vectors
        'feat_key': None,
        'feat_id_key': None,
        'smiles_file': 'cid2smiles.json',        # {cid: smiles}, needs cid2id to map to ids
        'cid2id_file': 'cid2id.json',
        'network_file': 'relations_2hop.txt',
        'tiger_smiles_from_feat': False,
        'tiger_blank_ids': [],
    },
}

DATASET_NAMES = list(DATASETS.keys())


def get_config(dataset):
    if dataset not in DATASETS:
        raise ValueError(
            "unknown dataset '{}'; known datasets: {}".format(dataset, ', '.join(DATASET_NAMES)))
    cfg = dict(DATASETS[dataset])
    cfg.setdefault('splits', DEFAULT_SPLITS)
    cfg['name'] = dataset
    return cfg


def apply_to_args(args):
    """Attach the dataset config to ``args`` -- call once, right after parsing."""
    cfg = get_config(args.dataset)
    args.dataset_cfg = cfg
    args.task = cfg['task']
    args.num_ent = cfg['num_ent']
    args.num_rel = cfg['num_rel']
    args.splits = cfg['splits']
    return cfg


def is_multiclass(args):
    return args.task == MULTICLASS


def is_multilabel(args):
    return args.task == MULTILABEL


def folder_name(args):
    """Split folder name, e.g. ``mecddi_cluster``."""
    return '{}_{}'.format(args.dataset, args.dataset_type)


def data_dir(args):
    return os.path.join('data', folder_name(args))


def split_path(args, split):
    return os.path.join(data_dir(args), '{}.txt'.format(split))


def initial_dir(args):
    return os.path.join('data', 'initial', args.dataset)


def initial_path(args, key):
    """Path of an ``initial/`` file declared in the registry, or None if absent."""
    filename = args.dataset_cfg.get(key)
    if filename is None:
        return None
    return os.path.join(initial_dir(args), filename)


def require_initial_path(args, key, needed_by):
    path = initial_path(args, key)
    if path is None:
        raise FileNotFoundError(
            "{} needs '{}' but dataset '{}' does not declare one in dataset_registry.py. "
            "Add the file to {}/ and set '{}' in the '{}' entry, or run on another "
            "dataset.".format(needed_by, key, args.dataset, initial_dir(args), key, args.dataset))
    if not os.path.exists(path):
        raise FileNotFoundError(
            "{} needs '{}' for dataset '{}' but the file is missing: {}".format(
                needed_by, key, args.dataset, path))
    return path


def network_path(args, needed_by):
    """Path of relations_2hop.txt; raises when the dataset has no such network."""
    return require_initial_path(args, 'network_file', needed_by)


def _describe_ids(ids, limit=10):
    ids = sorted(ids)
    head = ', '.join(str(i) for i in ids[:limit])
    return head + (', ...' if len(ids) > limit else '')


def load_features(args):
    """Molecular features as a list where position i IS drug id i.

    Some feature files are not stored in id order (MecDDI keeps the real id in a
    ``Node_ID`` column), so ``feat_id_key`` says which column to realign on.
    Drugs with no row get a zero vector and a warning -- silently mis-indexed
    fingerprints would be far worse.
    """
    path = require_initial_path(args, 'feat_file', 'use_feat=1')
    with open(path, 'rb') as f:
        x = pkl.load(f, encoding='utf-8')
    cfg = args.dataset_cfg
    key = cfg['feat_key']
    rows = list(x if key is None else x[key])
    id_key = cfg.get('feat_id_key')

    if id_key is None:
        if len(rows) < args.num_ent:
            raise ValueError(
                "dataset '{}' declares num_ent={} but {} only has {} feature rows, so "
                "drug ids >= {} would index out of bounds. Regenerate the feature file "
                "to cover every id used in {}/, or fix num_ent in "
                "dataset_registry.py.".format(args.dataset, args.num_ent, path, len(rows),
                                              len(rows), data_dir(args)))
        if len(rows) > args.num_ent:
            print("[dataset_registry] warning: {} has {} feature rows but num_ent={}; "
                  "extra rows ignored.".format(path, len(rows), args.num_ent))
        return rows[:args.num_ent]

    ids = [int(v) for v in x[id_key]]
    dim = len(rows[0])
    by_id = [None] * args.num_ent
    for drug_id, row in zip(ids, rows):
        if 0 <= drug_id < args.num_ent:
            by_id[drug_id] = row
    missing = [i for i, row in enumerate(by_id) if row is None]
    if missing:
        print("[dataset_registry] warning: {} of {} drugs in '{}' have no molecular "
              "features in {} (ids: {}); they get a zero vector. Regenerate that file "
              "to fix.".format(len(missing), args.num_ent, args.dataset, path,
                               _describe_ids(missing)))
        for i in missing:
            by_id[i] = [0.0] * dim
    return by_id


def load_id2smiles(args, needed_by='this model', require_all_ids=False):
    """``{str(drug_id): smiles}`` for the current dataset.

    The mapping is keyed by drug id, NOT positional: id2smiles.json is not
    guaranteed to be ordered 0..num_ent-1 (TWOSIDES and MecDDI are not), so
    callers must look drugs up by id rather than by enumeration order.
    """
    smiles_path = require_initial_path(args, 'smiles_file', needed_by)
    with open(smiles_path, 'r') as f:
        smiles = json.load(f)
    if args.dataset_cfg['cid2id_file'] is None:
        id2smiles = {str(k): v for k, v in smiles.items()}
    else:
        cid2id_path = require_initial_path(args, 'cid2id_file', needed_by)
        with open(cid2id_path, 'r') as f:
            cid2id = json.load(f)
        id2smiles = {str(cid2id[cid]): smiles[cid] for cid in smiles}
    if require_all_ids:
        missing = [i for i in range(args.num_ent) if str(i) not in id2smiles]
        if missing:
            raise ValueError(
                "{} needs a SMILES string for every drug of '{}' but {} of {} ids are "
                "absent from {} (ids: {}). Regenerate that file so it covers ids "
                "0..{}.".format(needed_by, args.dataset, len(missing), args.num_ent,
                                smiles_path, _describe_ids(missing), args.num_ent - 1))
    return id2smiles


def load_tiger_id2smiles(args):
    """SMILES map for TIGER (DrugBank keeps them inside the feature pickle)."""
    if not args.dataset_cfg['tiger_smiles_from_feat']:
        return load_id2smiles(args, needed_by='TIGER')
    path = require_initial_path(args, 'feat_file', 'TIGER')
    with open(path, 'rb') as f:
        x = pkl.load(f, encoding='utf-8')
    id2smiles = {str(j): x['SMILES'][j] for j in range(args.num_ent)}
    for j in args.dataset_cfg['tiger_blank_ids']:
        id2smiles[str(j)] = ''
    return id2smiles
