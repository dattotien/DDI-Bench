"""Add the drugs that are missing from the MecDDI feature table.

The DDI triplets use node ids 0..num_ent-1 but DB_molecular_feats.pkl only had a
row for 1472 of them, so the models either crashed on an out-of-range index or
silently read another drug's feature vector.

node2drugbank.json / id2smiles.json already cover every node id (recovered from
the MecDDI source csvs), so this only has to compute the fingerprints for the
drugs that are absent from the table and append them. Existing rows are never
touched.

Usage (needs rdkit, e.g. on Kaggle), from the DDI_Ben/DDI_Ben directory:
    python rebuild_drugbank_feats.py
"""

import argparse
import json
import os
import pickle as pkl

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

MORGAN_BITS = 1024
MORGAN_RADIUS = 2
RDKIT2D_DIM = 200


def morgan_counts(mol):
    """Count-based Morgan fingerprint, matching the encoding of the existing rows."""
    fp = AllChem.GetHashedMorganFingerprint(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    vec = np.zeros(MORGAN_BITS, dtype=np.float64)
    for idx, count in fp.GetNonzeroElements().items():
        vec[idx] = count
    return vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', default='data/initial/drugbank/DB_molecular_feats.pkl')
    ap.add_argument('--id2smiles', default='data/initial/drugbank/id2smiles.json')
    ap.add_argument('--node2drugbank', default='data/initial/drugbank/node2drugbank.json')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    with open(args.id2smiles) as f:
        id2smiles = {int(k): v for k, v in json.load(f).items()}
    with open(args.node2drugbank) as f:
        node2db = {int(k): v for k, v in json.load(f).items()}
    with open(args.feats, 'rb') as f:
        table = pkl.load(f, encoding='utf-8')
    known = set(int(j) for j in table['Node_ID'])

    missing = sorted(set(node2db) - known)
    print('{} node ids total, {} already in the feature table, {} to add'
          .format(len(node2db), len(known), len(missing)))
    no_smiles = [n for n in missing if not id2smiles.get(n, '').strip()]
    if no_smiles:
        print('  {} of them have no SMILES: {}'.format(len(no_smiles), no_smiles))

    rows = []
    for node_id in missing:
        smiles = id2smiles.get(node_id, '').strip()
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            print('  skip node {} ({}): rdkit cannot parse its SMILES'.format(node_id, node2db[node_id]))
            continue
        rows.append({
            'DrugBank_ID': node2db[node_id],
            'Node_ID': node_id,
            'SMILES': smiles,
            'Morgan_Features': morgan_counts(mol).tolist(),
            ### no model in this repo reads RDKit2D_Features, keep the column shape only
            'RDKit2D_Features': [0.0] * RDKIT2D_DIM,
        })
    print('built features for {} drugs'.format(len(rows)))

    if args.dry_run or not rows:
        return

    out = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
    assert out['Node_ID'].is_unique
    os.replace(args.feats, args.feats + '.bak')
    with open(args.feats, 'wb') as f:
        pkl.dump(out, f)
    print('wrote {} ({} rows, backup at {}.bak)'.format(args.feats, len(out), args.feats))


if __name__ == '__main__':
    main()
