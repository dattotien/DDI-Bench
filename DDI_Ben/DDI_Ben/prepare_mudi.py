"""Chuyển `data/mudi_raw/` sang layout chuẩn của DDI_Ben.

    python prepare_mudi.py            # dry run, chỉ in báo cáo
    python prepare_mudi.py --write    # ghi thật

Đọc:
    data/mudi_raw/MUDIv2_train.csv    Drug1, Pharmacodynamics, Pharmacokinetics, Drug2
    data/mudi_raw/MUDIv2_val.csv      (idem + sorted_key)
    data/mudi_raw/test_S{0,1,2}.csv   Drug1, Pharmacodynamics, Drug2, Adverse Effects, Pharmacokinetics
    data/mudi_raw/id2smiles.pt        {DrugBank_ID: SMILES}

Ghi:
    data/mudi_cluster/train.txt              mỗi dòng `head tail rel`
    data/mudi_cluster/valid_S{0,1,2}.txt     cả ba đều là bản copy của MUDIv2_val.csv
    data/mudi_cluster/test_S{0,1,2}.txt
    data/initial/mudi/id2smiles.json         {node_id: SMILES}
    data/initial/mudi/node2drugbank.json     {node_id: DrugBank_ID}
    data/initial/mudi/DB_molecular_feats.pkl DrugBank_ID / Node_ID / SMILES / Morgan_Features

## Hai điều quan trọng về thứ tự

1. **Không sắp xếp lại dòng.** File val/test của MUDI xếp theo dạng
   `[nửa đầu = chiều xuôi | nửa sau = chiều ngược]`; `metric.py` so dòng `i` với
   dòng `i + N/2`. Script giữ nguyên thứ tự gốc, và kiểm tra lại tính chất này.
2. **`valid_S0/S1/S2` là ba bản copy y hệt** của `MUDIv2_val.csv` — MUDI chỉ có
   một tập val, còn `trainer.py` chọn model theo từng `valid_S*` tương ứng với
   `test_S*`. Duplicate để mỗi test split có một valid cùng tên.

## Nhãn

Lấy từ cột `Pharmacodynamics`, đúng 4 lớp khớp `label_mapping` của mudi trong
`dataset_registry.py`: No Interaction=0, Synergism=1, Antagonism=2, New Effect=3.
Cột `Pharmacokinetics` và `Adverse Effects` không dùng ở đây.

## Node id

Gán theo thứ tự DrugBank ID tăng dần trên toàn bộ 1295 drug xuất hiện ở mọi
split (cùng quy ước với MecDDI), và lưu lại vào `node2drugbank.json`.
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pickle as pkl
import torch

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

import dataset_registry as R

RDLogger.DisableLog('rdApp.*')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

RAW_DIR = os.path.join('data', 'mudi_raw')
LABEL_COL = 'Pharmacodynamics'
MORGAN_RADIUS = 2
MORGAN_BITS = 1024
RDKIT2D_DIM = 200

### split đích -> file raw. valid_S0/S1/S2 dùng chung một file val (duplicate).
SPLIT_SOURCES = [
    ('train', 'MUDIv2_train.csv'),
    ('valid_S0', 'MUDIv2_val.csv'),
    ('valid_S1', 'MUDIv2_val.csv'),
    ('valid_S2', 'MUDIv2_val.csv'),
    ('test_S0', 'test_S0.csv'),
    ('test_S1', 'test_S1.csv'),
    ('test_S2', 'test_S2.csv'),
]


def strip_prefix(name):
    """'Compound::DB00842' -> 'DB00842'"""
    return str(name).split('::')[-1]


def morgan_count_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    vec = np.zeros(MORGAN_BITS)
    fp = AllChem.GetHashedMorganFingerprint(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    for bit, count in fp.GetNonzeroElements().items():
        vec[bit] = count
    return vec


def read_raw(raw_dir):
    """{split: DataFrame} giữ nguyên thứ tự dòng gốc."""
    frames, seen = {}, {}
    for split, filename in SPLIT_SOURCES:
        path = os.path.join(raw_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError('thiếu %s' % path)
        if filename not in seen:
            df = pd.read_csv(path)
            for col in ('Drug1', 'Drug2', LABEL_COL):
                if col not in df.columns:
                    raise ValueError("%s thiếu cột '%s' (đang có: %s)"
                                     % (path, col, list(df.columns)))
            seen[filename] = df
        frames[split] = seen[filename]
    return frames


def check_directed_layout(split, df):
    """Nửa sau phải là chiều ngược của nửa đầu — điều metric.py dựa vào."""
    n = len(df)
    if n % 2:
        return 'số dòng lẻ (%d) nên không thể chia đôi xuôi/ngược' % n
    half = n // 2
    a, b = df.iloc[:half], df.iloc[half:]
    ok = ((a['Drug1'].values == b['Drug2'].values) &
          (a['Drug2'].values == b['Drug1'].values)).sum()
    if ok != half:
        return 'chỉ %d/%d dòng nửa sau là chiều ngược của nửa đầu' % (ok, half)
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw-dir', default=RAW_DIR)
    ap.add_argument('--dataset_type', default='cluster', choices=['cluster', 'random'],
                    help='ghi vào data/mudi_<dataset_type>/')
    ap.add_argument('--write', action='store_true', help='ghi thật (mặc định dry run)')
    args = ap.parse_args()
    args.dataset = 'mudi'
    cfg = R.apply_to_args(args)

    label_mapping = cfg['label_mapping']
    frames = read_raw(args.raw_dir)

    ### ---- nhãn ----
    labels_seen = set()
    for split, df in frames.items():
        labels_seen |= set(df[LABEL_COL].dropna().unique())
    unknown = sorted(labels_seen - set(label_mapping))
    if unknown:
        raise ValueError(
            "cột '%s' có nhãn không nằm trong label_mapping của mudi: %s. "
            "Cập nhật label_mapping (và num_rel) trong dataset_registry.py."
            % (LABEL_COL, unknown))
    print('Nhãn      : %s' % ', '.join('%s=%d' % (k, v) for k, v in
                                       sorted(label_mapping.items(), key=lambda kv: kv[1])))

    ### ---- node id: sắp theo DrugBank ID trên toàn bộ drug của mọi split ----
    drugs = set()
    for split, df in frames.items():
        drugs |= set(df['Drug1'].map(strip_prefix)) | set(df['Drug2'].map(strip_prefix))
    drugs = sorted(drugs)
    drug2id = {d: i for i, d in enumerate(drugs)}
    print('Drug      : %d (num_ent khai báo: %d)' % (len(drugs), args.num_ent))
    if len(drugs) != args.num_ent:
        raise ValueError(
            'data raw có %d drug nhưng dataset_registry khai báo num_ent=%d — sửa một '
            'trong hai chỗ cho khớp.' % (len(drugs), args.num_ent))

    ### ---- SMILES ----
    smiles_path = os.path.join(args.raw_dir, 'id2smiles.pt')
    raw_smiles = torch.load(smiles_path, map_location='cpu', weights_only=False)
    raw_smiles = {strip_prefix(k): v for k, v in raw_smiles.items()}
    id2smiles, no_smiles, unparsable = {}, [], []
    rows = []
    for dbid in drugs:
        node_id = drug2id[dbid]
        smiles = (raw_smiles.get(dbid) or '').strip()
        if not smiles:
            no_smiles.append(node_id)
            continue
        fp = morgan_count_fp(smiles)
        if fp is None:
            unparsable.append((node_id, dbid))
            continue
        id2smiles[str(node_id)] = smiles
        rows.append({'DrugBank_ID': dbid, 'Node_ID': node_id, 'SMILES': smiles,
                     'Morgan_Features': fp, 'RDKit2D_Features': np.zeros(RDKIT2D_DIM)})
    print('SMILES    : %d/%d drug (%s)' % (len(id2smiles), len(drugs), smiles_path))
    if no_smiles:
        print('   %d drug không có SMILES: %s' % (len(no_smiles), R._describe_ids(no_smiles)))
    if unparsable:
        print('   %d drug rdkit không parse được: %s'
              % (len(unparsable), ', '.join('%d/%s' % u for u in unparsable[:8])))

    ### ---- các split ----
    out_dir = R.data_dir(args)
    initial_dir = R.initial_dir(args)
    planned = []
    for split, filename in SPLIT_SOURCES:
        df = frames[split]
        problem = None if split == 'train' else check_directed_layout(split, df)
        lines = ['%d %d %d' % (drug2id[strip_prefix(h)], drug2id[strip_prefix(t)],
                               label_mapping[r])
                 for h, t, r in zip(df['Drug1'], df['Drug2'], df[LABEL_COL])]
        planned.append((split, filename, lines, problem))
        note = '' if problem is None else '  <-- %s' % problem
        dup = '  (copy của %s)' % filename if split.startswith('valid_S') else ''
        print('%-9s <- %-18s %7d dòng%s%s' % (split, filename, len(lines), dup, note))

    broken = [(s, p) for s, _, _, p in planned if p]
    if broken:
        print('\nCẢNH BÁO: metric.py so dòng i với dòng i+N/2, các split sau không có '
              'cấu trúc đó nên metric sẽ sai:')
        for split, problem in broken:
            print('   %s: %s' % (split, problem))

    print('\nSẽ ghi    : %s/{%s}.txt' % (out_dir, ','.join(s for s, _ in SPLIT_SOURCES)))
    print('            %s/id2smiles.json, node2drugbank.json, DB_molecular_feats.pkl'
          % initial_dir)

    if not args.write:
        print('\n(dry run — thêm --write để ghi thật)')
        return 0
    if broken:
        print('\nDừng lại: sửa data raw trước, hoặc bỏ directed_eval của mudi trong '
              'dataset_registry.py nếu bạn muốn dùng metric thường.')
        return 1

    for d in (out_dir, initial_dir):
        if not os.path.isdir(d):
            os.makedirs(d)

    for split, _, lines, _ in planned:
        path = R.split_path(args, split)
        if os.path.exists(path):
            shutil.copy2(path, path + '.bak')
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(lines) + '\n')
    print('đã ghi %d split vào %s' % (len(planned), out_dir))

    for name, payload in [('smiles_file', id2smiles),
                          ('node_map_file', {str(drug2id[d]): d for d in drugs})]:
        filename = cfg.get(name) or ('node2drugbank.json' if name == 'node_map_file'
                                     else 'id2smiles.json')
        path = os.path.join(initial_dir, filename)
        if os.path.exists(path):
            shutil.copy2(path, path + '.bak')
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(payload, f)
        print('đã ghi %s' % path)

    feat_path = os.path.join(initial_dir, cfg['feat_file'])
    if os.path.exists(feat_path):
        shutil.copy2(feat_path, feat_path + '.bak')
    with open(feat_path, 'wb') as f:
        pkl.dump(pd.DataFrame(rows).sort_values('Node_ID').reset_index(drop=True), f)
    print('đã ghi %s (%d drug)' % (feat_path, len(rows)))

    print('\nkiểm tra lại: python check_data.py mudi %s' % args.dataset_type)
    return 0


if __name__ == '__main__':
    sys.exit(main())
