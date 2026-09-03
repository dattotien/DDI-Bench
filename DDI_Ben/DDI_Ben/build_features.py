"""Sinh DB_molecular_feats.pkl + id2smiles.json từ một file CSV `Drugbank_ID,SMILES`.

Dùng khi feature file của một dataset không phủ hết drug id (MecDDI: 1472/1567).

    python build_features.py --dataset mecddi --csv ../../drugbank_smiles_map.csv
    python build_features.py --dataset mecddi --csv ... --write

Mặc định chỉ in báo cáo (dry run); thêm `--write` mới ghi file (bản cũ được
backup thành `*.bak`).

## Node id lấy ở đâu

CSV chỉ có DrugBank ID, không có node id của dataset. Hai cách:

1. **Chính xác** — dùng file `node_map_file` khai báo trong registry
   (MecDDI: `data/initial/mecddi/node2drugbank.json`, phủ đủ id `0..num_ent-1`).
   Có file này thì không còn chỗ nào mơ hồ.
2. **Dò theo thứ tự** (fallback khi không có node map) — node id được đánh theo
   đúng thứ tự DrugBank ID tăng dần, nên khớp danh sách CSV đã sắp xếp với các
   node id đã biết là suy ra được phần lớn id còn thiếu. Chỗ nào một khoảng
   trống có nhiều ứng viên CSV hơn số id còn thiếu thì bỏ qua (trừ khi
   `--ambiguous guess` — cách đoán đó từng gán sai 4/7 drug của MecDDI).

## Morgan fingerprint

`AllChem.GetHashedMorganFingerprint(mol, radius=2, nBits=1024)` (count vector).
Công thức này tái tạo *đúng từng bit* feature có sẵn của MecDDI, nên feature mới
sinh ra cùng hệ với feature cũ.
"""

import argparse
import csv as csv_module
import json
import os
import shutil
import sys

import numpy as np
import pickle as pkl

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

import dataset_registry as R

RDLogger.DisableLog('rdApp.*')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

MORGAN_RADIUS = 2
MORGAN_BITS = 1024
RDKIT2D_DIM = 200


def morgan_count_fp(smiles):
    """Count-based Morgan fingerprint, hoặc None nếu rdkit không parse được."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    vec = np.zeros(MORGAN_BITS)
    fp = AllChem.GetHashedMorganFingerprint(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)
    for bit, count in fp.GetNonzeroElements().items():
        vec[bit] = count
    return vec


def rdkit2d_features(smiles):
    """200-dim normalized RDKit descriptors; zeros nếu chưa cài descriptastorus.

    DDI_Ben không đọc cột này, nó chỉ có mặt để giữ nguyên schema file gốc.
    """
    gen = getattr(rdkit2d_features, '_gen', 'unset')
    if gen == 'unset':
        try:
            from descriptastorus.descriptors import rdNormalizedDescriptors
            gen = rdNormalizedDescriptors.RDKit2DNormalized()
        except ImportError:
            gen = None
        rdkit2d_features._gen = gen
    if gen is None:
        return np.zeros(RDKIT2D_DIM)
    return np.array(gen.process(smiles)[1:])


def read_smiles_csv(path):
    with open(path, encoding='utf-8') as f:
        rows = list(csv_module.DictReader(f))
    if not rows:
        raise ValueError('%s rỗng' % path)
    cols = {c.lower(): c for c in rows[0]}
    id_col = cols.get('drugbank_id') or cols.get('drugbank id')
    smiles_col = cols.get('smiles')
    if not id_col or not smiles_col:
        raise ValueError("%s cần 2 cột 'Drugbank_ID' và 'SMILES', đang có: %s"
                         % (path, list(rows[0])))
    pairs = [(r[id_col].strip(), r[smiles_col].strip()) for r in rows if r[smiles_col]]
    pairs.sort(key=lambda p: p[0])
    return pairs


def load_existing(args):
    """{node_id: dict} của feature file hiện có, hoặc {} nếu chưa có file."""
    path = R.initial_path(args, 'feat_file')
    if not path or not os.path.exists(path):
        return {}, path
    with open(path, 'rb') as f:
        x = pkl.load(f, encoding='utf-8')
    id_key = args.dataset_cfg.get('feat_id_key')
    n = len(x[args.dataset_cfg['feat_key']])
    ids = [int(v) for v in x[id_key]] if id_key else list(range(n))
    known = {}
    for i, node_id in enumerate(ids):
        known[node_id] = {
            'DrugBank_ID': list(x['DrugBank_ID' if 'DrugBank_ID' in x else 'DrugBank ID'])[i],
            'SMILES': list(x['SMILES'])[i],
            'Morgan_Features': np.array(list(x[args.dataset_cfg['feat_key']])[i]),
            'RDKit2D_Features': np.array(list(x['RDKit2D_Features'])[i])
            if 'RDKit2D_Features' in x else np.zeros(RDKIT2D_DIM),
        }
    return known, path


def align(known, csv_pairs, num_ent):
    """Suy node id cho các drug còn thiếu.

    Trả về (resolved, ambiguous, unused) —
    resolved: {node_id: (drugbank_id, smiles)} chắc chắn,
    ambiguous: list (node_ids còn thiếu, [(dbid, smiles) ứng viên]),
    unused: các dòng CSV không thuộc dataset.
    """
    csv_index = {dbid: i for i, (dbid, _) in enumerate(csv_pairs)}

    anchors = [(-1, -1)]
    unmapped_known = []
    for node_id in sorted(known):
        dbid = known[node_id]['DrugBank_ID']
        if dbid in csv_index:
            anchors.append((node_id, csv_index[dbid]))
        else:
            unmapped_known.append((node_id, dbid))
    anchors.append((num_ent, len(csv_pairs)))

    for (n0, c0), (n1, c1) in zip(anchors, anchors[1:]):
        if n1 <= n0 or c1 <= c0:
            raise ValueError(
                'node id KHÔNG được đánh theo thứ tự DrugBank ID tăng dần '
                '(node %d -> csv %d rồi node %d -> csv %d). Không thể suy id từ CSV.'
                % (n0, c0, n1, c1))

    resolved, ambiguous, unused = {}, [], []
    for (n0, c0), (n1, c1) in zip(anchors, anchors[1:]):
        missing = list(range(n0 + 1, n1))
        cands = csv_pairs[c0 + 1:c1]
        if not missing:
            unused.extend(cands)
        elif len(missing) == len(cands):
            for node_id, (dbid, smiles) in zip(missing, cands):
                resolved[node_id] = (dbid, smiles)
        else:
            ambiguous.append((missing, cands))
    return resolved, ambiguous, unused, unmapped_known


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='mecddi', choices=R.DATASET_NAMES)
    p.add_argument('--dataset_type', default='cluster', choices=['cluster', 'random'],
                   help='chỉ dùng để in đường dẫn, initial/ dùng chung')
    p.add_argument('--csv', required=True, help='file CSV Drugbank_ID,SMILES')
    p.add_argument('--mode', default='fill', choices=['fill', 'rebuild'],
                   help="fill: giữ nguyên drug đã có, chỉ bù drug thiếu (mặc định). "
                        "rebuild: sinh lại toàn bộ từ CSV cho đồng nhất nguồn.")
    p.add_argument('--ambiguous', default='skip', choices=['skip', 'guess'],
                   help='chỉ dùng khi không có node map: bỏ qua, hay đoán theo thứ tự')
    p.add_argument('--node-map', default=None,
                   help='JSON {node_id: DrugBank_ID}; mặc định lấy node_map_file của dataset')
    p.add_argument('--out-dir', default=None, help='mặc định: data/initial/<dataset>')
    p.add_argument('--write', action='store_true', help='thực sự ghi file (mặc định dry run)')
    args = p.parse_args()
    R.apply_to_args(args)

    csv_pairs = read_smiles_csv(args.csv)
    known, feat_path = load_existing(args)
    print('CSV       : %d drug (%s)' % (len(csv_pairs), args.csv))
    print('Đã có     : %d/%d drug (%s)' % (len(known), args.num_ent, feat_path))

    node_map_path = args.node_map or R.initial_path(args, 'node_map_file')
    if node_map_path and os.path.exists(node_map_path):
        with open(node_map_path) as f:
            node_map = {int(k): v for k, v in json.load(f).items()}
        print('Node map  : %d id (%s)' % (len(node_map), node_map_path))
        smiles_of = dict(csv_pairs)
        resolved, no_smiles = {}, []
        for node_id, dbid in node_map.items():
            if node_id in known or node_id >= args.num_ent:
                continue
            if dbid in smiles_of:
                resolved[node_id] = (dbid, smiles_of[dbid])
            else:
                no_smiles.append((node_id, dbid))
        ambiguous, unmapped = [], []
        unused = [(d, s) for d, s in csv_pairs
                  if d not in set(node_map.values())]
        print('Suy ra    : %d drug thiếu lấy được id chính xác từ node map' % len(resolved))
        if no_smiles:
            print('Thiếu SMI : %d drug có trong node map nhưng CSV không có SMILES: %s'
                  % (len(no_smiles), ', '.join('%d/%s' % x for x in no_smiles[:8])))
    else:
        if node_map_path:
            print('Node map  : không có %s, chuyển sang dò theo thứ tự DrugBank ID'
                  % node_map_path)
        resolved, ambiguous, unused, unmapped = align(known, csv_pairs, args.num_ent)
    if unmapped:
        print('Cảnh báo  : %d drug đã có nhưng không nằm trong CSV, bị bỏ khỏi việc dò id: %s'
              % (len(unmapped), ', '.join(d for _, d in unmapped[:5])))
    n_amb = sum(len(m) for m, _ in ambiguous)
    if ambiguous:
        print('Suy ra    : %d drug thiếu xác định được id chắc chắn' % len(resolved))
    if ambiguous:
        print('Không chắc: %d drug nằm trong %d khoảng có nhiều ứng viên hơn số id thiếu:'
              % (n_amb, len(ambiguous)))
        for missing, cands in ambiguous:
            print('   node %s  <-  %d ứng viên: %s' % (
                '%d..%d' % (missing[0], missing[-1]) if len(missing) > 1 else str(missing[0]),
                len(cands), ', '.join(d for d, _ in cands)))
        print('   -> đang dùng --ambiguous %s' % args.ambiguous)
        if args.ambiguous == 'guess':
            for missing, cands in ambiguous:
                for node_id, (dbid, smiles) in zip(missing, cands):
                    resolved[node_id] = (dbid, smiles)
    print('CSV dư    : %d dòng không thuộc dataset' % len(unused))

    # ---- dựng bảng theo node id ----
    records, failed, still_missing = [], [], []
    for node_id in range(args.num_ent):
        row = None
        if args.mode == 'fill' and node_id in known:
            row = dict(known[node_id], Node_ID=node_id)
        elif node_id in resolved or node_id in known:
            if node_id in resolved:
                dbid, smiles = resolved[node_id]
            else:
                dbid, smiles = known[node_id]['DrugBank_ID'], known[node_id]['SMILES']
            fp = morgan_count_fp(smiles)
            if fp is None:
                failed.append((node_id, dbid))
            else:
                row = {'DrugBank_ID': dbid, 'Node_ID': node_id, 'SMILES': smiles,
                       'Morgan_Features': fp, 'RDKit2D_Features': rdkit2d_features(smiles)}
        if row is None:
            still_missing.append(node_id)
        else:
            records.append(row)

    print('\nKết quả   : %d/%d drug có feature (%s)'
          % (len(records), args.num_ent, 'mode=%s' % args.mode))
    if failed:
        print('   rdkit không parse được SMILES của %d drug: %s'
              % (len(failed), ', '.join('%d/%s' % f for f in failed[:8])))
    if still_missing:
        print('   vẫn thiếu %d id: %s' % (len(still_missing), R._describe_ids(still_missing)))
        if args.ambiguous == 'skip' and n_amb:
            print('   (chạy lại với --ambiguous guess nếu chấp nhận đoán %d drug)' % n_amb)

    if args.mode == 'rebuild' and known:
        changed = sum(1 for r in records if r['Node_ID'] in known
                      and not np.array_equal(r['Morgan_Features'],
                                             known[r['Node_ID']]['Morgan_Features']))
        print('   mode=rebuild: %d/%d drug cũ đổi feature (SMILES trong CSV khác nguồn cũ)'
              % (changed, len(known)))

    out_dir = args.out_dir or R.initial_dir(args)
    feat_out = os.path.join(out_dir, args.dataset_cfg['feat_file'])
    smiles_out = os.path.join(out_dir, args.dataset_cfg['smiles_file'])
    print('\nSẽ ghi    : %s\n            %s' % (feat_out, smiles_out))

    if not args.write:
        print('\n(dry run — thêm --write để ghi thật)')
        return 0

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for path in (feat_out, smiles_out):
        if os.path.exists(path):
            shutil.copy2(path, path + '.bak')
            print('backup    : %s.bak' % path)

    ### dict thuần, không phải DataFrame: pandas >= 3 lưu cột string bằng dtype `str`
    ### mới mà pandas cũ (Kaggle) không unpickle được
    records = sorted(records, key=lambda r: r['Node_ID'])
    table = {col: [r[col] for r in records] for col in
             ('DrugBank_ID', 'Node_ID', 'SMILES', 'Morgan_Features', 'RDKit2D_Features')}
    with open(feat_out, 'wb') as f:
        pkl.dump(table, f)
    with open(smiles_out, 'w') as f:
        json.dump({str(r['Node_ID']): r['SMILES'] for r in records}, f)
    print('đã ghi %d drug.' % len(records))
    print('kiểm tra lại: python check_data.py %s' % args.dataset)
    return 0


if __name__ == '__main__':
    sys.exit(main())
