"""Kiểm tra data của từng dataset trước khi chạy train.

    python check_data.py                 # tất cả dataset, cả cluster lẫn random
    python check_data.py mecddi          # chỉ một dataset
    python check_data.py mecddi cluster  # chỉ một split type

Báo cáo: split nào thiếu, num_ent/num_rel khai báo có khớp data không,
file feature/SMILES/network có đủ cho từng model không.
Không import torch nên chạy được ở bất kỳ đâu.
"""

import os
import pickle as pkl
import sys
from argparse import Namespace

import dataset_registry as R

try:  ### console Windows mac dinh cp1252, khong in duoc tieng Viet co dau
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

OK, WARN, BAD = '  ok  ', ' warn ', ' FAIL '


def read_split(path, task):
    """Trả về (n_dòng, set drug id, max_rel) — max_rel là None với multilabel."""
    ids, max_rel, n = set(), -1, 0
    with open(path) as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) < 3:
                continue
            n += 1
            ids.add(int(parts[0]))
            ids.add(int(parts[1]))
            if task == R.MULTICLASS:
                max_rel = max(max_rel, int(parts[2]))
    return n, ids, (max_rel if task == R.MULTICLASS else None)


def check(dataset, dataset_type):
    args = Namespace(dataset=dataset, dataset_type=dataset_type)
    cfg = R.apply_to_args(args)
    print('\n=== %s / %s  (task=%s, num_ent=%d, num_rel=%d) ===' % (
        dataset, dataset_type, args.task, args.num_ent, args.num_rel))

    problems = 0
    ids_all, max_rel_all, total = set(), -1, 0
    missing_splits = []
    for split in args.splits:
        path = R.split_path(args, split)
        if not os.path.exists(path):
            missing_splits.append(split)
            continue
        n, ids, max_rel = read_split(path, args.task)
        total += n
        ids_all |= ids
        if max_rel is not None:
            max_rel_all = max(max_rel_all, max_rel)

    if missing_splits:
        print('%s splits: thiếu %s (%s)' % (BAD, ', '.join(missing_splits), R.data_dir(args)))
        problems += 1
        if len(missing_splits) == len(args.splits):
            return problems
    else:
        print('%s splits: đủ %d file, %d dòng' % (OK, len(args.splits), total))

    if ids_all:
        hi = max(ids_all) + 1
        tag = OK if hi == args.num_ent else BAD
        problems += tag == BAD
        print('%s num_ent: khai báo %d, data dùng id 0..%d' % (tag, args.num_ent, hi - 1))
    if max_rel_all >= 0:
        tag = OK if max_rel_all < args.num_rel else BAD
        problems += tag == BAD
        print('%s num_rel: khai báo %d, rel lớn nhất trong data %d' % (
            tag, args.num_rel, max_rel_all))

    # ---- initial/ files ----
    feat_path = R.initial_path(args, 'feat_file')
    if feat_path and os.path.exists(feat_path):
        with open(feat_path, 'rb') as f:
            x = pkl.load(f, encoding='utf-8')
        key, id_key = cfg['feat_key'], cfg.get('feat_id_key')
        rows = len(x if key is None else x[key])
        if id_key is None:
            tag = OK if rows >= args.num_ent else BAD
            problems += tag == BAD
            print('%s features: %d dòng, xếp theo thứ tự id (cần >= %d) — %s' % (
                tag, rows, args.num_ent, feat_path))
        else:
            covered = set(int(v) for v in x[id_key])
            missing = sorted(set(range(args.num_ent)) - covered)
            tag = OK if not missing else WARN
            problems += tag == WARN
            print('%s features: %d/%d drug có feature, đánh index theo cột %r%s — %s' % (
                tag, args.num_ent - len(missing), args.num_ent, id_key,
                '' if not missing else ' (thiếu id: %s → zero vector)' % R._describe_ids(missing),
                feat_path))
    else:
        print('%s features: thiếu %s — mọi model với --use_feat 1 sẽ lỗi' % (BAD, feat_path))
        problems += 1

    smiles_path = R.initial_path(args, 'smiles_file')
    if smiles_path and os.path.exists(smiles_path):
        try:
            id2smiles = R.load_id2smiles(args)
            missing = [i for i in range(args.num_ent) if str(i) not in id2smiles]
            tag = OK if not missing else BAD
            problems += tag == BAD
            print('%s SMILES: %d/%d drug có SMILES%s — cần cho SSI-DDI / SAGAN / TIGER' % (
                tag, args.num_ent - len(missing), args.num_ent,
                '' if not missing else ' (thiếu id: %s)' % R._describe_ids(missing)))
        except Exception as e:
            print('%s SMILES: %s' % (BAD, e))
            problems += 1
    else:
        print('%s SMILES: chưa có %s — SSI-DDI / SAGAN / TIGER không chạy được' % (
            WARN, smiles_path or 'file nào được khai báo'))

    net_path = R.initial_path(args, 'network_file')
    if net_path and os.path.exists(net_path):
        print('%s network: %s — Decagon / TIGER chạy được' % (OK, net_path))
    else:
        print('%s network: chưa có relations_2hop.txt — Decagon / TIGER không chạy được' % WARN)

    return problems


def main():
    argv = sys.argv[1:]
    datasets = [argv[0]] if argv else R.DATASET_NAMES
    types = [argv[1]] if len(argv) > 1 else ['cluster', 'random']

    problems = 0
    for d in datasets:
        for t in types:
            problems += check(d, t)
    print('\n%s' % ('Tất cả đều ổn.' if not problems else '%d vấn đề cần xử lý.' % problems))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
