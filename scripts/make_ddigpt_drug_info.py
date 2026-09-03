"""Build DDI-GPT's drug_DDI_GPT.json for a dataset.

DDI-GPT prompts BioGPT with a drug's *name* (and optionally its description), and
looks them up by the numeric drug id used in the split files:

    {"0": {"name": ..., "description": ...}, "1": {...}, ...}

Descriptions, though, are collected per DrugBank ID (``{"DB00001": {...}}``).
MecDDI and MUDI ship ``data/initial/<dataset>/node2drugbank.json`` mapping node
id -> DrugBank ID, so this script bridges the two and reports exactly which ids
end up without text -- a missing id is a KeyError mid-epoch in the dataset
loader, so it is much better to see it here.

Usage:

    python scripts/make_ddigpt_drug_info.py --dataset mecddi --info mecddi_drug_info.json
    python scripts/make_ddigpt_drug_info.py --dataset mudi   --info mudi_drug_info.json

Writes DDI_Ben/DDI-GPT/data/<dataset>/drug_DDI_GPT.json, which is where
main_drugbank.py looks for it.
"""

import argparse
import io
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE_MAP = os.path.join(REPO, 'DDI_Ben', 'DDI_Ben', 'data', 'initial', '{}', 'node2drugbank.json')
SPLIT_DIR = os.path.join(REPO, 'DDI_Ben', 'DDI_Ben', 'data', '{}_{}')
OUT = os.path.join(REPO, 'DDI_Ben', 'DDI-GPT', 'data', '{}', 'drug_DDI_GPT.json')


def _load(path):
    with io.open(path, encoding='utf-8') as f:
        return json.load(f)


def ids_used_by_splits(dataset, dataset_type):
    """Every drug id the split files actually reference."""
    d = SPLIT_DIR.format(dataset, dataset_type)
    if not os.path.isdir(d):
        return None, d
    used = set()
    for name in os.listdir(d):
        if not name.endswith('.txt'):
            continue
        with io.open(os.path.join(d, name), encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                if len(parts) == 3:
                    used.add(int(parts[0]))
                    used.add(int(parts[1]))
    return used, d


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', required=True, choices=['mecddi', 'mudi'],
                   help='drugbank already ships its own drug_DDI_GPT.json')
    p.add_argument('--info', required=True,
                   help='JSON keyed by DrugBank ID: {"DB00001": {"name":..., "description":...}}')
    p.add_argument('--dataset-type', default='cluster', choices=['cluster', 'random'])
    p.add_argument('--name-key', default='name')
    p.add_argument('--description-key', default='description')
    a = p.parse_args()

    info = _load(a.info)
    node_map_path = NODE_MAP.format(a.dataset)
    if not os.path.exists(node_map_path):
        raise SystemExit('missing node id -> DrugBank map: {}'.format(node_map_path))
    node2db = _load(node_map_path)

    used, split_dir = ids_used_by_splits(a.dataset, a.dataset_type)
    if used is None:
        raise SystemExit('missing split files: {}'.format(split_dir))

    out = {}
    no_mapping, no_info, empty_desc = [], [], []
    for node_id in sorted(used):
        db_id = node2db.get(str(node_id))
        if db_id is None:
            no_mapping.append(node_id)
            continue
        entry = info.get(db_id)
        if entry is None:
            no_info.append(node_id)
            continue
        name = str(entry.get(a.name_key, '') or '').strip()
        desc = str(entry.get(a.description_key, '') or '').strip()
        if not desc:
            empty_desc.append(node_id)
        ### key order matters only for readability now - the loader reads by key
        out[str(node_id)] = {'name': name or db_id, 'description': desc}

    out_path = OUT.format(a.dataset)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with io.open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print('dataset       : {} ({})'.format(a.dataset, a.dataset_type))
    print('ids in splits : {}'.format(len(used)))
    print('written       : {} entries -> {}'.format(len(out), os.path.relpath(out_path, REPO)))
    print('no node->DrugBank mapping : {}{}'.format(len(no_mapping), _sample(no_mapping)))
    print('mapped but no info entry  : {}{}'.format(len(no_info), _sample(no_info)))
    print('name present but no text  : {}{}  (fine with drug_name_only: true)'.format(
        len(empty_desc), _sample(empty_desc)))

    if no_mapping or no_info:
        print()
        print('INCOMPLETE: {} of {} ids have no name at all. Training would raise a '
              'KeyError the first time one is sampled -- collect these before running.'.format(
                  len(no_mapping) + len(no_info), len(used)))
        raise SystemExit(1)
    print()
    print('OK: every id used by the splits has a name.')


def _sample(ids, limit=8):
    if not ids:
        return ''
    head = ', '.join(str(i) for i in ids[:limit])
    return '  e.g. [{}{}]'.format(head, ', ...' if len(ids) > limit else '')


if __name__ == '__main__':
    main()
