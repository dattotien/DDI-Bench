"""Generate a tiny MUDI-shaped EmerGNN fixture for smoke testing.

The point is to exercise every code path - data loading, training, the eval /
metric report, wandb logging - in seconds instead of minutes, so mistakes like a
stale variable in the metric block surface before a GPU is rented.

Shapes follow the real data's conventions, which the loaders depend on:

* drug ids occupy [0, n_drugs) and KG-only entity ids come after them, because
  ``models.py`` indexes the molecular-feature table by drug id.
* DDI relation ids occupy [0, n_ddi_rel) and KG relation ids come after, with
  relation 0 meaning "No Interaction" -- ``load_data.shuffle_train`` treats
  ``r == 0`` as the negative class.
* every eval file is laid out as [forward half | inverse half]: ``metric.py``
  splits at ``len // 2`` and scores row i against row i + len/2, so the halves
  must line up pair for pair.
* files end with a trailing newline, since the loaders do
  ``f.read().split('\\n')[:-1]`` and would otherwise drop the last line.

Run via scripts/emergnn-smoke.sh, or directly:

    python scripts/make_smoke_data.py --out .smoke_data
"""

import argparse
import json
import os
import random

N_DDI_REL = 4          # 0 No Interaction, 1 Synergism, 2 Antagonism, 3 New Effect
N_KG_REL = 4
LABEL_NAMES = ['No Interaction', 'Synergism', 'Antagonism', 'New Effect']


def _write_triples(path, triples):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for h, t, r in triples:
            f.write('{} {} {}\n'.format(h, t, r))


def _paired_eval(rng, drugs, n_pairs):
    """n_pairs pairs as [forward half | inverse half], so row i pairs with row i+n."""
    forward, inverse = [], []
    for _ in range(n_pairs):
        h, t = rng.sample(drugs, 2)
        r = rng.randint(0, N_DDI_REL - 1)
        ### the inverse direction of the same pair; label may differ, which is
        ### exactly the asymmetry metric.py's options are there to score
        r_inv = r if rng.random() < 0.7 else rng.randint(0, N_DDI_REL - 1)
        forward.append((h, t, r))
        inverse.append((t, h, r_inv))
    return forward + inverse


def generate(out_dir, n_drugs, n_kg_ent, n_train, seed):
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)

    drugs = list(range(n_drugs))
    kg_ents = list(range(n_drugs, n_drugs + n_kg_ent))

    ### train DDI: a dense-ish mix of positives and negatives. Density matters:
    ### shuffle_train holds out 20% of the drugs seen in the KG and S2 keeps only
    ### pairs with *both* endpoints held out, so sparse data leaves S2 empty.
    train = []
    for _ in range(n_train):
        h, t = rng.sample(drugs, 2)
        r = 0 if rng.random() < 0.25 else rng.randint(1, N_DDI_REL - 1)
        train.append((h, t, r))
    _write_triples(os.path.join(out_dir, 'train.txt'), train)

    _write_triples(os.path.join(out_dir, 'val.txt'), _paired_eval(rng, drugs, 60))
    for split in ['s0', 's1', 's2']:
        _write_triples(os.path.join(out_dir, 'test_{}.txt'.format(split)),
                       _paired_eval(rng, drugs, 40))

    ### KG: every drug gets at least one edge, so `ddi_in_kg` is never empty
    ### (shuffle_train calls np.random.choice on it and would raise on an empty list)
    kg = []
    for d in drugs:
        kg.append((d, rng.choice(kg_ents), N_DDI_REL + rng.randint(0, N_KG_REL - 1)))
    for _ in range(n_drugs * 3):
        h, t = rng.sample(drugs + kg_ents, 2)
        kg.append((h, t, N_DDI_REL + rng.randint(0, N_KG_REL - 1)))
    _write_triples(os.path.join(out_dir, 'KG.txt'), kg)

    ### node2id / entity_drug are name -> id; relation2id is id -> name
    with open(os.path.join(out_dir, 'node2id.json'), 'w', encoding='utf-8') as f:
        json.dump({'DRUG_{}'.format(d): d for d in drugs}, f, indent=1)
    with open(os.path.join(out_dir, 'entity_drug.json'), 'w', encoding='utf-8') as f:
        json.dump({'ENT_{}'.format(e): e for e in kg_ents}, f, indent=1)
    with open(os.path.join(out_dir, 'relation2id.json'), 'w', encoding='utf-8') as f:
        rel = {str(i): LABEL_NAMES[i] for i in range(N_DDI_REL)}
        rel.update({str(N_DDI_REL + i): 'kg_relation_{}'.format(i) for i in range(N_KG_REL)})
        json.dump(rel, f, indent=1)

    return train, kg


def validate(out_dir, train, kg, n_drugs, seed):
    """Re-check the fixture against what the loaders require.

    This mirrors load_data.shuffle_train's split logic in plain Python so a
    fixture that would leave a split with no training pairs is caught here
    rather than as an empty-tensor crash on the GPU box.
    """
    problems = []

    for name in ['train.txt', 'val.txt', 'test_s0.txt', 'test_s1.txt', 'test_s2.txt', 'KG.txt']:
        path = os.path.join(out_dir, name)
        raw = open(path, encoding='utf-8').read()
        if not raw.endswith('\n'):
            problems.append('{} does not end with a newline (loader drops the last line)'.format(name))
        rows = raw.split('\n')[:-1]  ### exactly how the loaders read it
        if not rows:
            problems.append('{} parses to zero rows'.format(name))
        for i, line in enumerate(rows):
            parts = line.split()
            if len(parts) != 3 or not all(p.lstrip('-').isdigit() for p in parts):
                problems.append('{} line {} is not "h t r": {!r}'.format(name, i + 1, line))
                break
        if name.startswith(('val', 'test')) and len(rows) % 2:
            problems.append('{} has an odd row count ({}); metric.py pairs i with i+len/2'.format(name, len(rows)))

    for name in ['node2id.json', 'entity_drug.json', 'relation2id.json']:
        json.load(open(os.path.join(out_dir, name), encoding='utf-8'))

    max_drug_id = max(max(h, t) for h, t, _ in train)
    if max_drug_id >= 1710:
        problems.append('drug id {} >= 1710 rows in data/DB_molecular_feats.pkl '
                        '(models.py indexes that table by drug id)'.format(max_drug_id))

    ### mirror shuffle_train: train_ent = train endpoints, ddi_in_kg = those also in the KG,
    ### then 20% of ddi_in_kg is held out of train_ent
    train_ent = {e for h, t, _ in train for e in (h, t)}
    kg_ent = {e for h, t, _ in kg for e in (h, t)}
    ddi_in_kg = train_ent & kg_ent
    if not ddi_in_kg:
        problems.append('no train drug appears in KG.txt; shuffle_train would call '
                        'np.random.choice on an empty set')

    rng = random.Random(seed)
    positives = [(h, t) for h, t, r in train if r != 0]
    n_ent = len(ddi_in_kg)
    held_out = set(rng.sample(sorted(ddi_in_kg), n_ent - int(n_ent * 0.8)))
    kept = train_ent - held_out
    counts = {
        'S0': len(positives),
        'S1': sum(1 for h, t in positives if (h in kept) != (t in kept)),
        'S2': sum(1 for h, t in positives if h not in kept and t not in kept),
    }
    for split, n in counts.items():
        if n == 0:
            problems.append('{} would train on zero pairs -- raise --n-train or --n-drugs'.format(split))

    return problems, counts


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default='.smoke_data', help='output directory')
    p.add_argument('--n-drugs', type=int, default=60)
    p.add_argument('--n-kg-ent', type=int, default=30)
    p.add_argument('--n-train', type=int, default=900)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()

    train, kg = generate(a.out, a.n_drugs, a.n_kg_ent, a.n_train, a.seed)
    problems, counts = validate(a.out, train, kg, a.n_drugs, a.seed)

    print('wrote fixture to {}/'.format(a.out))
    print('  drugs={} kg_entities={} train_ddi={} kg_edges={}'.format(
        a.n_drugs, a.n_kg_ent, len(train), len(kg)))
    print('  approx train pairs per split: ' + ', '.join(
        '{}={}'.format(k, v) for k, v in sorted(counts.items())))
    if problems:
        print('FAILED validation:')
        for msg in problems:
            print('  - ' + msg)
        raise SystemExit(1)
    print('  validation OK')


if __name__ == '__main__':
    main()
