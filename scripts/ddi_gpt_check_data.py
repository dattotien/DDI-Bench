"""Check DDI-GPT's prerequisites for each dataset, without loading torch.

Answers "will `dataset: X` actually run?" in a second, instead of finding out
after the BioGPT download and a few minutes of training:

* are the split files there, and do all seven splits parse,
* is there a drug_DDI_GPT.json covering every id the splits reference,
* does the dataset's relation count match the labels present in the files
  (a label >= num_labels is an IndexError inside CrossEntropyLoss),
* how many drugs have a name but no description text.

Usage:
    python scripts/ddi_gpt_check_data.py                # all datasets
    python scripts/ddi_gpt_check_data.py mecddi mudi
"""

import io
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GPT_DIR = os.path.join(REPO, 'DDI_Ben', 'DDI-GPT')
DATA_ROOTS = [os.path.join(GPT_DIR, 'data'), os.path.join(REPO, 'DDI_Ben', 'DDI_Ben', 'data')]
SPLITS = ['train', 'valid_S0', 'valid_S1', 'valid_S2', 'test_S0', 'test_S1', 'test_S2']

sys.path.insert(0, os.path.join(REPO, 'DDI_Ben', 'DDI_Ben'))
import dataset_registry  # noqa: E402


def _find(*parts):
    for root in DATA_ROOTS:
        p = os.path.join(root, *parts)
        if os.path.exists(p):
            return p
    return None


def check(dataset, split_strategy='cluster'):
    cfg = dataset_registry.get_config(dataset)
    print('=' * 66)
    print('{}  (task={}, num_labels={})'.format(dataset, cfg['task'], cfg['num_rel']))
    problems = []

    if cfg['task'] != dataset_registry.MULTICLASS:
        print('  skipped: multilabel dataset, handled by main_twosides.py')
        return []

    split_dir = _find('{}_{}'.format(dataset, split_strategy))
    if split_dir is None:
        problems.append('no split dir {}_{} under {}'.format(
            dataset, split_strategy, ' or '.join(os.path.relpath(r, REPO) for r in DATA_ROOTS)))
        print('  splits: MISSING')
        return problems
    print('  splits: {}'.format(os.path.relpath(split_dir, REPO)))

    ids_used, labels_used, counts = set(), set(), {}
    for split in SPLITS:
        path = os.path.join(split_dir, split + '.txt')
        if not os.path.exists(path):
            problems.append('{}: missing split file {}.txt'.format(dataset, split))
            continue
        n = 0
        with io.open(path, encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 3:
                    problems.append('{}/{}.txt: line is not "h t r": {!r}'.format(dataset, split, line[:40]))
                    break
                h, t, r = (int(x) for x in parts)
                ids_used.update((h, t))
                labels_used.add(r)
                n += 1
        counts[split] = n
    print('  pairs: ' + ', '.join('{}={}'.format(k, v) for k, v in counts.items()))

    if labels_used and max(labels_used) >= cfg['num_rel']:
        problems.append('{}: label id {} >= num_labels {} -> IndexError in CrossEntropyLoss'.format(
            dataset, max(labels_used), cfg['num_rel']))
    print('  label ids: {}..{} ({} distinct) vs num_labels {}'.format(
        min(labels_used), max(labels_used), len(labels_used), cfg['num_rel']))

    info_path = _find(dataset, 'drug_DDI_GPT.json')
    if info_path is None:
        problems.append("{}: no drug_DDI_GPT.json -- build it with "
                        "scripts/make_ddigpt_drug_info.py --dataset {}".format(dataset, dataset))
        print('  drug names: MISSING')
        return problems

    info = json.load(io.open(info_path, encoding='utf-8'))
    print('  drug names: {} ({} entries)'.format(os.path.relpath(info_path, REPO), len(info)))
    missing = sorted(i for i in ids_used if str(i) not in info)
    if missing:
        problems.append('{}: {} of {} drug ids have no entry (e.g. {}) -> KeyError mid-epoch'.format(
            dataset, len(missing), len(ids_used), missing[:8]))
    no_name = sorted(i for i in ids_used if str(i) in info
                     and not str(_field(info[str(i)], 'name')).strip())
    if no_name:
        problems.append('{}: {} drug ids have an empty name'.format(dataset, len(no_name)))
    no_desc = sorted(i for i in ids_used if str(i) in info
                     and not str(_field(info[str(i)], 'description')).strip())
    print('  coverage: {}/{} ids have a name; {} have no description text'.format(
        len(ids_used) - len(missing) - len(no_name), len(ids_used), len(no_desc)))
    if no_desc:
        print('    -> fine with drug_name_only: true; with descriptions those rows '
              'would train on an all-masked input')
    return problems


def _field(entry, key):
    if isinstance(entry, dict):
        return entry.get(key, entry.get('summary', '') if key == 'description' else '')
    idx = 0 if key == 'name' else 1
    return (list(entry) + [''])[idx]


def main():
    wanted = sys.argv[1:] or [d for d in dataset_registry.DATASET_NAMES]
    all_problems = []
    for ds in wanted:
        all_problems += check(ds)
    print('=' * 66)
    if all_problems:
        print('NOT READY:')
        for p in all_problems:
            print('  - ' + p)
        raise SystemExit(1)
    print('all checked datasets are ready to run.')


if __name__ == '__main__':
    main()
