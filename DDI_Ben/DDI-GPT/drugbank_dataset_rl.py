from torch.utils.data import Dataset
from transformers import AutoTokenizer
import random
import numpy as np
import copy
import torch
import json
import os
import pickle
import time
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
### DDI-GPT keeps drugbank/twosides splits under its own data/, while mecddi and
### mudi splits live in DDI_Ben's. Both roots are searched so no dataset needs its
### files duplicated; resolve_split_dir() reports which one it used.
DATA_ROOTS = [os.path.join(HERE, 'data'), os.path.join(HERE, '..', 'DDI_Ben', 'data')]

### the dataset is rebuilt every epoch, so the empty-description notice is emitted once
_WARNED_EMPTY_DESC = set()


def resolve_split_dir(args):
    """Directory holding <state>.txt for the configured dataset / split strategy."""
    roots = [getattr(args, 'data_root', '')] if getattr(args, 'data_root', '') else []
    roots += DATA_ROOTS
    folder = '{}_{}'.format(args.dataset, args.split_strategy)
    tried = []
    for root in roots:
        candidate = os.path.join(root, folder)
        tried.append(candidate)
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "no split directory '{}' for dataset '{}'. Looked in:\n  {}".format(
            folder, args.dataset, '\n  '.join(os.path.normpath(t) for t in tried)))


def resolve_drug_info_path(args):
    """Path of the {id: {name, description}} file, derived from the dataset.

    Derived rather than configured on purpose: a hardcoded ddi_dict_path is how
    you end up training on mecddi splits with drugbank's drug names and never
    notice.
    """
    for root in ([getattr(args, 'data_root', '')] if getattr(args, 'data_root', '') else []) + DATA_ROOTS:
        candidate = os.path.join(root, args.dataset, 'drug_DDI_GPT.json')
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "dataset '{}' has no drug_DDI_GPT.json (drug names/descriptions). Build one with:\n"
        "  python scripts/make_ddigpt_drug_info.py --dataset {} --info <drugbank-id-keyed>.json".format(
            args.dataset, args.dataset))


class drugbank_dataset_rl(Dataset):
    """Serves any multiclass DDI dataset (drugbank / mecddi / mudi).

    The name is kept for import compatibility; the dataset is chosen by
    ``args.dataset``, not by this module.
    """

    def __init__(self,args,state,adv=None):
        info_path = resolve_drug_info_path(args)
        with open(info_path, 'r', encoding='utf-8') as file:
            self.ddi_dict = json.load(file)

        split_path = os.path.join(resolve_split_dir(args), '{}.txt'.format(state))
        if not os.path.exists(split_path):
            raise FileNotFoundError("missing split '{}' for dataset '{}': {}".format(
                state, args.dataset, split_path))
        with open(split_path, 'r', encoding='utf-8') as file:
            self.data = [[int(num) for num in item.strip().split()] for item in file.readlines() if item.strip()]

        ### Fail here rather than with a KeyError deep inside an epoch: __getitem__
        ### looks drugs up by id, so one uncovered id kills a run after minutes.
        missing = sorted({d for row in self.data for d in row[:2] if str(d) not in self.ddi_dict})
        if missing:
            raise KeyError(
                "{} drug id(s) in {} have no entry in {} (e.g. {}). Regenerate that file so it "
                "covers every id used by the splits.".format(
                    len(missing), os.path.basename(split_path), os.path.basename(info_path),
                    missing[:10]))

        ### Report the description fallback once per dataset, and only when
        ### descriptions are actually used - otherwise it is noise repeated every epoch.
        if not getattr(args, 'drug_name_only', True) and args.dataset not in _WARNED_EMPTY_DESC:
            blank = sorted(i for i in {d for row in self.data for d in row[:2]}
                           if not str(self._entry_field(i, 'description')).strip())
            if blank:
                _WARNED_EMPTY_DESC.add(args.dataset)
                print("[{}] {} drug(s) have an empty description (e.g. {}); their prompt falls "
                      "back to the drug name, since a blank summary would leave that half of "
                      "the input fully masked.".format(args.dataset, len(blank), blank[:8]))

        if adv is not None:
            self.data = ((int(adv/len(self.data)) + 1)*self.data)[:adv]

        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
        if 'roberta' not in args.pretrained_model_path:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.state = state
        self.bg_max_length = int(200*(self.args.max_length/512))
        # self.softmaxed_sim_martix = torch.load('./data/softmaxed_sim_martix.pt')
    
    def __len__(self):
        return len(self.data)

    def _entry_field(self, drug_id, key):
        entry = self.ddi_dict[str(drug_id)]
        if isinstance(entry, dict):
            return entry.get(key, entry.get('summary', '') if key == 'description' else '')
        return (list(entry) + [''])[0 if key == 'name' else 1]

    def _name_and_summary(self, drug_id):
        entry = self.ddi_dict[str(drug_id)]
        if isinstance(entry, dict):
            name = entry.get('name', '')
            summary = entry.get('description', entry.get('summary', ''))
        else: ### legacy [name, description] pairs
            name, summary = (list(entry) + [''])[:2]
        name, summary = str(name or ''), str(summary or '')
        ### An empty description is not harmless in the description-based path: the
        ### prompt is built by unmasking only the summary tokens, so a blank one
        ### leaves that half of the input entirely masked out (and a pair where both
        ### are blank leaves nothing attended at all). Falling back to the drug name
        ### keeps the row informative instead of training on padding.
        return name, (summary or name)

    def __getitem__(self, index):
        drug1_id,drug2_id,rel_id = self.data[index]
        # drug1_accession,drug2_accession = self.id2accession[drug1_id],self.id2accession[drug2_id]
        ### Read by key, not by .values(): unpacking values() required exactly two
        ### fields in insertion order, so an entry with only a name raised
        ### ValueError and one written {description, name} silently swapped the two.
        drug1_name, drug1_summary = self._name_and_summary(drug1_id)
        drug2_name, drug2_summary = self._name_and_summary(drug2_id)
      
        prompt = "The drug-drug interactions between {} and {} is: ".format(drug1_name,drug2_name)

        if self.args.drug_name_only:
            prompt_tokenized = self.tokenizer(prompt,
                add_special_tokens=True,
                return_token_type_ids=False,
                padding="max_length",
                truncation=True,
                max_length=self.args.drug_only_max_length)
            example = [prompt_tokenized['input_ids'],prompt_tokenized['attention_mask'],rel_id]
            example = [torch.tensor(t,dtype=torch.long) for t in example]
            return example
        
        # if self.args.not_random: ### pass
        drug1_summary_0 = self.tokenizer(drug1_summary,
                                        add_special_tokens=False,
                                        return_token_type_ids=False,
                                        truncation=True,
                                        max_length=self.bg_max_length)['input_ids']
        drug1_summary_ = self.tokenizer.decode(drug1_summary_0)
        # if drug1_summary_[-1]!='.':
        #     drug1_summary_ = drug1_summary_[:-1] + '.'

        drug2_summary_0 = self.tokenizer(drug2_summary,
                                        add_special_tokens=False,
                                        return_token_type_ids=False,
                                        truncation=True,
                                        max_length=self.bg_max_length)['input_ids']
        drug2_summary_ = self.tokenizer.decode(drug2_summary_0)
        # if drug2_summary_[-1]!='.':
        #     drug2_summary_ = drug2_summary_[:-1] + '.'

        prompt = drug1_summary_ + "</s>" + drug2_summary_ 
        # + " " + prompt 
        prompt_tokenized = self.tokenizer(prompt,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            max_length=self.args.max_length)
        # prompt_tokenized['attention_mask'] = attention_mask_generation(len(drug1_summary_0), len(drug2_summary_0), len(prompt_tokenized['attention_mask']))

        input_ids = [2 for j in range(len(prompt_tokenized['input_ids'])*2)]
        input_ids[1:len(drug1_summary_0)+1] = drug1_summary_0
        input_ids[len(prompt_tokenized['input_ids']) + 2 + len(drug1_summary_0): len(prompt_tokenized['input_ids']) + 2 + len(drug1_summary_0) + len(drug2_summary_0)] = drug2_summary_0
        prompt_tokenized['input_ids'] = input_ids

        mask_ids = [0 for j in range(len(prompt_tokenized['attention_mask']) * 2)]
        mask_ids[1:len(drug1_summary_0)+1] = [1 for j in range(len(drug1_summary_0))]
        mask_ids[len(prompt_tokenized['attention_mask']) + 2 + len(drug1_summary_0): len(prompt_tokenized['attention_mask']) + 2 + len(drug1_summary_0) + len(drug2_summary_0)] = [1 for j in range(len(drug2_summary_0))]
        prompt_tokenized['attention_mask'] = mask_ids

        example = [prompt_tokenized['input_ids'],prompt_tokenized['attention_mask'],rel_id]
        example = [torch.tensor(t,dtype=torch.long) for t in example]
        return example
