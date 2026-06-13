"""
Per-epoch fact/label splitter for EmerGNN training.

Ports the `shuffle_train(ratio=0.8)` logic from the reference implementation at
DDI-Bench/DDI_Ben/EmerGNN/DrugBank/load_data.py:138-177. Each epoch:

- Splits training DDI triplets into a "fact" subset (added to KG sparse
  tensor for message passing) and a "label" subset (used as supervision).
- S0: random 80/20 shuffle of all train DDI triplets.
- S1: select 80% of drugs in `ddi_in_kg` as `train_ent`. Fact requires
  both endpoints in `train_ent`; label requires exactly one. Other
  triplets are dropped from this epoch.
- S2: same fact rule as S1; label requires neither endpoint in
  `train_ent`. Other triplets are dropped.
"""

from __future__ import annotations

import json
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

import logging

logger = logging.getLogger(__name__)


class EmerGNNFactSplitter:
    def __init__(
        self,
        train_csv_path: str,
        node2id_path: str,
        kg_triplets: np.ndarray,
        scenario: str,
        ratio: float = 0.8,
        seed: Optional[int] = None,
    ):
        scenario = scenario.upper()
        if scenario not in {"S0", "S1", "S2"}:
            raise ValueError(f"Unknown scenario {scenario!r}; expected S0/S1/S2")
        self.scenario = scenario
        self.ratio = ratio
        self.seed = seed

        with open(node2id_path, "r") as f:
            raw = json.load(f)
        self.node2id = {str(k): int(v) for k, v in raw.items()}

        triplets = []
        kept_indices = []
        missing = 0

        if train_csv_path.endswith('.csv'):
            df = pd.read_csv(train_csv_path)
            for csv_idx, (d1, d2, rel) in enumerate(
                zip(df["Drug1"], df["Drug2"], df["Interaction"])
            ):
                h = self.node2id.get(str(d1))
                t = self.node2id.get(str(d2))
                if h is None or t is None:
                    missing += 1
                    continue
                triplets.append([int(h), int(t), int(rel)])
                kept_indices.append(csv_idx)
        else:
            # Assume space-separated text file of integer IDs
            with open(train_csv_path, "r") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
                    triplets.append([h, t, r])
                    kept_indices.append(idx)

        if missing:
            logger.warning(
                f"EmerGNNFactSplitter: dropped {missing} train rows with "
                f"unknown drug IDs (not in node2id)"
            )

        self.train_triplets = np.array(triplets, dtype=np.int64)
        # csv_row index for each entry in train_triplets — used to translate
        # back into dataset indices after splitting.
        self._csv_indices = np.array(kept_indices, dtype=np.int64)

        # Drugs that appear in any train DDI triplet.
        train_ent_arr = np.unique(self.train_triplets[:, :2])
        self.train_ent: set[int] = set(int(x) for x in train_ent_arr)

        # Drugs appearing in BOTH train DDI and the KG.
        kg_entities = np.unique(kg_triplets[:, :2]) if len(kg_triplets) else np.empty(0, dtype=np.int64)
        kg_entity_set = set(int(x) for x in kg_entities)
        self.ddi_in_kg: set[int] = self.train_ent & kg_entity_set

        logger.info(
            f"EmerGNNFactSplitter[{self.scenario}]: "
            f"{len(self.train_triplets)} train triplets, "
            f"{len(self.train_ent)} drugs, "
            f"{len(self.ddi_in_kg)} drugs also in KG, "
            f"ratio={self.ratio}"
        )

    def _epoch_rng(self, epoch: int) -> np.random.Generator:
        seed = (self.seed if self.seed is not None else 0) * 1_000_003 + int(epoch)
        return np.random.default_rng(seed & 0xFFFFFFFF)

    def _compute_subset(self, rng: np.random.Generator) -> set:
        """For S1/S2: keep `ratio` of `ddi_in_kg`, drop the rest from `train_ent`."""
        ddi_in_kg = sorted(self.ddi_in_kg)
        n_drop = len(ddi_in_kg) - int(len(ddi_in_kg) * self.ratio)
        if n_drop > 0:
            dropped = rng.choice(ddi_in_kg, size=n_drop, replace=False)
            return self.train_ent - set(int(x) for x in dropped)
        return set(self.train_ent)

    def shuffle(self, epoch: int) -> Tuple[np.ndarray, List[int]]:
        """
        Re-sample fact/label split for `epoch`.

        Returns:
            fact_triplets: ndarray [n_fact, 3] of (h, t, r) — to be added
                to the KG sparse tensor for message passing.
            label_indices: list[int] of CSV-row indices — the rows of the
                original train.csv that should be iterated for supervision
                this epoch.
        """
        rng = self._epoch_rng(epoch)
        n_all = len(self.train_triplets)
        if n_all == 0:
            return np.empty((0, 3), dtype=np.int64), []

        if self.scenario == "S0":
            perm = rng.permutation(n_all)
            n_fact = int(n_all * self.ratio)
            fact_local = perm[:n_fact]
            label_local = perm[n_fact:]
            fact_triplets = self.train_triplets[fact_local]
            label_indices = self._csv_indices[label_local].tolist()
            return fact_triplets, label_indices

        # S1 / S2: subset of drugs visible during fact phase.
        subset = self._compute_subset(rng)

        fact_local: List[int] = []
        label_local: List[int] = []
        for i in range(n_all):
            h = int(self.train_triplets[i, 0])
            t = int(self.train_triplets[i, 1])
            h_in = h in subset
            t_in = t in subset
            if self.scenario == "S1":
                if h_in and t_in:
                    fact_local.append(i)
                elif h_in or t_in:
                    label_local.append(i)
            else:  # S2
                if h_in and t_in:
                    fact_local.append(i)
                elif (not h_in) and (not t_in):
                    label_local.append(i)

        fact_triplets = (
            self.train_triplets[fact_local]
            if fact_local
            else np.empty((0, 3), dtype=np.int64)
        )
        label_indices = self._csv_indices[label_local].tolist() if label_local else []
        return fact_triplets, label_indices
