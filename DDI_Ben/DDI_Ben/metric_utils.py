"""Small pure helpers used by metric.py.

They used to live in utils.py, but utils.py imports the whole model stack
(torchdrug, torch_scatter, ...) and fcntl, so importing metric.py from anywhere
outside the DDI_Ben training environment - DDI-GPT reuses it to score MUDI the
same way - dragged in dependencies that scoring does not need.
"""

import numpy as np


def _softmax(x):
    """Numerically stable softmax over a 1-D score vector."""
    x = np.array(x, dtype=np.float64)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _precision(TP, FP):
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0


def _recall(TP, FN):
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0


def _f1_score(TP, FP, FN):
    p = _precision(TP, FP)
    r = _recall(TP, FN)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
