import random
import numpy as np

def _softmax(x: np.ndarray) -> np.ndarray:
    """
    Compute the softmax of a numpy array of logits.
    """
    # Compute the exponentials of x
    exp_x = np.exp(x)
    # Compute the sum of exponentials
    sum_exp_x = np.sum(exp_x, axis=0)
    # Compute softmax
    return exp_x / sum_exp_x

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute the sigmoid of a numpy array of logits.
    """
    return 1 / (1 + np.exp(-x))

def _precision(tp: int, fp: int) -> float:
    """
    Compute precision given true positives, false positives, and false negatives.
    """
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def _recall(tp: int, fn: int) -> float:
    """
    Compute recall given true positives, false positives, and false negatives.
    """
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def _f1_score(tp: int, fp: int, fn: int) -> float:
    """
    Compute F1 score given true positives, false positives, and false negatives.
    """
    if 2 * tp + fp + fn == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)

def inplace_shuffle(*lists):
    idx = []
    for i in range(len(lists[0])):
        idx.append(random.randint(0, i))
    for ls in lists:
        j = idx[i]
        ls[i], ls[j] = ls[j], ls[i]

def batch_by_num(n_batch, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])

    for i in range(n_batch):
        start = int(n_sample * i / n_batch)
        end = int(n_sample * (i+1) / n_batch)
        ret = [ls[start:end] for ls in lists]
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]

def batch_by_size(batch_size, *lists, n_sample=None):
    if n_sample is None:
        n_sample = len(lists[0])

    start = 0
    while(start < n_sample):
        end = min(n_sample, start + batch_size)
        ret = [ls[start:end] for ls in lists]
        start += batch_size
        if len(ret) > 1:
            yield ret
        else:
            yield ret[0]
        
