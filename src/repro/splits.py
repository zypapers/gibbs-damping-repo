"""Shared random train/val/test split. Kept dependency-free (numpy + torch only) so the reproduction
scripts do not import synthetic-data machinery just to obtain a split."""
import numpy as np
import torch


def split(n, ntr_frac, nval_frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ntr, nval = int(ntr_frac * n), int(nval_frac * n)
    tr = torch.zeros(n, dtype=torch.bool); va = torch.zeros(n, dtype=torch.bool); te = torch.zeros(n, dtype=torch.bool)
    tr[perm[:ntr]] = True; va[perm[ntr:ntr + nval]] = True; te[perm[ntr + nval:]] = True
    return tr, va, te
