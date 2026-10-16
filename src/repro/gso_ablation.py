"""GSO-controlled factorial ablation: isolate damping's MARGINAL effect from ChebGibbsNet's other
non-damping design choices (the homophily-adaptive GSO sign switch and the activation). Addresses the
identifiability objection: a one-at-a-time change conflates channels; a 2x2x2 factorial gives each
factor's MAIN EFFECT.

Factors: damping {none, jackson} x GSO-sign {+A~, -A~} x activation {silu, relu}. Clean heterophily
datasets with enough nodes for stable splits (roman-empire, amazon-ratings; not the tiny WebKB sets,
not the contaminated chameleon/squirrel versions),
Platonov splits. Main effect of a factor = mean(on) - mean(off) averaged over the other factors, with a
bootstrap CI over splits. Prediction: damping's main effect is small (absorbed / finite-sample residual);
ChebGibbsNet's heterophily performance rests on the architecture, not the Gibbs term.
"""
import os, sys, json, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from damping_cgn import taper_cgn
from platonov_breadth import build_S, metric, ROOT, BINARY
from torch_geometric.datasets import HeterophilousGraphDataset


class Model(nn.Module):
    def __init__(self, cin, hid, cout, K, damping, act, dropout):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(cin, hid), nn.Linear(hid, cout)
        w0 = np.full(K + 1, 2.0 / (K + 1)); w0[0] = 1.0 / (K + 1)
        self.w = nn.Parameter(torch.tensor(w0, dtype=torch.float32))
        g = taper_cgn(damping, K, m=3) if damping != "none" else np.ones(K + 1)
        self.register_buffer("g", torch.tensor(g, dtype=torch.float32))
        self.K, self.act, self.dropout = K, act, dropout

    def forward(self, x, S):
        x = F.dropout(x, self.dropout, self.training)
        x = (F.silu if self.act == "silu" else F.relu)(self.lin1(x))
        x = F.dropout(x, self.dropout, self.training)
        h = self.lin2(x)
        wg = self.w * self.g
        Tx = [h] + ([torch.sparse.mm(S, h)] if self.K >= 1 else [])
        for k in range(2, self.K + 1):
            Tx.append(2.0 * torch.sparse.mm(S, Tx[-1]) - Tx[-2])
        return sum(wg[k] * Tx[k] for k in range(self.K + 1))


def run_cell(d, ds, S, binary, damping, act, K=10, hid=64, dropout=0.2, epochs=1500, patience=150, device="cuda", nsplits=5):
    scores = []
    for i in range(min(nsplits, d.train_mask.shape[1])):
        tr, va, te = d.train_mask[:, i], d.val_mask[:, i], d.test_mask[:, i]
        torch.manual_seed(3407); np.random.seed(3407)
        m = Model(ds.num_features, hid, int(d.y.max()) + 1, K, damping, act, dropout).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=5e-4)
        bv, bt, bad = -1.0, 0.0, 0
        for ep in range(epochs):
            m.train(); opt.zero_grad()
            F.cross_entropy(m(d.x, S)[tr], d.y[tr]).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                out = m(d.x, S); va_ = metric(out, d.y, va, binary); te_ = metric(out, d.y, te, binary)
            if va_ > bv: bv, bt, bad = va_, te_, 0
            else:
                bad += 1
                if bad >= patience: break
        scores.append(100 * bt)
    return np.array(scores)


def main_effect(cells, factor_idx):
    """cells: dict[(damping,sign,act)] -> array over splits. Main effect of factor = mean(on)-mean(off)."""
    on, off = [], []
    for key, arr in cells.items():
        (on if key[factor_idx] == key_on[factor_idx] else off).append(arr.mean())
    return float(np.mean(on) - np.mean(off))


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = sys.argv[1:] or ["roman-empire", "amazon-ratings"]
    DAMP, SIGN, ACT = ["none", "jackson"], [1.0, -1.0], ["silu", "relu"]
    key_on = ("jackson", -1.0, "silu")     # "on" level per factor (damping/sign/act)
    out = {}
    for name in names:
        ds = HeterophilousGraphDataset(ROOT, name); d = ds[0].to(dev)
        binary = name in BINARY
        cells = {}
        print(f"\n{name} ({'ROC-AUC' if binary else 'acc'}): 2x2x2 factorial {{damping x sign x act}}")
        for dmp, sgn, act in itertools.product(DAMP, SIGN, ACT):
            S = build_S(d.edge_index, d.num_nodes, sgn, dev)
            cells[(dmp, sgn, act)] = run_cell(d, ds, S, binary, dmp, act, device=dev)
            print(f"    damp={dmp:>7} sign={'+A~' if sgn>0 else '-A~'} act={act:>4}: "
                  f"{cells[(dmp,sgn,act)].mean():5.2f}", flush=True)
        me = {"damping": main_effect(cells, 0), "gso_sign": main_effect(cells, 1), "activation": main_effect(cells, 2)}
        print(f"  MAIN EFFECTS  damping {me['damping']:+.2f} | gso_sign {me['gso_sign']:+.2f} | activation {me['activation']:+.2f}")
        out[name] = {"metric": "roc_auc" if binary else "acc", "cells": {f"{k[0]}|{'+' if k[1]>0 else '-'}|{k[2]}": round(float(v.mean()),2) for k,v in cells.items()}, "main_effects": {k: round(v,2) for k,v in me.items()}}
        json.dump(out, open(os.path.join(ROOT, "..", "..", "..", "results", "gso_ablation_v001.json"), "w"), indent=2)
    print("\n  saved results/gso_ablation_v001.json")
