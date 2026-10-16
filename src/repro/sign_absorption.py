#!/usr/bin/env python3
"""GSO-sign absorption check: S = +A~ vs S = -A~ on Cornell.

Chebyshev parity gives T_k(-S) = (-1)^k T_k(S), so with freely learned w_k the sign of the shift
operator is absorbed into the coefficients and the reachable filter class is unchanged
(Proposition 1). This runs the deployed ChebGibbsNet at both signs under an identical budget and
reports the two accuracies.

Setup follows webkb_fidelity.py: WebKB Cornell, geom-gcn splits, per-dataset dropout 0.2, fixed
model seed 3407 per split, a flat 2000-epoch budget and no early stopping, so the comparison is
"accuracy after 2000 epochs" rather than a best-validation checkpoint. The manuscript reports the
10-split comparison, where the arms differ by 1.62 points. Split 0 is also printed: there the two
score identically (75.68 vs 75.68), but a 37-node test split resolves accuracy only to 1/37, so the
exact tie on one split is not evidence that finite training makes the arms agree.

Run:  python3 src/repro/sign_absorption.py        # ~1-2 min, Cornell is 183 nodes
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chebgibbsnet import ChebGibbsNet
from gpr_gnn import gcn_norm_adj
from torch_geometric.datasets import WebKB
import torch_geometric.transforms as T

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cache", "webkb")
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")


def run_sign(name, gso_sign, damping="none", m=3, K=10, hid=64, dropout=0.2,
             lr=1e-2, wd=5e-4, epochs=2000, splits=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ds = WebKB(ROOT, name, transform=T.NormalizeFeatures())
    d = ds[0].to(device)
    P = gcn_norm_adj(d.edge_index, d.num_nodes, device)
    accs = []
    idx = range(d.train_mask.shape[1]) if splits is None else splits
    for i in idx:
        tr, te = d.train_mask[:, i], d.test_mask[:, i]
        torch.manual_seed(3407); np.random.seed(3407)
        model = ChebGibbsNet(ds.num_features, hid, ds.num_classes, K, damping, m, gso_sign, dropout).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            model.train(); opt.zero_grad()
            F.cross_entropy(model(d.x, P)[tr], d.y[tr]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(d.x, P).argmax(1)
            accs.append((pred[te] == d.y[te]).float().mean().item())
    return 100 * float(np.mean(accs)), 100 * float(np.std(accs))


def main():
    ap = argparse.ArgumentParser(description="GSO-sign absorption check (Cornell).")
    ap.add_argument("--dataset", default="Cornell")
    ap.add_argument("--damping", default="none")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"GSO-sign absorption | {args.dataset} | {args.epochs} epochs | {dev}\n")
    out = {"task": "gso_sign_absorption", "dataset": args.dataset, "epochs": args.epochs,
           "note": ("T_k(-S) = (-1)^k T_k(S), so with free w_k the sign of S is absorbed into the "
                    "coefficients and the reachable filter class is identical. Finite training need "
                    "not make the two arms agree: they start from the same raw rather than "
                    "parity-matched initialization.")}
    for scope, splits in (("split0", [0]), ("all_splits", None)):
        cell = {}
        for sign, label in ((1.0, "plus"), (-1.0, "minus")):
            mean, std = run_sign(args.dataset, sign, damping=args.damping,
                                 epochs=args.epochs, splits=splits, device=dev)
            cell[label] = {"mean": round(mean, 2), "std": round(std, 2)}
        cell["abs_diff"] = round(abs(cell["plus"]["mean"] - cell["minus"]["mean"]), 4)
        out[scope] = cell
        tag = "split 0 (single split; 1/37 resolution)" if scope == "split0" else "10 geom-gcn splits (reported)"
        print(f"  {tag}:")
        print(f"    S = +A~ : {cell['plus']['mean']:6.2f}   S = -A~ : {cell['minus']['mean']:6.2f}"
              f"   |diff| = {cell['abs_diff']:.2f}")
    path = args.out or os.path.join(RES, "sign_absorption_v001.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
