#!/usr/bin/env python3
"""One consolidated taper table across many datasets.

The mechanism experiments in the submitted version were concentrated on a single or few citation
datasets (taper comparison: PubMed only; coefficient profiles: Cora only). One consolidated table
across several datasets -- taper, accuracy/effect and coefficient behavior together -- both
strengthens and simplifies the interpretation.

This runs the SAME taper comparison that produced the paper's Figure 1 (PubMed only) across the
citation datasets AND the clean Platonov heterophily suite, in one table:

    taper in {none, jackson(KPM), lanczos(KPM), cosine(low-pass), geometric(low-pass), increasing(anti)}

The claim under test, fixed before these runs: the KPM tapers should hold no consistent advantage
over GENERIC (non-KPM) low-pass tapers, and the anti-low-pass taper should be non-positive throughout, with the
largest degradations where the low-pass gain is largest.
If a dataset breaks that pattern, it is reported as-is.

Also records the realized-coefficient profile |c_k| = |w_k g_k| per taper per dataset, which is the
"coefficient behavior" column reported alongside accuracy (paper Figure 3 was Cora only).

Run:  python3 src/repro/rebuttal_breadth.py                  # citation + Platonov (hours; 1 GPU)
      python3 src/repro/rebuttal_breadth.py --citation-only  # fast subset (~15-25 min)
      python3 src/repro/rebuttal_breadth.py --quick          # smoke test
"""
from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chebgibbsnet import ChebGibbsNet
from gpr_gnn import gcn_norm_adj
from lowpass_vs_kpm import tapers
from splits import split

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
DROP = {"Cora": 0.6, "CiteSeer": 0.0, "PubMed": 0.1}


def run_citation(name, g, K, seeds, dev, dropout, epochs=300, patience=30):
    """Public-split citation run with a fixed forward taper g. Returns (accs, mean |c_k| profile)."""
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
    root = os.path.join(ROOT, "data", "cache", "planetoid")
    ds = Planetoid(root, name, transform=T.NormalizeFeatures())
    d = ds[0].to(dev)
    P = gcn_norm_adj(d.edge_index, d.num_nodes, dev)
    accs, profiles = [], []
    for s in seeds:
        torch.manual_seed(s); np.random.seed(s)
        m = ChebGibbsNet(ds.num_features, 64, ds.num_classes, K, "none", 1, 1, dropout, taper_arr=g).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=5e-4)
        bv, bt, bad, bestc = 0.0, 0.0, 0, None
        for ep in range(epochs):
            m.train(); opt.zero_grad()
            F.cross_entropy(m(d.x, P)[d.train_mask], d.y[d.train_mask]).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                pred = m(d.x, P).argmax(1)
                va = (pred[d.val_mask] == d.y[d.val_mask]).float().mean().item()
                te = (pred[d.test_mask] == d.y[d.test_mask]).float().mean().item()
            if va > bv:
                bv, bt, bad = va, te, 0
                bestc = (m.w * m.g).detach().abs().cpu().numpy().copy()   # realized |c_k| at best-val
            else:
                bad += 1
                if bad >= patience: break
        accs.append(100 * bt)
        if bestc is not None: profiles.append(bestc)
    prof = np.mean(profiles, axis=0) if profiles else np.zeros(K + 1)
    prof = prof / max(prof[0], 1e-12)                       # normalize to |c_0| as in the paper's Fig. 3
    return accs, prof.tolist()


def run_platonov(name, g, K, dev, n_splits, epochs=3000, patience=200):
    from platonov_breadth import (ChebGibbsSparse, build_S, metric, ROOT as PROOT, BINARY, DROP as PDROP)
    from torch_geometric.datasets import HeterophilousGraphDataset
    from torch_geometric.utils import homophily as pyg_homophily
    ds = HeterophilousGraphDataset(PROOT, name)
    d = ds[0].to(dev)
    binary = name in BINARY
    h = float(pyg_homophily(d.edge_index, d.y, method="node"))
    gso = 1.0 if h > 0.5 else -1.0
    S = build_S(d.edge_index, d.num_nodes, gso, dev)
    dp = PDROP.get(name, 0.2)
    nsp = min(n_splits, d.train_mask.shape[1])
    scores, profiles = [], []
    for i in range(nsp):
        tr, va, te = d.train_mask[:, i], d.val_mask[:, i], d.test_mask[:, i]
        torch.manual_seed(3407); np.random.seed(3407)
        m = ChebGibbsSparse(ds.num_features, 64, int(d.y.max()) + 1, K, "none", 3, gso, dp,
                            taper_arr=g).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=5e-4)
        bv, bt, bad, bestc = -1.0, 0.0, 0, None
        for ep in range(epochs):
            m.train(); opt.zero_grad()
            out = m(d.x, S)
            F.cross_entropy(out[tr], d.y[tr]).backward(); opt.step()
            m.eval()
            with torch.no_grad():
                out = m(d.x, S)
                va_ = metric(out, d.y, va, binary); te_ = metric(out, d.y, te, binary)
            if va_ > bv:
                bv, bt, bad = va_, te_, 0
                bestc = (m.w * m.g).detach().abs().cpu().numpy().copy()
            else:
                bad += 1
                if bad >= patience: break
        scores.append(100 * bt)
        if bestc is not None: profiles.append(bestc)
    prof = np.mean(profiles, axis=0) if profiles else np.zeros(K + 1)
    prof = prof / max(prof[0], 1e-12)
    return scores, prof.tolist(), ("ROC-AUC" if binary else "acc")


def main():
    ap = argparse.ArgumentParser(description="Consolidated multi-dataset taper table.")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--platonov-splits", type=int, default=10)  # 10 = the provided Platonov splits,
    # which is what results/rebuttal_breadth_v002.json and the paper report.
    ap.add_argument("--citation", nargs="+", default=["Cora", "CiteSeer", "PubMed"])
    ap.add_argument("--platonov", nargs="+",
                    default=["roman-empire", "amazon-ratings", "minesweeper", "tolokers",
                               "questions"])
    ap.add_argument("--citation-only", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.quick:
        args.citation, args.platonov, args.seeds, args.platonov_splits = ["Cora"], ["minesweeper"], 2, 1
    if args.citation_only:
        args.platonov = []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(args.seeds))
    suf = "smoke" if args.quick else "v002"
    out = args.out or os.path.join(RES, f"rebuttal_breadth_{suf}.json")

    T = tapers(args.K)
    blob = {"task": "rebuttal_breadth_taper_table", "K": args.K, "seeds": seeds,
            "platonov_splits": args.platonov_splits, "device": dev,
            "note": ("Paper Fig. 1 taper comparison (PubMed only) extended across citation + clean "
                     "Platonov heterophily datasets, with the realized |c_k|/|c_0| profile per cell. "
                     "Claim fixed before these runs: generic non-KPM low-pass largely reproduces the KPM "
                     "gain, with no consistent KPM advantage; anti-low-pass is non-positive throughout, "
                     "with the largest degradations on the citation sets. Deviations "
                     "are reported as-is."),
            "tapers": {k: np.asarray(v).tolist() for k, v in T.items()}, "datasets": {}}

    print(f"Consolidated taper table | K={args.K} | {len(seeds)} seeds | {dev}")
    print(f"tapers: {list(T)}\n")
    hdr = f"  {'dataset':>16} {'metric':>8} | " + " ".join(f"{n.split('(')[0][:9]:>10}" for n in T)
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    for name in args.citation:
        cells = {}
        for tname, g in T.items():
            accs, prof = run_citation(name, g, args.K, seeds, dev, DROP.get(name, 0.5))
            cells[tname] = {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
                            "values": accs, "coef_profile_abs_norm": prof}
        base = cells["none"]["mean"]
        for tname in cells: cells[tname]["gain_vs_none"] = cells[tname]["mean"] - base
        blob["datasets"][name] = {"metric": "acc", "split": "public", "cells": cells}
        print(f"  {name:>16} {'acc':>8} | " + " ".join(f"{cells[n]['mean']:10.2f}" for n in T), flush=True)
        json.dump(blob, open(out, "w"), indent=2)

    for name in args.platonov:
        cells = {}
        met = "acc"
        for tname, g in T.items():
            sc, prof, met = run_platonov(name, g, args.K, dev, args.platonov_splits)
            cells[tname] = {"mean": float(np.mean(sc)), "std": float(np.std(sc)),
                            "values": sc, "coef_profile_abs_norm": prof}
        base = cells["none"]["mean"]
        for tname in cells: cells[tname]["gain_vs_none"] = cells[tname]["mean"] - base
        blob["datasets"][name] = {"metric": met, "split": "platonov", "cells": cells}
        print(f"  {name:>16} {met:>8} | " + " ".join(f"{cells[n]['mean']:10.2f}" for n in T), flush=True)
        json.dump(blob, open(out, "w"), indent=2)

    print("\n" + "=" * 78)
    print("GAINS vs. no taper (consolidated effect column):")
    print(f"  {'dataset':>16} | " + " ".join(f"{n.split('(')[0][:9]:>10}" for n in T if n != "none"))
    for name, blk in blob["datasets"].items():
        c = blk["cells"]
        print(f"  {name:>16} | " + " ".join(f"{c[n]['gain_vs_none']:+10.2f}" for n in T if n != "none"))
    print("\n  Expected pattern: generic low-pass close to KPM (small |difference|); anti-low-pass "
          "non-positive throughout, with the largest degradations on the citation sets.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {(time.time()-t0)/60:.1f} min")
