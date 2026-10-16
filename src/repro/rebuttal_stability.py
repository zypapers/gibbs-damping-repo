#!/usr/bin/env python3
"""Is the damping benefit finite-sample regularization, or training instability?

The condition-number test rules out one static explanation
(linear-filter Hessian conditioning) but not optimizer/training DYNAMICS. Hariri et al. (NeurIPS 2025,
"Return of ChebNet", arXiv:2506.07624) show high-order Chebyshev expansion drives ChebNet into an
UNSTABLE training regime. A fixed decaying taper shrinks the high-order terms, so it could be helping
purely as a stabilizer rather than as a capacity/regularization prior.

The two accounts make DIFFERENT predictions, and this script tests them head-on:

  INSTABILITY account: the deficit is a training-dynamics failure. ANY independent stabilizer
    (gradient clipping, smaller learning rate) should already fix it, so once a stabilizer is applied
    the damping benefit should COLLAPSE toward 0 -- the taper has nothing left to fix.

  REGULARIZATION account (ours): the deficit is capacity vs. labels. A stabilizer does not change
    capacity, so the damping benefit should SURVIVE clipping/low-LR, and should stay LABEL-DEPENDENT
    (large at small label counts, ~0 at large label counts).

Design: {none, jackson} x {baseline, grad-clip, low-LR} x {small-label, large-label}, 5 seeds,
plus training-dynamics diagnostics (gradient-norm max/mean, loss spikes, divergence) for every arm.
The label-fraction axis is what separates the two: a stabilizer story predicts the SAME benefit
collapse at both label sizes; ours predicts benefit tracks labels regardless of stabilizer.

The outcome is whatever it is. If clipping/low-LR removes the benefit, the
instability account wins.

Run:  python3 src/repro/rebuttal_stability.py                    # full (~20-40 min, 1 GPU)
      python3 src/repro/rebuttal_stability.py --quick            # fewer seeds, fast sanity check
"""
from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chebgibbsnet import ChebGibbsNet
from gpr_gnn import gcn_norm_adj
from splits import split

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
DROP = {"Cora": 0.6, "CiteSeer": 0.0, "PubMed": 0.1}


@torch.no_grad()
def order_norms(model, x, P):
    """||T_k(S) f(x)|| per polynomial order k. Hariri et al.'s instability is high-order blow-up:
    if these norms stay bounded, that failure mode is not active in this configuration."""
    model.eval()
    h = model.lin2(F.silu(model.lin1(x)))
    S = model.gso_sign * P
    Tx = [h] + ([S @ h] if model.K >= 1 else [])
    for k in range(2, model.K + 1):
        Tx.append(2.0 * (S @ Tx[-1]) - Tx[-2])
    return [float(t.norm()) for t in Tx]


def train_one(ds, d, P, damping, K, dropout, tr, va, te, seed, device,
              lr=1e-2, wd=5e-4, clip=None, epochs=300, patience=50):
    """One run. Returns (test_acc, diagnostics) where diagnostics describe TRAINING DYNAMICS.

    Protocol matches the paper's data-regime runs (epochs=300, patience=50, Adam, lr 1e-2, wd 5e-4)
    so the baseline arm is directly comparable to the published numbers."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = ChebGibbsNet(ds.num_features, 64, ds.num_classes, K, damping, 1, 1, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    bv, bt, bad = 0.0, 0.0, 0
    gnorms, losses, nclipped = [], [], 0
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(d.x, P)[tr], d.y[tr])
        loss.backward()
        # gradient norm BEFORE any clipping = the instability signal Hariri et al. describe
        gn = torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None))
        gnorms.append(float(gn)); losses.append(float(loss.detach()))
        if clip is not None:
            if float(gn) > clip: nclipped += 1        # track whether the stabilizer actually BINDS
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(d.x, P).argmax(1)
            va_ = (pred[va] == d.y[va]).float().mean().item()
            te_ = (pred[te] == d.y[te]).float().mean().item()
        if va_ > bv: bv, bt, bad = va_, te_, 0
        else:
            bad += 1
            if bad >= patience: break
    g = np.array(gnorms); L = np.array(losses)
    diverged = bool(np.any(~np.isfinite(L)))
    dL = np.diff(L) if len(L) > 1 else np.array([0.0])
    spikes = int(np.sum(dL > 0.2 * np.maximum(L[:-1], 1e-8))) if len(L) > 1 else 0
    on = order_norms(model, d.x, P)
    return 100 * bt, {"grad_max": float(np.nanmax(g)), "grad_mean": float(np.nanmean(g)),
                      "loss_spikes": spikes, "diverged": diverged, "epochs": len(L),
                      "clip_frac": (nclipped / max(len(g), 1)) if clip is not None else None,
                      "order_norms": on, "order_norm_ratio": float(on[-1] / max(on[0], 1e-12))}


def main():
    ap = argparse.ArgumentParser(description="Regularization vs. training-instability probe.")
    ap.add_argument("--datasets", nargs="+", default=["Cora", "PubMed"])
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="2 seeds, Cora only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.quick:
        args.datasets, args.seeds = ["Cora"], 2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(args.seeds))
    suf = "smoke" if args.quick else "v001"
    out = args.out or os.path.join(RES, f"rebuttal_stability_{suf}.json")

    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T

    # regimes: an INDEPENDENT stabilizer is applied in arms 2 and 3.
    # clip=0.25 is set BELOW the observed gradient scale so the stabilizer actually binds (clip_frac
    # is reported); low-LR gets 10x the epoch/patience budget so the lower rate is not confounded with
    # a shorter effective training budget.
    REGIMES = [("baseline", dict()),
               ("grad-clip", dict(clip=0.25)),
               ("low-LR", dict(lr=1e-3, epochs=3000, patience=500))]
    # label axis: small = the regime where damping reportedly helps; large = where our decay law says ~0
    FRACS = [("small-label", 0.02), ("large-label", 0.60)]

    blob = {"task": "rebuttal_stability_probe", "K": args.K, "seeds": seeds, "device": dev,
            "note": ("Separates finite-sample regularization from training instability. INSTABILITY "
                     "predicts the damping benefit collapses once an independent stabilizer (grad-clip "
                     "or low-LR) is applied; REGULARIZATION predicts it survives and stays "
                     "label-dependent. Diagnostics report pre-clip gradient norms and loss spikes."),
            "regimes": [r[0] for r in REGIMES], "label_fracs": dict(FRACS), "datasets": {}}

    print(f"Regularization vs. instability | K={args.K} | {len(seeds)} seeds | {dev}\n")
    for name in args.datasets:
        root = os.path.join(ROOT, "data", "cache", "planetoid")
        ds = Planetoid(root, name, transform=T.NormalizeFeatures())
        d = ds[0].to(dev)
        P = gcn_norm_adj(d.edge_index, d.num_nodes, dev)
        dp = DROP.get(name, 0.5)
        blob["datasets"][name] = {}
        print(f"=== {name} (dropout {dp}) ===")
        print(f"  {'label regime':>12} {'stabilizer':>11} | {'none':>13} {'jackson':>13} | "
              f"{'benefit':>8} | {'gradmax n/j':>17} {'spikes n/j':>11}")
        for fname, frac in FRACS:
            for rname, kw in REGIMES:
                arm = {}
                for damping in ("none", "jackson"):
                    accs, diags = [], []
                    for s in seeds:
                        tr, va, te = split(d.num_nodes, frac, min(0.2, (1 - frac) / 2), seed=1000 + s)
                        a, dg = train_one(ds, d, P, damping, args.K, dp,
                                          tr.to(dev), va.to(dev), te.to(dev), s, dev, **kw)
                        accs.append(a); diags.append(dg)
                    arm[damping] = {
                        "mean": float(np.mean(accs)), "std": float(np.std(accs)), "values": accs,
                        "grad_max": float(np.mean([x["grad_max"] for x in diags])),
                        "grad_mean": float(np.mean([x["grad_mean"] for x in diags])),
                        "loss_spikes": float(np.mean([x["loss_spikes"] for x in diags])),
                        "diverged_runs": int(sum(x["diverged"] for x in diags)),
                        "clip_frac": (float(np.mean([x["clip_frac"] for x in diags]))
                                      if diags[0]["clip_frac"] is not None else None),
                        "order_norm_ratio": float(np.mean([x["order_norm_ratio"] for x in diags])),
                        "order_norms": np.mean([x["order_norms"] for x in diags], axis=0).tolist(),
                    }
                pair = np.array(arm["jackson"]["values"]) - np.array(arm["none"]["values"])
                arm["benefit_paired"] = {"mean": float(pair.mean()), "std": float(pair.std()),
                                         "se": float(pair.std(ddof=1) / np.sqrt(len(pair))) if len(pair) > 1 else 0.0}
                blob["datasets"][name][f"{fname}|{rname}"] = arm
                print(f"  {fname:>12} {rname:>11} | {arm['none']['mean']:6.2f}±{arm['none']['std']:4.2f} "
                      f"{arm['jackson']['mean']:6.2f}±{arm['jackson']['std']:4.2f} | "
                      f"{arm['benefit_paired']['mean']:+8.2f} | "
                      f"{arm['none']['grad_max']:7.1f}/{arm['jackson']['grad_max']:-7.1f} "
                      f"{arm['none']['loss_spikes']:5.1f}/{arm['jackson']['loss_spikes']:-5.1f}", flush=True)
                json.dump(blob, open(out, "w"), indent=2)   # incremental: partial results survive a stop
        print()

    # ---- verdict: does an independent stabilizer remove the damping benefit? ----
    print("=" * 78)
    print("READOUT (per dataset, small-label regime -- where damping reportedly helps):")
    for name, cells in blob["datasets"].items():
        b0 = cells.get("small-label|baseline", {}).get("benefit_paired", {}).get("mean")
        bc = cells.get("small-label|grad-clip", {}).get("benefit_paired", {}).get("mean")
        bl = cells.get("small-label|low-LR", {}).get("benefit_paired", {}).get("mean")
        bg = cells.get("large-label|baseline", {}).get("benefit_paired", {}).get("mean")
        if b0 is None: continue
        keep_c = "n/a" if bc is None else f"{100*bc/b0:5.0f}%" if abs(b0) > 1e-9 else "n/a"
        keep_l = "n/a" if bl is None else f"{100*bl/b0:5.0f}%" if abs(b0) > 1e-9 else "n/a"
        print(f"  {name}: benefit baseline {b0:+.2f} | grad-clip {bc:+.2f} ({keep_c} retained) | "
              f"low-LR {bl:+.2f} ({keep_l} retained) | large-label baseline {bg:+.2f}")
    print("\n  INSTABILITY account  => benefit should COLLAPSE toward 0 under grad-clip / low-LR.")
    print("  REGULARIZATION account => benefit SURVIVES the stabilizers and stays label-dependent.")
    print("  Report whichever pattern appears, including a result against our account.")
    print("\nIS THE INSTABILITY REGIME EVEN ACTIVE? (high-order blow-up + whether the clip binds)")
    for name, cells in blob["datasets"].items():
        for key in ("small-label|baseline", "small-label|grad-clip"):
            c = cells.get(key)
            if not c: continue
            cf = c["none"].get("clip_frac")
            cf_s = "n/a" if cf is None else f"{100*cf:.0f}% of steps clipped"
            print(f"  {name} [{key}]: ||T_K h||/||T_0 h|| none {c['none']['order_norm_ratio']:.2f} / "
                  f"jackson {c['jackson']['order_norm_ratio']:.2f} | diverged none "
                  f"{c['none']['diverged_runs']}/{len(seeds)} jackson {c['jackson']['diverged_runs']}/{len(seeds)}"
                  f" | {cf_s}")
    print("  (ratio ~1 => high-order terms are NOT blowing up => Hariri-style instability is not the")
    print("   operative failure mode in this configuration.)")
    # sanity anchor: baseline small-label benefit should track the paper's published Cora @2% (+7.12)
    ref = blob["datasets"].get("Cora", {}).get("small-label|baseline", {}).get("benefit_paired", {}).get("mean")
    if ref is not None:
        print(f"\nSANITY: Cora small-label baseline benefit {ref:+.2f} vs. published Fig. 2 @2% +7.12 "
              f"(protocol matches: epochs 300, patience 50)")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {(time.time()-t0)/60:.1f} min")
