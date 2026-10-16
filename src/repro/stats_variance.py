#!/usr/bin/env python3
"""Per-seed / per-split VARIANCE re-run of the paper's four headline experiments.

This script computes error bars / confidence intervals for the figures and tables. It
re-runs the exact experiments that already back the paper -- reusing the identical models, loaders,
splits, GSO conventions, damping tapers and hyperparameters from the existing repro scripts -- but
CAPTURES and dumps the full per-seed / per-split distribution (not just mean+/-std) so the paper can
add:
  * fig:lowpass  (F2)  : PubMed public split, K=10, all 6 forward tapers -> per-seed accuracies, and the
                       paired gain-vs-none with a bootstrap 95% CI.
  * fig:data   (F3)  : Cora AND PubMed, K=10, {none,jackson}, 6 train fractions -> per-seed benefit
                       (jackson-none) with mean/SEM and a paired bootstrap 95% CI per fraction.
  * Table 4 (factorial): 2x2x2 {damping x GSO-sign x activation} on roman-empire & amazon-ratings
                       (Platonov splits) -> per-SPLIT value for each of the 8 cells (mean+/-std) AND
                       each main effect as a per-split paired sample with a bootstrap 95% CI.
  * Table 3 (Platonov): the 5 CLEAN datasets, {none,jackson} -> per-SPLIT PAIRED difference so the
                       damping-effect row gets a paired SE and a bootstrap 95% CI (not per-arm std).

Everything is reused, nothing is re-implemented differently:
  - low-pass   : lowpass_vs_kpm.tapers + lowpass_vs_kpm.run_taper (already returns the per-seed array)
  - data    : chebgibbsnet.run called one-seed-at-a-time (seeds=[s]) -> exact per-seed decomposition
              of the same run() code path, dense split, per-dataset Table-7 dropout
  - factorial: gso_ablation.run_cell (already returns the per-split array); main effects computed per
              split then bootstrapped over splits
  - platonov: a per-split loop that MIRRORS platonov_breadth.run exactly (same ChebGibbsSparse model,
              build_S, metric, seed 3407 per split, K=10, m=3, epochs=3000, patience=200) but returns
              the full per-split score vector so none/jackson can be paired by split index.

Robustness: each dataset is wrapped in try/except (one failure does not abort the run); partial JSON is
flushed after every cell/fraction/dataset so a crash never loses completed work; progress prints flush
immediately; device auto-selects cuda if available. Outputs go to results/ under NEW versioned names
(never overwriting existing files):
    results/lowpass_seeds_v001.json
    results/data_regime_seeds_v001.json
    results/factorial_splits_v001.json
    results/platonov_paired_v001.json

--------------------------------------------------------------------------------------------------
EXPECTED TOTAL RUNTIME (this is a >10-min run):
  On ONE modern GPU (e.g. RTX Pro 6000), defaults (5 seeds; 10 Platonov splits):
      lowpass   :  ~15-25 min   (6 tapers x 5 seeds, PubMed dense, 300 ep)
      data      :  ~30-50 min   (Cora+PubMed x {none,jackson} x 6 fracs x 5 seeds, dense)
      factorial : ~60-120 min   (2 sets x 8 cells x 10 splits x 1500 ep, sparse)
      platonov  : ~60-150 min   (5 sets x {none,jackson} x 10 splits x 3000 ep, sparse; questions=49k)
      -----------------------------------------------------------------------------------------
      TOTAL     :  ~3-5.5 hours on one GPU (substantially longer on CPU; Platonov/factorial dominate).
  Run a subset with --stages, shrink split counts with --factorial-splits/--platonov-splits, or do a
  fast end-to-end sanity check with --smoke (writes *_smoke.json, does NOT touch the v001 files).
--------------------------------------------------------------------------------------------------

Run:  python3 src/repro/stats_variance.py                 # everything, defaults
              python3 src/repro/stats_variance.py --stages lowpass data
              python3 src/repro/stats_variance.py --smoke          # ~2-4 min sanity check
"""
from __future__ import annotations
import os
import hashlib
import sys
import json
import time
import argparse
import itertools

import numpy as np
import torch
import torch.nn.functional as F

# ---- locate the existing repro scripts (they live in src/repro/) and import them verbatim ----
HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.join(HERE, "..", "repro")
sys.path.insert(0, REPRO)
sys.path.insert(0, HERE)

# reuse EXACT models / loaders / conventions
from chebgibbsnet import run as cgn_run                                    # noqa: E402
from lowpass_vs_kpm import tapers as lowpass_tapers, run_taper            # noqa: E402
from gso_ablation import run_cell                                          # noqa: E402
from platonov_breadth import (ChebGibbsSparse, build_S, metric,           # noqa: E402
                              ROOT as PLAT_ROOT, BINARY, DROP as PLAT_DROP)
from torch_geometric.datasets import HeterophilousGraphDataset             # noqa: E402
from torch_geometric.utils import homophily as pyg_homophily              # noqa: E402

def _seed(key):
    """Deterministic bootstrap seed. Python's builtin hash() is salted per process
    (PYTHONHASHSEED), so it cannot be used for a reproducible resample."""
    return int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16)


DEV = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
os.makedirs(RESULTS, exist_ok=True)

# Per-dataset dropout for the deployed-model Planetoid runs (ChebGibbsNet Table-7 values used across the
# repo: review_experiments.py / chebgibbsnet.py / lowpass_vs_kpm.py). Documented so the run matches the paper.
PLANETOID_DROP = {"Cora": 0.6, "CiteSeer": 0.0, "PubMed": 0.1}
DATA_FRACS = [0.02, 0.05, 0.1, 0.2, 0.4, 0.6]


# ------------------------------- small stats helpers -------------------------------
def bootstrap_ci(x, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI of the mean of x (a 1-D sample)."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        v = float(x[0]) if x.size else float("nan")
        return [v, v]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, (n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
    return [float(lo), float(hi)]


def summ(arr):
    """Distributional summary of a per-seed / per-split sample (keeps the raw values)."""
    a = np.asarray(arr, dtype=float)
    n = int(a.size)
    return {
        "mean": float(a.mean()) if n else float("nan"),
        "std": float(a.std()) if n else float("nan"),                    # population std (matches repo)
        "sem": float(a.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,       # standard error of the mean
        "n": n,
        "values": [round(float(v), 4) for v in a],
    }


def paired(a, b, boot_seed=0):
    """Paired-difference summary a-b (paired element-wise) with bootstrap CI + paired SE."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    s = summ(d)
    s["ci95"] = bootstrap_ci(d, seed=boot_seed)
    return s


def dump(obj, fname):
    path = os.path.join(RESULTS, fname)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)
    return path


# =========================================================================================
# 1) LOW-PASS vs KPM (fig:lowpass / F2): PubMed public split, K=10, 6 tapers, per-seed accuracies.
# =========================================================================================
def stage_lowpass(seeds, dev, fname, epochs=300, patience=30):
    K = 10
    T = lowpass_tapers(K)      # {none, jackson(KPM), lanczos(KPM), geometric(low-pass), cosine(low-pass), increasing(anti)}
    print(f"\n[LOWPASS] PubMed public split, K={K}, {len(list(seeds))} seeds, tapers={list(T)} (dev={dev})", flush=True)
    out = {
        "task": "lowpass_vs_kpm",
        "figure": "fig:lowpass (F2)",
        "dataset": "PubMed",
        "split": "public",
        "K": K,
        "dropout": PLANETOID_DROP["PubMed"],
        "n_seeds": len(list(seeds)),
        "seeds": list(seeds),
        "note": ("deployed ChebGibbsNet (Eq.13), free w_k, fixed forward taper; per-seed test accuracy [%]. "
                 "gains_vs_none are paired by init seed (public split is fixed across tapers/seeds)."),
        "tapers": {},
    }
    per_seed = {}
    for name, g in T.items():
        accs = 100.0 * run_taper(g, K=K, dropout=PLANETOID_DROP["PubMed"], seeds=list(seeds), device=dev)
        per_seed[name] = accs
        out["tapers"][name] = summ(accs)
        print(f"  {name:>22}: {accs.mean():6.2f} +/- {accs.std():4.2f}   seeds={np.round(accs, 2)}", flush=True)
        dump(out, fname)      # flush partial after every taper
    # paired gain vs none (per seed), bootstrap CI
    none = per_seed["none"]
    out["gains_vs_none"] = {}
    for name, accs in per_seed.items():
        if name == "none":
            continue
        out["gains_vs_none"][name] = paired(accs, none, boot_seed=1)
    for name, gv in out["gains_vs_none"].items():
        print(f"    gain {name:>22}: {gv['mean']:+5.2f}  95%CI[{gv['ci95'][0]:+.2f},{gv['ci95'][1]:+.2f}]", flush=True)
    dump(out, fname)
    return out


# =========================================================================================
# 2) DATA-REGIME (fig:data / F3): Cora AND PubMed, K=10, {none,jackson}, 6 fractions, per-seed benefit.
#    Reuses chebgibbsnet.run one-seed-at-a-time -> exact per-seed decomposition of the SAME code path.
# =========================================================================================
def stage_data(seeds, dev, fname):
    K = 10
    seeds = list(seeds)
    print(f"\n[DATA] Cora+PubMed dense split, K={K}, {{none,jackson}}, fracs={DATA_FRACS}, {len(seeds)} seeds (dev={dev})", flush=True)
    out = {
        "task": "data_regime",
        "figure": "fig:data (F3)",
        "K": K,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "train_fracs": DATA_FRACS,
        "dropout": {d: PLANETOID_DROP[d] for d in ("Cora", "PubMed")},
        "note": ("ChebGibbsNet (Eq.13), dense random split (seed=1000+s), per-dataset Table-7 dropout. "
                 "benefit = jackson - none, PAIRED per seed (same split+init per seed). Means may differ "
                 "slightly from the committed single-config point estimates; this run reports the seed "
                 "distribution."),
        "datasets": {},
    }
    for ds in ["Cora", "PubMed"]:
        out["datasets"][ds] = []
        for frac in DATA_FRACS:
            per = {}
            for damping in ["none", "jackson"]:
                accs = []
                for s in seeds:
                    m, _ = cgn_run(ds, damping, K=K, dropout=PLANETOID_DROP[ds],
                                   split_kind="dense", trainfrac=frac, seeds=[s], device=dev)
                    accs.append(m)                              # run() already returns 100*acc for the single seed
                per[damping] = np.asarray(accs, dtype=float)
            benefit = paired(per["jackson"], per["none"], boot_seed=_seed(f"{ds}:{frac}"))
            entry = {"train_frac": frac,
                     "jackson": summ(per["jackson"]),
                     "none": summ(per["none"]),
                     "benefit": benefit}
            out["datasets"][ds].append(entry)
            print(f"  {ds:>6} frac={frac:<5}: jackson {per['jackson'].mean():6.2f}+/-{per['jackson'].std():4.2f}  "
                  f"none {per['none'].mean():6.2f}+/-{per['none'].std():4.2f}  benefit {benefit['mean']:+5.2f} "
                  f"95%CI[{benefit['ci95'][0]:+.2f},{benefit['ci95'][1]:+.2f}]", flush=True)
            dump(out, fname)          # flush partial after every fraction
    return out


# =========================================================================================
# 3) FACTORIAL (Table 4): 2x2x2 {damping x GSO-sign x activation} on roman-empire & amazon-ratings.
#    Reuses gso_ablation.run_cell (returns the per-split array). Main effects computed per split, then
#    bootstrapped over splits.
# =========================================================================================
FACT_DAMP = ["none", "jackson"]
FACT_SIGN = [1.0, -1.0]
FACT_ACT = ["silu", "relu"]
FACT_KEY_ON = ("jackson", -1.0, "silu")     # "on" level per factor (damping / gso_sign / activation)


def _cell_tag(k):
    return f"{k[0]}|{'+A' if k[1] > 0 else '-A'}|{k[2]}"


def stage_factorial(n_splits, dev, fname, epochs=1500, patience=150):
    print(f"\n[FACTORIAL] 2x2x2 {{damping x gso-sign x activation}} on roman-empire, amazon-ratings; "
          f"<= {n_splits} Platonov splits/cell (dev={dev})", flush=True)
    out = {
        "task": "factorial_2x2x2",
        "table": "Table 4",
        "factors": {"damping": FACT_DAMP, "gso_sign": ["+A", "-A"], "activation": FACT_ACT},
        "key_on": {"damping": FACT_KEY_ON[0], "gso_sign": "-A", "activation": FACT_KEY_ON[2]},
        "epochs": epochs, "patience": patience,
        "note": ("Reuses gso_ablation.run_cell (per-split test metric [%]). Each of the 8 cells reports "
                 "mean+/-std over Platonov splits. Main effect of a factor = mean(on cells) - mean(off "
                 "cells) computed PER SPLIT (paired), then summarised with a bootstrap 95% CI + paired SE."),
        "datasets": {},
    }
    for name in ["roman-empire", "amazon-ratings"]:
        try:
            ds = HeterophilousGraphDataset(PLAT_ROOT, name)
            d = ds[0].to(dev)
            binary = name in BINARY                 # both are multiclass -> accuracy
            ns = min(n_splits, d.train_mask.shape[1])
            print(f"  {name} ({'ROC-AUC' if binary else 'acc'}): {ns} splits x 8 cells", flush=True)
            cells = {}
            dset_out = {"metric": "roc_auc" if binary else "acc", "n_splits": ns, "cells": {}, "main_effects": {}}
            out["datasets"][name] = dset_out
            for dmp, sgn, act in itertools.product(FACT_DAMP, FACT_SIGN, FACT_ACT):
                S = build_S(d.edge_index, d.num_nodes, sgn, dev)
                arr = run_cell(d, ds, S, binary, dmp, act, K=10, epochs=epochs, patience=patience,
                               device=dev, nsplits=ns)          # per-split % array
                cells[(dmp, sgn, act)] = np.asarray(arr, dtype=float)
                dset_out["cells"][_cell_tag((dmp, sgn, act))] = summ(arr)
                print(f"    damp={dmp:>7} sign={'+A' if sgn > 0 else '-A'} act={act:>4}: "
                      f"{arr.mean():6.2f} +/- {arr.std():4.2f}", flush=True)
                dump(out, fname)        # flush partial after every cell
            # per-split main effects (paired), bootstrap over splits
            keys = list(cells.keys())
            n = int(min(len(v) for v in cells.values()))
            for fi, fname_factor in [(0, "damping"), (1, "gso_sign"), (2, "activation")]:
                on = np.stack([cells[k][:n] for k in keys if k[fi] == FACT_KEY_ON[fi]], axis=0).mean(0)
                off = np.stack([cells[k][:n] for k in keys if k[fi] != FACT_KEY_ON[fi]], axis=0).mean(0)
                eff = summ(on - off)
                eff["ci95"] = bootstrap_ci(on - off, seed=fi + 1)
                dset_out["main_effects"][fname_factor] = eff
                print(f"    MAIN EFFECT {fname_factor:>10}: {eff['mean']:+5.2f} +/- {eff['sem']:4.2f} (SE)  "
                      f"95%CI[{eff['ci95'][0]:+.2f},{eff['ci95'][1]:+.2f}]", flush=True)
            dump(out, fname)
        except Exception as e:                      # robustness: one dataset failing must not kill the run
            out["datasets"][name] = {"error": repr(e)}
            print(f"  !! {name} FAILED: {e!r} (continuing)", flush=True)
            dump(out, fname)
    return out


# =========================================================================================
# 4) PLATONOV (Table 3): 5 clean datasets, {none,jackson}, per-SPLIT paired differences.
#    Mirrors platonov_breadth.run EXACTLY (same model / build_S / metric / seed 3407 per split / HPs)
#    but returns the full per-split vector so none & jackson pair by split index.
# =========================================================================================
PLAT_SETS = ["roman-empire", "amazon-ratings", "minesweeper", "tolokers", "questions"]


def _platonov_per_split(name, damping, dev, K=10, m=3, hid=64, lr=1e-2, wd=5e-4,
                        epochs=3000, patience=200, n_splits=None):
    """Per-split test scores for one (dataset, damping) arm -- identical to platonov_breadth.run."""
    ds = HeterophilousGraphDataset(PLAT_ROOT, name)
    d = ds[0].to(dev)
    binary = name in BINARY
    h = float(pyg_homophily(d.edge_index, d.y, method="node"))
    gso = 1.0 if h > 0.5 else -1.0
    S = build_S(d.edge_index, d.num_nodes, gso, dev)
    dp = PLAT_DROP.get(name, 0.2)
    total = d.train_mask.shape[1]
    ns = total if n_splits is None else min(n_splits, total)
    scores = []
    for i in range(ns):
        tr, va, te = d.train_mask[:, i], d.val_mask[:, i], d.test_mask[:, i]
        torch.manual_seed(3407); np.random.seed(3407)
        model = ChebGibbsSparse(ds.num_features, hid, int(d.y.max()) + 1, K, damping, m, gso, dp).to(dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        bv, bt, bad = -1.0, 0.0, 0
        for ep in range(epochs):
            model.train(); opt.zero_grad()
            F.cross_entropy(model(d.x, S)[tr], d.y[tr]).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                o = model(d.x, S)
                va_ = metric(o, d.y, va, binary); te_ = metric(o, d.y, te, binary)
            if va_ > bv:
                bv, bt, bad = va_, te_, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        scores.append(100.0 * bt)
    return np.asarray(scores, dtype=float), h, ("ROC-AUC" if binary else "acc")


def stage_platonov(n_splits, dev, fname, epochs=3000, patience=200):
    print(f"\n[PLATONOV] {PLAT_SETS}, {{none,jackson}}, <= {n_splits} splits, K=10 sparse (dev={dev})", flush=True)
    out = {
        "task": "platonov_breadth_paired",
        "table": "Table 3",
        "K": 10, "damping_m_for_jackson": "jackson (Eq.10)",
        "epochs": epochs, "patience": patience,
        "note": ("Mirrors platonov_breadth.run (ChebGibbsSparse, build_S, seed 3407 per split, node-"
                 "homophily GSO sign). damp_effect = jackson - none PAIRED by split index (same split "
                 "mask + init per index), summarised with a paired SE and bootstrap 95% CI."),
        "datasets": {},
    }
    for name in PLAT_SETS:
        try:
            sj, h, met = _platonov_per_split(name, "jackson", dev, n_splits=n_splits, epochs=epochs, patience=patience)
            sn, _, _ = _platonov_per_split(name, "none", dev, n_splits=n_splits, epochs=epochs, patience=patience)
            npair = int(min(len(sj), len(sn)))
            eff = paired(sj[:npair], sn[:npair], boot_seed=_seed(name))
            out["datasets"][name] = {
                "metric": met, "homophily": round(h, 4), "n_splits": npair,
                "jackson": summ(sj), "none": summ(sn), "damp_effect": eff,
            }
            print(f"  {name:>15} {met:>8} h={h:5.3f} | J {sj.mean():6.2f}+/-{sj.std():4.2f}  "
                  f"none {sn.mean():6.2f}+/-{sn.std():4.2f}  effect {eff['mean']:+5.2f} "
                  f"(SE {eff['sem']:4.2f}) 95%CI[{eff['ci95'][0]:+.2f},{eff['ci95'][1]:+.2f}]", flush=True)
            dump(out, fname)          # flush partial after every dataset
        except Exception as e:
            out["datasets"][name] = {"error": repr(e)}
            print(f"  !! {name} FAILED: {e!r} (continuing)", flush=True)
            dump(out, fname)
    return out


# =========================================================================================
def main():
    ap = argparse.ArgumentParser(description="Per-seed / per-split variance re-run.")
    ap.add_argument("--stages", nargs="+", default=["lowpass", "data", "factorial", "platonov"],
                    choices=["lowpass", "data", "factorial", "platonov"],
                    help="which experiments to (re-)run (default: all four)")
    ap.add_argument("--lowpass-seeds", type=int, default=5, help=">=5 seeds for fig:lowpass (default 5)")
    ap.add_argument("--data-seeds", type=int, default=5, help=">=5 seeds for fig:data (default 5)")
    ap.add_argument("--factorial-splits", type=int, default=10, help="Platonov splits per factorial cell (default 10)")
    ap.add_argument("--platonov-splits", type=int, default=10, help="Platonov splits per arm for Table 3 (default 10)")
    ap.add_argument("--smoke", action="store_true",
                    help="fast end-to-end sanity check (2 seeds, 2 splits, tiny epochs) -> writes *_smoke.json")
    args = ap.parse_args()

    suf = "smoke" if args.smoke else "v001"
    files = {"lowpass": f"lowpass_seeds_{suf}.json", "data": f"data_regime_seeds_{suf}.json",
             "factorial": f"factorial_splits_{suf}.json", "platonov": f"platonov_paired_{suf}.json"}

    if args.smoke:
        m_seeds, d_seeds, f_splits, p_splits = 2, 2, 2, 2
        fact_kw = dict(epochs=60, patience=20)
        plat_kw = dict(epochs=120, patience=30)
        print("*** SMOKE MODE: tiny settings, results are NOT publication-grade (writes *_smoke.json) ***", flush=True)
    else:
        m_seeds, d_seeds = max(5, args.lowpass_seeds), max(5, args.data_seeds)
        f_splits, p_splits = args.factorial_splits, args.platonov_splits
        fact_kw, plat_kw = {}, {}

    print(f"stats_variance | device={DEV} | stages={args.stages} | results -> {RESULTS}", flush=True)
    if DEV == "cpu":
        print("WARNING: no CUDA -> this will be MUCH slower (many hours). Consider --smoke first.", flush=True)
    t0 = time.time()
    timings = {}

    if "lowpass" in args.stages:
        t = time.time(); stage_lowpass(range(m_seeds), DEV, files["lowpass"]); timings["lowpass"] = time.time() - t
        print(f"  -> wrote results/{files['lowpass']}  ({timings['lowpass']/60:.1f} min)", flush=True)
    if "data" in args.stages:
        t = time.time(); stage_data(range(d_seeds), DEV, files["data"]); timings["data"] = time.time() - t
        print(f"  -> wrote results/{files['data']}  ({timings['data']/60:.1f} min)", flush=True)
    if "factorial" in args.stages:
        t = time.time(); stage_factorial(f_splits, DEV, files["factorial"], **fact_kw); timings["factorial"] = time.time() - t
        print(f"  -> wrote results/{files['factorial']}  ({timings['factorial']/60:.1f} min)", flush=True)
    if "platonov" in args.stages:
        t = time.time(); stage_platonov(p_splits, DEV, files["platonov"], **plat_kw); timings["platonov"] = time.time() - t
        print(f"  -> wrote results/{files['platonov']}  ({timings['platonov']/60:.1f} min)", flush=True)

    total = time.time() - t0
    print("\n================ DONE ================", flush=True)
    for k, v in timings.items():
        print(f"  {k:>10}: {v/60:6.1f} min  -> results/{files[k]}", flush=True)
    print(f"  {'TOTAL':>10}: {total/60:6.1f} min", flush=True)


if __name__ == "__main__":
    main()
