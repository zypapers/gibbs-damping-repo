#!/usr/bin/env python3
"""Canonical derived statistics: ONE source for every CI quoted in Appendices D.2 and D.3.

Those CIs must agree exactly wherever they are quoted. They previously did not, because
each was bootstrapped from a generator whose draws were consumed in a different order. Here every
interval gets its own freshly seeded generator keyed by name, so the value depends only on the data and
the key, not on evaluation order. Writes results/rebuttal_stats_v001.json.

Run: python3 src/repro/rebuttal_stats.py
"""
from __future__ import annotations
import os, json, hashlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
B = 20000


def boot(x, key):
    """Mean + percentile bootstrap CI, seeded from `key` so it is reproducible and order-independent."""
    x = np.asarray(x, float)
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(x)), "key": key}


def main():
    br = json.load(open(os.path.join(RES, "rebuttal_breadth_v002.json")))
    st = json.load(open(os.path.join(RES, "rebuttal_stability_v001.json")))
    pb = json.load(open(os.path.join(RES, "rebuttal_planted_band_v002.json")))
    K = br["K"]
    out = {"note": ("Canonical derived statistics. Every CI is a percentile bootstrap with B=20000 and a "
                    "generator seeded from the quoted key, so the quoted intervals cannot drift apart."),
           "B": B, "K": K}

    # ---- breadth: Jackson-cosine contrast + high-order mass fraction, per dataset ----
    out["breadth"] = {}
    for name, blk in br["datasets"].items():
        c = blk["cells"]
        d = np.array(c["jackson(KPM)"]["values"]) - np.array(c["cosine(low-pass)"]["values"])
        mhi = {}
        for t in c:
            q = np.array(c[t]["coef_profile_abs_norm"]) ** 2
            mhi[t] = float(q[K // 2 + 1:].sum() / q.sum())
        out["breadth"][name] = {
            "metric": blk["metric"], "none": c["none"]["mean"],
            "gain": {t: c[t]["gain_vs_none"] for t in c},
            "jackson_minus_cosine": boot(d, f"breadth|{name}|jackson-cosine"),
            "M_hi": mhi,
        }

    # ---- stability: per-cell paired benefit + taper x control interaction ----
    out["stability"] = {}
    for ds in st["datasets"]:
        for cell, c in st["datasets"][ds].items():
            d = np.array(c["jackson"]["values"]) - np.array(c["none"]["values"])
            out["stability"][f"{ds}|{cell}"] = {
                "none": c["none"]["mean"], "jackson": c["jackson"]["mean"],
                "benefit": boot(d, f"stability|{ds}|{cell}|benefit"),
                "clip_frac": {"none": c["none"].get("clip_frac"),
                                "jackson": c["jackson"].get("clip_frac")},
            }
    out["interaction"] = {}
    for ds in ("Cora", "PubMed"):
        base = (np.array(st["datasets"][ds]["small-label|baseline"]["jackson"]["values"])
                - np.array(st["datasets"][ds]["small-label|baseline"]["none"]["values"]))
        for reg in ("grad-clip", "low-LR"):
            cur = (np.array(st["datasets"][ds][f"small-label|{reg}"]["jackson"]["values"])
                   - np.array(st["datasets"][ds][f"small-label|{reg}"]["none"]["values"]))
            out["interaction"][f"{ds}|{reg}"] = boot(cur - base, f"interaction|{ds}|{reg}")

    # ---- planted band ----
    raw = pb["arms"]["raw"]
    fr = (np.array(raw["frozen"]["jackson"]) - np.array(raw["frozen"]["none"])) * 100
    ols = np.abs(np.array(raw["free_ols"]["jackson"]) - np.array(raw["free_ols"]["none"]))
    out["planted"] = {
        "frozen_response": pb["arms"]["frozen_response"],
        "frozen_task_effect": boot(fr, "planted|frozen|task-effect"),
        # three distinct quantities; the first is accuracy, the other two the realized response
        # and coefficients, which is what D.1 quotes.
        "free_ols_max_abs_acc_diff": float(ols.max()),
        "free_ols_max_abs_fit_diff": pb["free_coefficient_agreement"]["fit_max_over_seeds"],
        "free_ols_max_abs_coef_diff": pb["free_coefficient_agreement"]["coef_max_over_seeds"],
        "lam0": pb["lam0"], "K": pb["K"], "n": pb["n"], "seeds": len(raw["frozen"]["none"]),
    }
    # the O(1/K) band actually used by the diagnostic (half-width in lambda)
    lam0, Kp = pb["lam0"], pb["K"]
    out["planted"]["band_half_width_lambda"] = float(np.pi / Kp * np.sqrt(lam0 * (2 - lam0)))

    path = os.path.join(RES, "rebuttal_stats_v001.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"wrote {path}")
    print("\nJackson - cosine (paired, per dataset):")
    for n, v in out["breadth"].items():
        j = v["jackson_minus_cosine"]
        print(f"  {n:>16} {j['mean']:+6.2f} [{j['lo']:+.2f},{j['hi']:+.2f}]")
    print("\nStability benefit (small-label):")
    for k, v in out["stability"].items():
        if "small-label" in k:
            b = v["benefit"]; print(f"  {k:>28} none {v['none']:6.2f} jack {v['jackson']:6.2f} "
                                    f"benefit {b['mean']:+.2f} [{b['lo']:+.2f},{b['hi']:+.2f}]")
    print("\nInteractions:")
    for k, v in out["interaction"].items():
        print(f"  {k:>18} {v['mean']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}]")
    p = out["planted"]
    print(f"\nPlanted: band half-width {p['band_half_width_lambda']:.4f} in lambda "
          f"(lam0={p['lam0']}, K={p['K']}); frozen task effect "
          f"{p['frozen_task_effect']['mean']:+.2f} [{p['frozen_task_effect']['lo']:+.2f},"
          f"{p['frozen_task_effect']['hi']:+.2f}]; free-OLS max|diff| "
          f"acc {p['free_ols_max_abs_acc_diff']:.2e}, fit {p['free_ols_max_abs_fit_diff']:.2e}, "
          f"coef {p['free_ols_max_abs_coef_diff']:.2e}")


if __name__ == "__main__":
    main()
