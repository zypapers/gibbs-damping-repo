"""The non-vanishing Gibbs overshoot.

Falsifiable prediction: the degree-K truncated Chebyshev projection of an ideal
spectral step overshoots the jump by a fraction that does NOT vanish as K grows,
converging to the Gibbs-Wilbraham constant  G = Si(pi)/pi - 1/2 = 0.0894899...,
independent of K (once large) and of the interior jump location lambda0.

Cross-checks built in (adversarial, must agree):
  (i)  chebval(x, c)  vs  c0 + sum c_k cos(k theta)      -> machine precision
  (ii) closed-form coeffs vs independent DCT projection  -> O(1/N), small
  (iii) overshoot(K->inf) -> G                            -> the physics claim
  (iv) overshoot independent of lambda0                   -> universality
  (v)  overshoot non-vanishing (does not -> 0)            -> the whole point

Outputs: results/theory_step1_gibbs_overshoot_v001.json
         figures/theory_step1_gibbs_overshoot.{pdf,png}
Run:     python3 src/theory/step1_gibbs_overshoot.py
"""
from __future__ import annotations

import json
import os
import numpy as np
from numpy.polynomial.chebyshev import chebval

import cheb_step as cs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")

G = cs.gibbs_constant()

K_LIST = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
LAM0_LIST = [0.4, 0.8, 1.0, 1.2, 1.6]


def check_i_eval_paths():
    """chebval(T_k recurrence) vs cos(k theta) evaluation of the same polynomial."""
    lam0, K = 1.0, 200
    c = cs.step_cheb_coeffs(lam0, K)
    x = np.linspace(-0.999, 0.999, 2001)
    via_chebval = chebval(x, c)                       # T_k via recurrence
    via_cos = cs.partial_sum_theta(c, np.arccos(x))   # cos(k theta)
    return float(np.max(np.abs(via_chebval - via_cos)))


def check_ii_coeff_paths():
    """closed-form coeffs vs independent DCT projection at large N."""
    lam0, K, N = 1.0, 64, 200_000
    c_closed = cs.step_cheb_coeffs(lam0, K)
    c_dct = cs.step_cheb_coeffs_dct(lam0, K, N=N)
    return float(np.max(np.abs(c_closed - c_dct))), N


def main():
    report = {
        "task": "1.1", "step": 1,
        "claim": "non-vanishing Gibbs overshoot of the truncated Chebyshev "
                 "projection of a spectral step",
        "gibbs_constant_target": G,
        "convention": "normalized Laplacian, lambda in [0,2], x=lambda-1",
        "K_list": K_LIST, "lambda0_list": LAM0_LIST,
    }

    # --- cross-check (i): evaluation paths ---
    d_eval = check_i_eval_paths()
    # --- cross-check (ii): coefficient paths ---
    d_coeff, N_dct = check_ii_coeff_paths()

    # --- overshoot table over (K, lambda0) ---
    table = []
    for lam0 in LAM0_LIST:
        for K in K_LIST:
            m = cs.measure_overshoot(lam0, K)
            table.append(m)
    report["overshoot_table"] = table

    # overshoot at the largest K, per lambda0
    Kmax = K_LIST[-1]
    over_at_Kmax = {m["lambda0"]: m["overshoot"] for m in table if m["K"] == Kmax}
    over_vals_Kmax = np.array(list(over_at_Kmax.values()))
    over_lam1 = over_at_Kmax[1.0]

    # --- verdicts ---
    tol_eval = 1e-10
    tol_coeff = 5e-4
    tol_G = 2e-3
    tol_universal = 2e-3
    checks = {
        "i_eval_paths_agree": {
            "value": d_eval, "tol": tol_eval, "pass": d_eval < tol_eval,
            "desc": "chebval vs cos(k theta) max|diff|"},
        "ii_coeff_paths_agree": {
            "value": d_coeff, "tol": tol_coeff, "pass": d_coeff < tol_coeff,
            "N": N_dct, "desc": "closed-form vs DCT coeff max|diff|"},
        "iii_overshoot_to_G": {
            "value": over_lam1, "target": G, "abs_err": abs(over_lam1 - G),
            "tol": tol_G, "pass": abs(over_lam1 - G) < tol_G,
            "desc": f"overshoot at K={Kmax}, lambda0=1.0 vs G"},
        "iv_universal_in_lambda0": {
            "value": float(over_vals_Kmax.std()), "tol": tol_universal,
            "pass": float(over_vals_Kmax.std()) < tol_universal,
            "per_lambda0": {str(k): v for k, v in over_at_Kmax.items()},
            "desc": f"std of overshoot across lambda0 at K={Kmax}"},
        "v_non_vanishing": {
            "value": over_lam1, "floor": 0.085, "pass": over_lam1 > 0.085,
            "desc": f"overshoot at K={Kmax} stays well above 0 (does not vanish)"},
    }
    report["checks"] = checks
    report["all_pass"] = all(c["pass"] for c in checks.values())
    # undershoot symmetry (extra evidence, not a gate)
    und_lam1 = [m["undershoot"] for m in table if m["K"] == Kmax and m["lambda0"] == 1.0][0]
    report["undershoot_at_Kmax_lambda1"] = und_lam1

    os.makedirs(RESULTS, exist_ok=True)
    out_json = os.path.join(RESULTS, "theory_step1_gibbs_overshoot_v001.json")
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2)

    _make_figure(table)

    # --- console report ---
    print("=" * 72)
    print("Non-vanishing Gibbs overshoot")
    print("=" * 72)
    print(f"Gibbs-Wilbraham target  G = Si(pi)/pi - 1/2 = {G:.10f}")
    print("-" * 72)
    for name, c in checks.items():
        status = "PASS" if c["pass"] else "FAIL"
        print(f"  [{status}] {name}: {c['desc']}")
        if name == "iii_overshoot_to_G":
            print(f"          measured={c['value']:.8f}  target={G:.8f}  "
                  f"|err|={c['abs_err']:.2e}  (tol {c['tol']:.0e})")
        elif name == "iv_universal_in_lambda0":
            print(f"          per-lambda0 overshoot @K={Kmax}: " +
                  ", ".join(f"{k}:{v:.6f}" for k, v in c['per_lambda0'].items()))
            print(f"          std={c['value']:.2e} (tol {c['tol']:.0e})")
        else:
            print(f"          value={c['value']:.3e}  tol={c['tol'] if 'tol' in c else c.get('floor'):.0e}")
    print("-" * 72)
    print("  overshoot vs K at lambda0=1.0 (approach to G):")
    for m in table:
        if m["lambda0"] == 1.0:
            print(f"    K={m['K']:5d}  overshoot={m['overshoot']:.8f}  "
                  f"(over/G={m['overshoot']/G:.5f})  lambda_peak={m['lambda_at_overshoot']:.5f}")
    print(f"  undershoot @K={Kmax},lambda0=1.0 = {und_lam1:.8f}  (expect ~ -G)")
    print("-" * 72)
    print(f"  ALL PASS: {report['all_pass']}")
    print(f"  results -> {os.path.relpath(out_json, ROOT)}")
    print(f"  figure  -> figures/theory_step1_gibbs_overshoot.pdf/.png")
    return report


def _make_figure(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # colorblind-safe (Wong)
    blue, orange, green, vermillion = "#0072B2", "#E69F00", "#009E73", "#D55E00"
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    # LEFT: partial sums S_K(lambda) at lambda0=1.0 for a few K, with overshoot lines
    lam0 = 1.0
    lam = np.linspace(0.001, 1.999, 4000)
    axL.plot(lam, (lam < lam0).astype(float), color="k", lw=1.2, ls="--",
             label="ideal step")
    for K, col in [(16, green), (64, orange), (256, vermillion)]:
        c = cs.step_cheb_coeffs(lam0, K)
        axL.plot(lam, cs.partial_sum_lambda(c, lam), color=col, lw=1.0,
                 label=f"K={K}")
    axL.axhline(1 + G, color=blue, lw=0.8, ls=":", label=f"1+G={1+G:.4f}")
    axL.axhline(-G, color=blue, lw=0.8, ls=":")
    axL.set_xlabel(r"$\lambda$ (normalized-Laplacian spectrum)")
    axL.set_ylabel(r"degree-$K$ response")
    axL.set_title(r"Ringing at a step ($\lambda_0=1$)", fontsize=9)
    axL.legend(fontsize=6, loc="center left", framealpha=0.9)
    axL.set_xlim(0, 2)

    # RIGHT: overshoot vs K (approach to G), all lambda0
    cols = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
    for lam0, col in zip(cs.LAM0_LIST if hasattr(cs, "LAM0_LIST") else LAM0_LIST, cols):
        Ks = [m["K"] for m in table if m["lambda0"] == lam0]
        ov = [m["overshoot"] for m in table if m["lambda0"] == lam0]
        axR.semilogx(Ks, ov, "o-", ms=3, lw=0.9, color=col, label=fr"$\lambda_0$={lam0}")
    axR.axhline(G, color="k", lw=1.0, ls="--", label=f"G={G:.5f}")
    axR.set_xlabel(r"degree $K$")
    axR.set_ylabel("peak overshoot")
    axR.set_title("Overshoot does not vanish", fontsize=9)
    axR.legend(fontsize=6, ncol=2, framealpha=0.9)
    axR.set_ylim(0.084, 0.095)

    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, "theory_step1_gibbs_overshoot.pdf"))
    fig.savefig(os.path.join(FIGURES, "theory_step1_gibbs_overshoot.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
