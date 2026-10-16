#!/usr/bin/env python3
"""Band-edge positive control: a task on which classical ringing is present by construction.

One way to test the ringing premise is to estimate the task-relevant spectral response directly and
ask whether real datasets contain a sharp discontinuity with meaningful spectral energy in the
Gibbs-sensitive region. Another is to construct a synthetic task with a known Gibbs-sensitive
response as a positive control.

This is that positive control. We build a task whose population-optimal UNRESTRICTED LINEAR response
under squared loss is a sharp step -- optimal for linear denoising of the latent score, not the Bayes
rule for the binary labels -- so classical Gibbs ringing genuinely applies, and then ask what the
damping taper does in each coefficient regime. The design makes all three of the paper's claims falsifiable on one task:

  ARM 1 (FROZEN coefficients = classical setting). Coefficients are the degree-K Chebyshev projection
    of the ideal step. Ringing is real here, so an undamped filter should overshoot near the band edge
    and Jackson damping should remove the overshoot. If this fails, our measurement is broken and the
    whole positive control is uninformative -- it is the control on the control.

  ARM 2 (FREE coefficients, unregularized). The coefficients are fit to the continuous latent score y_cont, not the binary labels. Because the fit is
    linear least squares in the coefficients, a positive taper is an exact reparameterization: the
    fitted response, and hence the accuracy, must be IDENTICAL with and without the taper. This is
    Proposition 1 made numerically exact rather than argued.

Reporting: realized response overshoot near the band edge, the share of squared response error inside
the O(1/K) ringing band, and test accuracy per arm. Whatever the arms show is what we report.

Run:  python3 src/repro/rebuttal_planted_band.py            # ~2-5 min, CPU (fp64 eigendecomposition)
      python3 src/repro/rebuttal_planted_band.py --quick
"""
from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src", "synth"))
sys.path.insert(0, os.path.join(ROOT, "src", "score"))
from damping_cgn import taper_cgn                                    # noqa: E402
import graph_spectrum as gs                                          # noqa: E402
from csbm import generate_sbm                                        # noqa: E402

RES = os.path.join(ROOT, "results")


def cheb_project_step(lam0, K, n_quad=4000):
    """Chebyshev coefficients (on lambda in [0,2], mu = lambda-1) of the ideal low-pass step
    h*(lambda) = 1 for lambda <= lam0 else 0. Classical projection via Chebyshev-Gauss quadrature."""
    th = (np.arange(n_quad) + 0.5) * np.pi / n_quad
    mu = np.cos(th)                       # mu in [-1,1]
    lam = mu + 1.0                        # lambda in [0,2]
    f = (lam <= lam0).astype(float)
    c = np.empty(K + 1)
    for k in range(K + 1):
        c[k] = (2.0 / n_quad) * np.sum(f * np.cos(k * th))
    c[0] *= 0.5
    return c


def response(c, lam):
    """Realized spectral response sum_k c_k T_k(lambda-1) evaluated on a lambda grid."""
    mu = np.clip(lam - 1.0, -1.0, 1.0)
    th = np.arccos(mu)
    return np.array([np.sum(c * np.cos(np.arange(len(c)) * t)) for t in th])


def ringing_diagnostics(c, lam0, K, grid=None):
    """Overshoot near the band edge + share of squared error inside the O(1/K) band."""
    if grid is None:
        grid = np.linspace(0.0, 2.0, 4001)
    h = response(c, grid)
    ideal = (grid <= lam0).astype(float)
    err2 = (h - ideal) ** 2
    # the O(1/K) ringing band around the jump (width scaled by the local Chebyshev metric)
    half = np.pi / K * np.sqrt(max(lam0 * (2 - lam0), 1e-12))
    inband = np.abs(grid - lam0) <= half
    peak = float(h[grid <= lam0 - 1e-9].max()) if np.any(grid <= lam0 - 1e-9) else float("nan")
    return {"overshoot": peak - 1.0, "max_response": float(h.max()),
            "band_half_width": float(half),
            "err_share_in_band": float(err2[inband].sum() / max(err2.sum(), 1e-30)),
            "band_frac_of_axis": float(inband.mean())}


def cheb_basis_apply(S, x, K):
    """[T_0(S)x, ..., T_K(S)x] as columns."""
    cols = [x, S @ x] if K >= 1 else [x]
    for _ in range(2, K + 1):
        cols.append(2.0 * (S @ cols[-1]) - cols[-2])
    return np.stack(cols[:K + 1], axis=1)


def main():
    ap = argparse.ArgumentParser(description="Planted band-edge positive control.")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--lam0", type=float, default=0.6, help="planted band edge (normalized Laplacian)")
    ap.add_argument("--snr", type=float, default=1.0, help="feature SNR: x = y_cont + noise/snr")
    ap.add_argument("--seeds", type=int, default=10)  # 10 is what the shipped
    # results/rebuttal_planted_band_v002.json and the manuscript use; the previous default of 5
    # silently produced different numbers for anyone running the documented command.
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.quick:
        args.n, args.seeds = 1200, 2
    suf = "smoke" if args.quick else "v002"
    out = args.out or os.path.join(RES, f"rebuttal_planted_band_{suf}.json")
    TAPERS = {"none": np.ones(args.K + 1), "jackson": taper_cgn("jackson", args.K),
              "lanczos": taper_cgn("lanczos", args.K, m=3)}

    blob = {"task": "planted_band_positive_control", "n": args.n, "K": args.K, "lam0": args.lam0,
            "snr": args.snr, "seeds": list(range(args.seeds)),
            "note": ("Positive control on a task whose population-optimal unrestricted linear response "
                     "under squared loss is a sharp step -- optimal for linear denoising of the latent "
                     "score, not the Bayes rule for the binary labels -- so classical ringing applies. Arm 1 frozen Chebyshev-projection coefficients "
                     "(ringing real, damping should help); arm 2 free coefficients fit by ordinary "
                     "least squares (a positive taper is an exact reparameterization, so accuracy must "
                     "be identical)."),
            "arms": {}}

    print(f"PLANTED-BAND POSITIVE CONTROL | n={args.n} K={args.K} lam0={args.lam0} "
          f"seeds={args.seeds}\n")

    # ---------- ARM 1: frozen coefficients (classical regime) ----------
    print("[ARM 1] FROZEN coefficients = Chebyshev projection of the ideal step")
    print(f"  {'taper':>9} {'overshoot':>10} {'max h':>8} {'err share in O(1/K) band':>26}")
    arm1 = {}
    for tname, g in TAPERS.items():
        c = cheb_project_step(args.lam0, args.K) * g
        d = ringing_diagnostics(c, args.lam0, args.K)
        arm1[tname] = d
        print(f"  {tname:>9} {d['overshoot']:+10.4f} {d['max_response']:8.4f} "
              f"{d['err_share_in_band']:>25.3f}  (band is {100*d['band_frac_of_axis']:.1f}% of the axis)")
    blob["arms"]["frozen_response"] = arm1

    # ---------- graph + planted task ----------
    print("\n[TASK] cSBM graph, band-edge label planted with a sharp edge at lam0")
    agree = {"fit_max_abs_diff": [], "coef_max_abs_diff": []}
    accs = {"frozen": {t: [] for t in TAPERS},
            "free_ols": {t: [] for t in TAPERS}}
    for s in range(args.seeds):
        sbm = generate_sbm(args.n, homophily=0.8, avg_degree=10, seed=s)
        A = sbm["A"]
        L = gs.normalized_laplacian(A)
        w, U = gs.eig_fp64(L)
        S = L - np.eye(args.n)                     # spectrum = lambda-1 in [-1,1], matching the
                                                   # mu = lambda-1 convention of cheb_project_step
        rng = np.random.default_rng(1000 + s)
        # planted signal: energy strictly inside [0, lam0] -> optimal LINEAR response is the step at lam0
        band = w <= args.lam0
        coef = rng.standard_normal(int(band.sum()))
        y_cont = U[:, band] @ coef
        y_cont /= np.linalg.norm(y_cont) / np.sqrt(args.n)
        x = y_cont + rng.standard_normal(args.n) / max(args.snr, 1e-9)
        y = (y_cont > np.median(y_cont)).astype(int)
        B = cheb_basis_apply(S, x, args.K)          # (n, K+1) Chebyshev features
        idx = rng.permutation(args.n)
        te = idx[: args.n // 2]                     # fixed test half
        pool = idx[args.n // 2:]

        def acc(c):
            z = B @ c
            return float(((z > np.median(z)).astype(int)[te] == y[te]).mean())

        fits_this_seed, creal_this_seed = {}, {}
        for tname, g in TAPERS.items():
            # ARM 1 accuracy: frozen projection coefficients (scaled by the taper)
            accs["frozen"][tname].append(acc(cheb_project_step(args.lam0, args.K) * g))
            # ARM 2: free coefficients, ordinary least squares on the full labelled pool
            Bg = B * g[None, :]
            wfit, *_ = np.linalg.lstsq(Bg[pool], y_cont[pool], rcond=None)
            accs["free_ols"][tname].append(acc(wfit * g))
            # D.1 quotes how closely the free-coefficient arms agree across tapers. Record both
            # per seed so the manuscript's tolerances trace to this file rather than to a console run.
            fits_this_seed[tname] = B @ (wfit * g)      # realized fitted response on all nodes
            creal_this_seed[tname] = wfit * g           # realized coefficients c_k = g_k w_k
        base = "none"
        agree["fit_max_abs_diff"].append(float(max(
            np.abs(fits_this_seed[t_] - fits_this_seed[base]).max() for t_ in TAPERS if t_ != base)))
        agree["coef_max_abs_diff"].append(float(max(
            np.abs(creal_this_seed[t_] - creal_this_seed[base]).max() for t_ in TAPERS if t_ != base)))

    def ms(v):
        v = np.asarray(v, float); return {"mean": float(100 * v.mean()), "std": float(100 * v.std())}

    print("\n[ARM 1] frozen coefficients -- test accuracy (ringing is real here)")
    for t in TAPERS:
        m = ms(accs["frozen"][t]); print(f"  {t:>9} {m['mean']:6.2f} +/- {m['std']:.2f}")
    b1 = ms(accs["frozen"]["none"])["mean"]
    print(f"  => damping effect: jackson {ms(accs['frozen']['jackson'])['mean']-b1:+.2f}, "
          f"lanczos {ms(accs['frozen']['lanczos'])['mean']-b1:+.2f}")

    print("\n[ARM 2] free coefficients, ordinary least squares (absorption should be EXACT)")
    for t in TAPERS:
        m = ms(accs["free_ols"][t]); print(f"  {t:>9} {m['mean']:6.2f} +/- {m['std']:.2f}")
    b2 = ms(accs["free_ols"]["none"])["mean"]
    print(f"  => damping effect: jackson {ms(accs['free_ols']['jackson'])['mean']-b2:+.4f}, "
          f"lanczos {ms(accs['free_ols']['lanczos'])['mean']-b2:+.4f}")

    blob["arms"]["accuracy"] = {
        "frozen": {t: ms(accs["frozen"][t]) for t in TAPERS},
        "free_ols": {t: ms(accs["free_ols"][t]) for t in TAPERS},
    }
    blob["free_coefficient_agreement"] = {
        "note": ("Across-taper agreement in the free-coefficient (OLS) arm, per seed. fit = realized "
                 "fitted response on all nodes; coef = realized c_k = g_k w_k. Both are max |difference| "
                 "against the no-taper arm. D.1 quotes the maxima over seeds."),
        "fit_max_abs_diff": agree["fit_max_abs_diff"],
        "coef_max_abs_diff": agree["coef_max_abs_diff"],
        "fit_max_over_seeds": max(agree["fit_max_abs_diff"]),
        "coef_max_over_seeds": max(agree["coef_max_abs_diff"]),
    }
    blob["arms"]["raw"] = {k: accs[k] for k in accs}
    json.dump(blob, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {(time.time()-t0)/60:.1f} min")
