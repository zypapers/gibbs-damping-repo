"""ChebGibbsNet's EXACT damping factors (their Eqs. 10-11), for the reproduction / re-attribution.

NOTE the convention difference: the KPM literature (Weisse et al.) states Jackson with N1 = K+1,
while ChebGibbsNet Eq. 10 uses N1 = K+2. Reproduction MUST use ChebGibbsNet's convention.

Eq. 10 (Jackson):
  g^J_{k,K} = [ (K+2-k) sin(a) cos(k a) + cos(a) sin(k a) ] / [ (K+2) sin(a) ],  a = pi/(K+2)
Eq. 11 (Lanczos^m):
  g^La_{k,K} = ( sinc(k/(K+1)) )^m = ( sin(k pi/(K+1)) / (k pi/(K+1)) )^m,  m in Z+
Conditions (their sec 4.3): g_{0,K}=1 ; lim_{K->inf} g_{1,K} -> 1.
"""
from __future__ import annotations
import numpy as np


def jackson_cgn(K):
    a = np.pi / (K + 2)
    k = np.arange(K + 1)
    num = (K + 2 - k) * np.sin(a) * np.cos(k * a) + np.cos(a) * np.sin(k * a)
    return num / ((K + 2) * np.sin(a))


def lanczos_cgn(K, m=1):
    k = np.arange(K + 1, dtype=float)
    x = k * np.pi / (K + 1)
    sinc = np.ones_like(x)
    nz = k > 0
    sinc[nz] = np.sin(x[nz]) / x[nz]
    return sinc ** m


def fejer(K):
    k = np.arange(K + 1, dtype=float)
    return 1.0 - k / (K + 1)


def taper_cgn(name, K, m=3):
    # m=3 is ChebGibbsNet's deployed Lanczos exponent (Eq. 11). The default used to be 1,
    # which silently handed callers a different kernel than the one under study.
    name = name.lower()
    if name in ("none", "nog", "identity"):
        return np.ones(K + 1)
    if name == "jackson":
        return jackson_cgn(K)
    if name == "lanczos":
        return lanczos_cgn(K, m=m)
    if name == "fejer":
        return fejer(K)
    raise ValueError(name)


if __name__ == "__main__":
    print("ChebGibbsNet damping factors (Eqs. 10-11) — verification")
    for K in (2, 4, 8, 10):
        gJ, gL1, gL2, gL3 = jackson_cgn(K), lanczos_cgn(K, 1), lanczos_cgn(K, 2), lanczos_cgn(K, 3)
        print(f"\nK={K}")
        print(f"  Jackson    g_0={gJ[0]:.4f} (want 1) g_1={gJ[1]:.4f}  min={gJ.min():.4f} all>0:{np.all(gJ>0)}")
        print(f"  Lanczos^1  g_0={gL1[0]:.4f} (want 1) g_1={gL1[1]:.4f}  min={gL1.min():.4f} all>0:{np.all(gL1>0)}")
        print(f"  Lanczos^2  g_0={gL2[0]:.4f}          min={gL2.min():.4f} all>0:{np.all(gL2>0)}")
        print(f"  Lanczos^3  g_0={gL3[0]:.4f}          min={gL3.min():.4f} all>0:{np.all(gL3>0)}"
              f"   (paper: m=3 'close to Jackson but not strictly positive')")
    # g_1 -> 1 as K grows (their condition 2)
    print("\n  Jackson g_1 as K->inf:", [f"{jackson_cgn(K)[1]:.3f}" for K in (2, 8, 32, 128)])
