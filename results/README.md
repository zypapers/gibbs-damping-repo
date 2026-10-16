# Result files

Every value is the recorded output of an actual run. Provenance:

**Written directly by the listed script** (re-running regenerates the file):
`platonov_paired`, `factorial_splits`, `lowpass_seeds`, `data_regime_seeds`, `falsifications_seeds`
(all from `stats_variance.py` / `falsifications_variance.py`); `coef_profile`, `wd_ablation`, `anti_control`
(`review_experiments.py`); `conditioning` (`conditioning.py`); `webkb_repro` (`webkb_fidelity.py`);
`full_table6` (`full_table6.py`).

**Collated from the script's printed summary table** (the script prints the values; the JSON is the
recorded run): `chebgibbsnet_repro` (`chebgibbsnet.py`), `zoo_absorption` (`polyfilter.py`),
`fixed_vs_free` (`chebgibbsnet.py`), `data_regime` (`chebgibbsnet.py` over train fractions).

`chebgibbsnet.py` and its train-fraction sweep reproduce their rows directly from the command line.
For `zoo_absorption` and `fixed_vs_free` the shipped entry points run only a subset of the recorded
configurations: `polyfilter.py` exercises the free and nonneg cells but not the frozen arm, and
`absorption_test.py` runs PubMed free-coefficient regimes, not the Cora and frozen arms. The frozen
arms come from the freeze flags on the library functions -- `polyfilter.run(..., freeze=True)` for
`zoo_absorption`, and `chebgibbsnet.run(..., freeze_w=True)` for `fixed_vs_free` (note that
`absorption_test.py` has no `run` of its own; it imports `chebnet_gibbs.run`). Call those directly to
regenerate the frozen rows; the JSONs record the runs as performed. The exact cells, verified to
reproduce the shipped values:

```python
import sys; sys.path.insert(0, "src/repro")
import chebgibbsnet, polyfilter
from damping_cgn import taper_cgn

# fixed_vs_free: Cora and PubMed, each frozen and free, each damped and undamped
DROP = {"Cora": 0.6, "PubMed": 0.1}
for ds in ("Cora", "PubMed"):
    for damping in ("jackson", "none"):
        for frozen in (True, False):
            chebgibbsnet.run(ds, damping, K=10, dropout=DROP[ds], split_kind="public",
                             seeds=range(8), epochs=300, patience=30, freeze_w=frozen)

# zoo_absorption: three bases, taper and no taper, frozen and free
for b in ("mono", "cheb", "bern"):
    for taper in (taper_cgn("jackson", 10), None):
        for frozen in (True, False):
            polyfilter.run("Cora", basis=b, constraint="free", init="ppr", K=10,
                           dropout=0.6, patience=30, seeds=range(5),
                           taper_arr=taper, freeze=frozen)
```

Note `taper_arr` is required: `polyfilter.run` applies no taper when it is omitted, so leaving it
out would run the no-taper cell twice.

**Analytic / closed-form** (Section 3, computed to fp64, verifiable from the paper's formulas):
`theory_step1_gibbs_overshoot`.

**Appendix D** (written directly by the listed script): `rebuttal_planted_band_v002`
(`rebuttal_planted_band.py`; cSBM + fp64 eigendecomposition, deterministic across runs),
`rebuttal_breadth_v002` (`rebuttal_breadth.py`), `rebuttal_stability_v001` (`rebuttal_stability.py`),
and `rebuttal_stats_v001` (`rebuttal_stats.py`, derived from the three above and the single source of
every bootstrap interval quoted in D.2 and D.3). `sign_absorption_v001` (`sign_absorption.py`) holds
the GSO-sign check behind §4; the older `gso_sign_absorption` block inside `data_regime_v001` is the
earlier single-split run and is superseded by it.

**One provenance caveat.** The cached `damp_effect` bootstrap-CI fields inside `platonov_paired` and
`data_regime_seeds` were produced before the resample seeding in `stats_variance.py` was made
deterministic, so re-running those two stages reproduces every mean and standard deviation but not
those two cached CI fields. No value reported in the paper is read from them.
