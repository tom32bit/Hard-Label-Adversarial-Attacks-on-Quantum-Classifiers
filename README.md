# Hard-Label Adversarial Attacks on Quantum Classifiers

Implementation of the research plan in
[`hard_label_quantum_attacks_research_plan.md`](hard_label_quantum_attacks_research_plan.md):
a **shot-budget-aware, Born-rule-calibrated decision-based (hard-label) attack** for
variational quantum classifiers, its baselines, the first gradient-free evaluation of
two published quantum defenses, and a concentration-vs-robustness guardrail.

Target venue: *Quantum Machine Intelligence*. Classical simulation, 4–12 qubits,
PennyLane + PyTorch. **No GPU or quantum hardware required.**

---

**Proofs:** the theory is worked out with full proofs in
[`docs/theory.md`](docs/theory.md) — the exact Binomial oracle (Prop 1), the
shot-limited boundary resolution `τ(S)` and why the naive port diverges with budget
(Prop 2), the query/shot economics with a *proved* answer that reframes RQ2 (Prop 3),
and the concentration ⟹ exponential hard-label-cost theorem behind the defense results
(Thm 4).

## The core idea

A deployed quantum classifier returns a *measured label*, not a gradient. With a finite
shot budget `S` that label is a Bernoulli sample whose flip probability is fixed by the
Born-rule margin:

```
p_flip(x, S) ≈ Φ( − m(x) · sqrt( S / (1 − f_θ(x)²) ) ),      m(x) = |f_θ(x)|
```

So the oracle is **natively stochastic**, and near the boundary — exactly where a
decision-based attack spends its queries — the label is almost a fair coin. Two
consequences drive the whole design:

1. **You cannot resolve the sign at the boundary at any finite budget.** The shot budget
   instead sets a *boundary-localization resolution* `τ(S) = z_δ · sqrt((1−f²)/S)`. The
   calibrated attack bisects only to that resolution and stops conservatively inside it
   (**M1**). A naive port that ignores this suffers a **noise-induced inward bias**: one
   false-positive "adversarial" reading collapses the binary search toward the original
   point and never recovers, so it returns examples that are *not actually adversarial*.
2. **Shots are a controllable knob**, so the attacker chooses its own oracle noise and
   trades queries against shots under a fixed total budget `T = Q × S` (**M2**).

## Results are always *verified*

A hard-label attacker only sees noisy labels, so it can believe it crossed the boundary
when it did not. Every returned adversarial is therefore re-checked against the exact
(infinite-shot) model, and a run counts as a success **only if the returned point is
truly adversarial**. This single choice is what separates the methods.

---

## Layout

```
hlq/                        the library
  config.py                 typed configs + presets (JSON round-trippable)
  data.py                   REAL MNIST / Fashion-MNIST binary pairs (full scope)
                            + the synthetic known-boundary control (T3 only)
  classifier.py             the VQC: encoding / observable / ansatz switches
  classical.py              parameter-matched classical NN (the classical anchor)
  oracle.py                 exact-Binomial stochastic oracle, Born p_flip, shot controller
  budget.py                 query/shot economics: the (B, S_probe) objective (RQ2)
  attacks/
    base.py                 the ONE HopSkipJump skeleton; subclasses override policy only
    hsja_quantum.py         calibrated M1+M2  (ours)
    hsja_fixed.py           naive fixed-shot port (and the deterministic anchor)
    popskipjump.py          constant-flip-rate baseline
    pgd_whitebox.py         white-box reference (strongest attacker)
    momentum.py             momentum-based quantum attack (QMI precedent [6])
    transfer.py             classical-surrogate transfer
  defenses/
    noise.py                depolarizing defense (true density-matrix simulation)
    randomized_encoding.py  fresh random rotations per query
  metrics.py                verification + the plan's statistics
  concentration.py          RQ5 guardrail (var[f], above-chance, trivially-robust flag)
  analysis.py               budget curves, interior-optimum test, model validation
  runner.py                 one experiment cell = model x defense x attack x seed
experiments/
  run_sanity.py             T1-T6
  driver.py                 composes cells into RQ1-RQ5 + ablations
  make_figures.py           research figures from the JSON
  make_notebook.py          generates the self-contained notebook from these sources
notebooks/                  the generated Kaggle/Colab notebook
results/                    JSON outputs
figures/                    PNG + vector PDF
```

**No duplication by construction:** the HSJA geometry exists once in `attacks/base.py`;
each attack overrides only its shot-budget and side-of-boundary *policy*. Defenses wrap a
classifier behind the same `decision_function` interface, so the oracle and every attack
run on them unchanged. The notebook is *generated* from these sources and cannot drift.

---

## Quick start

```bash
python experiments/run_sanity.py --quick        # the trust gate (minutes)
python experiments/driver.py --rq rq1 --preset smoke
python experiments/make_figures.py
```

Scale with `--preset {smoke,medium,full}` and `--jobs N`. `full` is the
heavier-statistics scope (250 attacked images/cell, 8 seeds) and is a multi-hour job —
run it one RQ at a time.

Trained models and dataset features are cached (`models_cache/`, `data_cache/`), so
models are trained once and reused across every attack, defense and RQ.

### Kaggle

Open `notebooks/hard_label_quantum_attacks.ipynb`, use a **CPU** session and **enable
Internet** (for the real MNIST download; the loader also falls back to keras or an
attached dataset), then run top to bottom. A GPU does not help — these circuits are tiny
and the bottleneck is the number of sequential circuit evaluations. The notebook defaults
to the two heaviest blocks (RQ4 defenses, RQ5 concentration); set `RQS = ALL_RQS` to run
the whole study there.

---

## The sanity gate (plan Sec. 7)

Code is not trusted until **T1-T4** pass. All six pass.

| ID | Test | Status |
|----|------|--------|
| **T1** | `p_flip` closed form vs Monte-Carlo, and vs PennyLane's own shot sampling | PASS |
| **T2** | infinite-shot limit -> deterministic HopSkipJump (paired per-image) | PASS |
| **T3** | attack vs an analytically known boundary (recovered distance error 0.09%) | PASS |
| **T4** | already-adversarial input returned unchanged | PASS |
| **T5** | budget monotonicity (paired, common success set) | PASS |
| **T6** | concentration guardrail fires on a concentrated model, not a trained one | PASS |

Two of these needed **paired per-image** statistics: the perturbation varies ~10x across
inputs and the set of images that succeed changes with the budget, so comparing medians
of different small subsets measures image variance and selection bias rather than the
effect under test.

---

## Headline results (real MNIST 3-vs-5, n=8, L=5, angle encoding)

40 attacked test images per cell x 3 seeds; VQC test accuracy 0.884 / 0.913 / 0.903.
Success is **verified against the exact model**. (These are from the `medium` preset; the
`full` preset scales to 250 images/cell x 8 seeds.)

**RQ1 — feasibility, at T = 60 000 shots**

| method | verified success | median l2 | median queries |
|---|---|---|---|
| **Calibrated HSJA (ours)** | **0.59 ± 0.07** | 0.840 | 3494 |
| PopSkipJump (constant noise) | 0.41 ± 0.01 | 0.919 | 616 |
| Fixed-shot HSJA (naive port) | 0.14 ± 0.01 | 0.893 | 876 |
| Classical-surrogate transfer | 1.00 | 0.517 | 331 |
| HSJA on matched classical NN | 1.00 | 0.532 | 1441 |
| White-box PGD (reference) | 1.00 | 0.492 | — |

The anchor is the point: the *same* boundary walk breaks a parameter-matched **classical**
NN 100% of the time in ~1.4k queries, but reaches only 59% against the VQC. The
stochastic Born-rule oracle — not the attack — is what costs.

**RQ3 — calibration payoff (verified success at equal budget)**

| T | Calibrated | PopSkipJump | Fixed |
|---|---|---|---|
| 15 000 | **0.70** | 0.00 | 0.36 |
| 60 000 | **0.59** | 0.41 | 0.14 |
| 120 000 | **0.58** | 0.38 | 0.07 |

Two effects worth naming:

* **The naive port gets *worse* as the budget grows** (0.36 -> 0.14 -> 0.07). That is the
  inward-bias signature: more iterations means more chances for a false-positive
  "adversarial" reading, and the binary search never recovers from one.
* **The calibrated attack trades success for precision** (success 0.70 -> 0.58 while its
  perturbation tightens 0.952 -> 0.756). More budget buys finer boundary resolution `τ`,
  so the returned point sits closer to the true boundary and is inherently more marginal.

**RQ2 — query/shot economics.** A weak interior optimum appears in the probe-shot
allocation (perturbation dips at `S_probe ≈ 2`) and a clearer one in the M1/M2 split
(best around 0.75 of the per-iteration budget to the normal estimate), partially
supporting H2. **Ablations:** re-uploading is the most attackable encoding (0.74 success)
and amplitude the least (0.24); MNIST 0-vs-1 (near-separable, clean acc 0.995) is easier
to attack than 3-vs-5 or Fashion-MNIST.

**RQ4 — first gradient-free test of two published defenses** (n=4, 3 seeds; the
calibrated hard-label attacker never computes a gradient). Both sides of H4 appear, one
per defense:

| defense | attack success | clean accuracy | verdict |
|---|---|---|---|
| none | 0.80 | 0.878 | baseline (attackable) |
| depolarizing noise | 0.40 | **0.480 (below chance)** | H4a — false sense of security (the model collapsed, it was not defended) |
| randomized encoding | **0.12** | 0.832 | H4b — a genuine, non-masking defense (accuracy preserved, attack degraded) |

The chance line is the whole story: the noise defense's apparent robustness is its
accuracy falling *below* chance, exactly the confound RQ5 guards against; the randomized
encoding is the real thing.

**RQ5 — concentration vs robustness** (5 seeds, n = 4→10). `Var[f_θ(x)]` decays
exponentially with qubit count — local `Z₀`: `R² = 0.991`, decay `b = 0.065`,
**p = 4.6e-3**; global `Z^⊗n`: `R² = 0.85`, `p = 0.079` (not significant). Over the same
range attack success falls (0.68 → 0.42 global / 0.51 local) and median perturbation
grows (0.69 → 0.93/0.96), **while clean accuracy stays 0.87–0.91** — i.e. this is
genuine robustness-with-scale, *not* the concentration collapse the guardrail is there to
catch (contrast the depolarizing defense in RQ4, where accuracy fell to 0.48). Reporting
robustness against accuracy-above-chance is what separates the two.

All of RQ4/RQ5 were produced by the Kaggle notebook in **~2.2 h** (well inside the 12 h
limit) on the `kaggle` preset, with the checkpointing described below.

---

## Design notes worth knowing

* **The analytic Born oracle.** For a diagonal observable each shot is an independent
  Bernoulli with `p(+1) = (1+f)/2`, so the exact `S`-shot label is a single
  `Binomial(S, (1+f)/2)` draw from the once-computed `f`. This reproduces real
  shot-based measurement *exactly* (cross-checked against PennyLane's sampler in T1)
  without materialising `S` samples.
* **`f` is cached per point.** Repeated queries at the *same* point (the calibrated
  attack's shot escalation, PopSkipJump's repeats) need only one simulation; the shots
  remain independent draws from that same `f`. Verified **bit-identical** to the
  uncached path — it is purely a compute optimisation.
* **PopSkipJump gets its constant `p0` from config, not from measurement.** Estimating
  the flip rate at a convenient (far-from-boundary) point returns ~0 and silently
  collapses the baseline into the naive fixed-shot attack. Its premise is a constant,
  oracle-agnostic noise level, so it is given exactly that.
* **Figures**: validated colorblind-safe categorical palette used in fixed order and
  always with a second encoding (marker + linestyle); single-hue sequential ramps for
  magnitude; **never** a dual y-axis; error bars are s.d. over seeds.

---

## License

MIT — see [LICENSE](LICENSE).
