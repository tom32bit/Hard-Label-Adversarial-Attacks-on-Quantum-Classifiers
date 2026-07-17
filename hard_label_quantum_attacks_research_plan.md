# Hard-Label Adversarial Attacks on Quantum Classifiers
### A Shot-Budget-Aware, Born-Rule-Calibrated Decision-Based Attack for Variational Quantum Classifiers

**Target venue:** *Quantum Machine Intelligence* (Springer)
**Format fit:** Research Article — method + algorithm + benchmark (the journal's dominant template)
**Status of idea:** Pre-emption-checked across four literatures; **OPEN** with one partial adjacency that strengthens rather than blocks it (see §2)
**Compute regime:** Classical simulation, 4–12 qubits, PennyLane + PyTorch. No quantum hardware required. Optional single hardware-validation run.

---

## 0. One-paragraph summary

Every published adversarial attack on quantum classifiers operates in the **white-box, gradient-based** threat model — the attacker differentiates the loss through the circuit (quantum FGSM/BIM/PGD). This is the *least* realistic deployment scenario: a quantum classifier served through a cloud API returns a **measured class label**, not a loss gradient, not even a class probability vector at full precision. The classical adversarial-ML community abandoned the white-box assumption years ago and developed **decision-based (hard-label) attacks** that craft adversarial examples from the top-1 output label alone (Boundary Attack, HopSkipJump, SignOPT, qFool). None of these has been ported to quantum classifiers. The port is *not* mechanical: a quantum classifier is a **natively stochastic oracle** — the returned label is a Bernoulli sample whose flip probability is set by the Born-rule margin `|⟨M⟩|` and the shot budget `S`. Decision-based attacks assume a deterministic oracle and are known to break when labels are noisy (PopSkipJump). We therefore develop a **shot-budget-aware, Born-rule-calibrated hard-label attack** for variational quantum classifiers, characterize its query/shot economics, and use it to re-evaluate existing "quantum noise as a defense" and "randomized encoding" claims — which were only ever tested against gradient-based adversaries.

---

## 1. Motivation and positioning

### 1.1 The threat-model gap (the core contribution)

Quantum adversarial machine learning was opened by Lu, Duan & Deng (*Phys. Rev. Research* 2020) [1] and Liu & Wittek (*Phys. Rev. A* 2020) [2], who showed VQCs are as vulnerable to imperceptible perturbations as classical nets. Everything downstream — universal perturbations [3,4], experimental demonstrations on superconducting hardware [5], momentum-boosted attacks (QMI, 2026) [6], transfer-learning robustness studies [7] — inherits their **white-box, gradient-based** attacker. A recent Systematization-of-Knowledge paper (Nov 2025) [8] catalogues quantum attacks across poisoning, backdoor, and evasion, and confirms that evasion attacks in the quantum literature are gradient-based (FGSM/PGD); it implements a black-box **poisoning** attack but no decision-based **evasion** attack.

Meanwhile, the classical field's realistic threat model is the **hard-label** setting: the attacker observes only `argmax` of the output and must reconstruct enough boundary geometry to craft a perturbation. This is the standard against which real deployed classifiers are stress-tested [9–14], because production APIs typically return only a top-1 label.

**No decision-based hard-label attack has been developed for quantum classifiers.** That is the gap.

### 1.2 Why the port is a genuine research problem, not an engineering exercise

A decision-based attack such as HopSkipJump [11] works by:
1. locating a point on the decision boundary via binary search between a safe and an adversarial point;
2. estimating the **boundary normal direction** by Monte-Carlo sampling: perturb the boundary point with `B` random vectors, query the label of each, and average the ±1 labels to get a gradient-direction estimate;
3. stepping along that direction and repeating.

Step 2 assumes the oracle is **deterministic**: the sign returned for each probe is the true side of the boundary. For a quantum classifier this is false. The predicted label is
```
ŷ(x) = sign( ⟨M⟩_x ),     ⟨M⟩_x = ⟨0| U†(x,θ) M U(x,θ) |0⟩,
```
but with a finite shot budget `S` we never observe `⟨M⟩_x`; we observe an **empirical mean** `⟨M̂⟩_x` over `S` shots, and hence a **stochastic label** whose probability of matching the true label is governed by the margin `|⟨M⟩_x|` and by `S`. Near the boundary (`⟨M⟩_x → 0`) — which is *exactly where the attack spends all its queries* — the label is almost a fair coin.

The classical community already discovered that decision-based attacks **fail against stochastic oracles**: PopSkipJump [15] shows that flipping just 1 in 20 query labels makes state-of-the-art decision-based attacks fail, and builds a probabilistic HopSkipJump that adapts its query count to the noise level. **But PopSkipJump assumes an abstract, constant label-flip probability.** The quantum case has *structure PopSkipJump does not exploit*:
- the flip probability is not constant — it is a known function of the Born margin and shot count;
- the shot count `S` is a **controllable knob**, so the attacker chooses its own oracle noise per query, trading queries against shots under a fixed total measurement budget.

This turns "port an attack" into a real question with a clean, quantum-specific answer: **how should a hard-label adversary allocate a fixed measurement budget between more queries (noisier labels) and more shots per query (cleaner labels)?**

### 1.3 Why *Quantum Machine Intelligence*

- QMI publishes exactly this genre — take a classical ML mechanism, port it to the quantum setting with a quantum-specific twist, benchmark it. The June 2026 momentum-attack paper [6] is the direct precedent and a required comparison.
- QMI also publishes the defenses this work re-evaluates: quantum-noise defenses [16], randomized-encoding defenses [17], and depolarization white-box attack/defense studies [18].
- Scope alignment (from the author's priority list): **quantum-enhanced robustness in ML models** (primary), **quantum variational circuits and their applications**, **encoding and processing of data in quantum systems**.

---

## 2. Pre-emption verdict (Phase 1 output)

**Systematic search across four literatures.** Verdict per component:

| Component of the idea | Status | Closest prior work | Delta we own |
|---|---|---|---|
| Adversarial vulnerability of VQCs | **EXISTS** (foundational) | Lu–Duan–Deng [1]; Liu–Wittek [2] | We attack, don't re-establish vulnerability |
| White-box gradient attacks on VQCs (FGSM/PGD/momentum) | **EXISTS** | QMI momentum attack [6]; SoK [8] | We use a *different threat model* (hard-label) |
| Black-box attacks on quantum models | **PARTIAL** | Quantum-autoencoder transfer attack (Physica A 2025) [19]; SoK black-box **poisoning** [8] | Those are *transfer/poisoning*; we do *decision-based evasion* |
| Decision-based hard-label attacks (Boundary/HSJA/SignOPT/qFool) | **EXISTS classically, ABSENT in quantum** | Chen–Jordan–Wainwright [11]; Brendel [10]; Cheng SignOPT [12] | **First quantum port** |
| Decision-based attack on a **stochastic** oracle | **PARTIAL** | PopSkipJump [15] | PopSkipJump assumes constant flip-rate; we use **Born-rule-calibrated, shot-controllable** noise |
| Shot-budget / query economics of a quantum attack | **OPEN** | — | Fully open |
| Re-evaluating quantum-noise & randomized-encoding defenses vs. a *gradient-free* attacker | **OPEN** | Defenses [16,17] tested only vs. gradient attacks | Fully open |

**Overall: OPEN.** The single closest adjacency (PopSkipJump) is *not* a quantum paper and does not use the Born-rule structure; it becomes our strongest **baseline**, which is exactly what a rigorous attack paper needs.

**Residual pre-emption risk (declared, not hidden):**
- **R1.** The Deng group (authors of the noise defense [16], randomized-encoding defense [17], universal perturbations [3,4]) is the most likely to pre-empt. A Semantic Scholar **forward-citation crawl** of [1], [11], and [15] is the recommended final check before submission (not yet run — flagged in the closing statement).
- **R2.** The SoK [8] and a 2026 "kill chain" security taxonomy [20] are very recent; both must be read in full to confirm neither quietly implements a decision-based evasion attack. Excerpts reviewed so far indicate they do not.
- **R3.** Formal robustness-verification work (Guan–Fang–Ying [21], Weber et al. [22]) *computes* worst-case adversarial bounds via SDP/hypothesis-testing but does not run a **query-based** attack; adjacent, not overlapping. Must be cited as the "certification" counterpart.

---

## 3. Research questions and hypotheses

**RQ1 (feasibility).** Can a decision-based hard-label attack fool a variational quantum classifier using only measured labels, at a query cost comparable to classical decision-based attacks on similarly-sized classical models?

**RQ2 (the quantum twist).** Given a fixed total measurement budget `T = Q × S` (queries × shots-per-query), what allocation minimizes the final adversarial perturbation norm? Is there a non-trivial interior optimum?

**H2.** Yes. Too few shots → labels near the boundary are coin-flips, and the normal-direction estimate is dominated by measurement noise; too many shots → too few queries to converge the boundary walk. The optimum `S*` scales with proximity to the boundary and can be predicted from the Born-margin model.

**RQ3 (calibration payoff).** Does exploiting the *known* Born-rule flip-probability model (our calibrated attack) beat a black-box constant-noise treatment (PopSkipJump [15]) at equal measurement budget?

**H3.** Yes, because the attacker can allocate shots adaptively — spending more only where the margin is small — instead of PopSkipJump's oracle-agnostic schedule.

**RQ4 (defense re-evaluation).** Do "quantum noise as defense" [16] and "randomized encoding" [17] — both proven only against *gradient-based* adversaries — actually degrade a *gradient-free* hard-label attack?

**H4 (two-sided, both publishable).**
- (a) If the defenses **fail**: they were gradient-masking in the classical sense [23] — a false sense of security — and the field needs gradient-free evaluation as standard. (Quantum analogue of Athalye et al. [23].)
- (b) If the defenses **hold**: there is a genuine, non-masking protection, because the *same* barren-plateau effect that suppresses the attacker's gradient also suppresses the information a boundary-walk can extract — a real, provable asymmetry worth a theorem. (Motivated by the gradient-free barren-plateau result of Arrasmith et al. [24].)

**RQ5 (confound control).** Exponential concentration [25,26] makes a VQC's output margin shrink with qubit count; a model deep in concentration is nearly **data-independent** and would register spuriously high "robustness" (zero accuracy drop under attack) while being useless. Does apparent robustness track concentration rather than genuine defense?

**H5.** Robustness must be reported **relative to clean accuracy above chance**; we provide a concentration diagnostic (output-margin variance vs. qubit count) alongside every robustness number.

---

## 4. Formal specification (Phase 2, Step 1)

### 4.1 The classifier under attack

Binary variational quantum classifier on `n` qubits:
```
ŷ(x) = sign( f_θ(x) ),     f_θ(x) = ⟨0|^{⊗n}  E(x)† W(θ)† M W(θ) E(x)  |0⟩^{⊗n}
```
- `E(x)` — data-encoding unitary (angle encoding as primary; amplitude and re-uploading as ablations, per [8] which shows encoding strongly changes robustness).
- `W(θ)` — trained variational ansatz (hardware-efficient; layers `L ∈ {2,5,10}` to probe the concentration/robustness interaction observed in [8]).
- `M` — Hermitian measurement observable (single-qubit `Z_0` primary; global `Z^{⊗n}` as a concentration-inducing ablation, since global observables are a named concentration source [26]).
- `f_θ(x) ∈ [−1, 1]` is the ideal (infinite-shot) decision value; **margin** `m(x) = |f_θ(x)|`.

### 4.2 The stochastic oracle (the crux)

With `S` shots and observable `M` with eigenvalues in `{−1,+1}`, the estimator `f̂` of `f_θ(x)` from `S` measurements has
```
E[f̂] = f_θ(x),      Var[f̂] = (1 − f_θ(x)²) / S.
```
The **label** returned is `ŷ_S(x) = sign(f̂)`. Its flip probability relative to the true label is, by a Gaussian (CLT) approximation valid for moderate `S`,
```
p_flip(x, S) ≈ Φ( − m(x) · sqrt( S / (1 − f_θ(x)²) ) ),
```
where `Φ` is the standard normal CDF.

> **VERIFY (analytic limits):**
> - `S → ∞` ⟹ `p_flip → 0` (deterministic oracle recovered). ✓
> - `m(x) → 0` (on boundary) ⟹ `p_flip → Φ(0) = 1/2` (fair coin). ✓
> - `p_flip` monotonically decreasing in both `S` and `m(x)`. ✓
> These three must be reproduced numerically before any attack code is trusted (sanity test T1, §7).

This closed form is what distinguishes the quantum attack from PopSkipJump's constant-`p_flip` assumption. It also gives the attacker a **per-query shot controller**: to guarantee `p_flip ≤ δ` at margin estimate `m̂`, set
```
S(δ, m̂) = ⌈ (1 − f̂²) · (Φ⁻¹(δ) / m̂)² ⌉.
```

### 4.3 The attack (calibrated hard-label boundary walk)

We adapt HopSkipJump [11]. The two quantum-specific modifications:

**(M1) Shot-calibrated boundary search.** Binary search for the boundary requires reliable side-of-boundary labels. Near the boundary the margin is small, so we *increase* `S` adaptively using `S(δ, m̂)` to hold the per-decision error at `δ`, rather than using a fixed `S`.

**(M2) Budget-aware normal estimation.** HSJA estimates the boundary normal by averaging labels of `B` random probes. With a stochastic oracle, each probe's label has variance from `p_flip`. Under a total budget `T`, we jointly choose `(B, S_probe)` to minimize the variance of the estimated normal direction, given the closed-form `p_flip`. This is the paper's core algorithmic novelty — an **optimal query/shot split** derived from §4.2.

**Objective (untargeted, ℓ₂):**
```
min_{x'} ‖x' − x‖₂   s.t.   ŷ(x') ≠ ŷ(x),
```
solved through the calibrated boundary walk under measurement budget `T`.

### 4.4 Baselines and controls (Phase 2, Step 3)

| Role | Method | Purpose |
|---|---|---|
| **Upper-bound reference** | White-box quantum PGD (parameter-shift gradients) [1,6] | The strongest possible attacker; lower bound on achievable perturbation |
| **Naive port** | HSJA with **fixed** shots, no calibration [11] | Shows the port fails / is inefficient without M1–M2 |
| **Stochastic-oracle baseline** | PopSkipJump (constant `p_flip`) [15] | The key comparison for RQ3 — beats naive but is oracle-agnostic |
| **Transfer baseline** | Classical-surrogate transfer attack [19] | Alternative black-box route; contextualizes query cost |
| **Classical anchor** | HSJA on a classical NN of matched parameter count | Calibrates whether quantum query costs are "normal" |
| **Ablation A** | Remove M1 (fixed shots) | Isolates value of shot calibration |
| **Ablation B** | Remove M2 (fixed `B`, `S_probe`) | Isolates value of budget-aware normal estimation |
| **Ablation C** | Encoding ∈ {angle, amplitude, re-uploading} | Tests encoding-dependence of attackability [8] |
| **Concentration control** | Local `Z_0` vs global `Z^{⊗n}`; `n ∈ {4,…,12}` | Separates genuine robustness from concentration artifact [25,26] |

### 4.5 Defenses evaluated (RQ4)

- **No defense** (baseline VQC).
- **Quantum-noise defense** [16] — depolarizing/rotation noise injected at inference.
- **Randomized-encoding defense** [17] — random unitary / QEC encoder inducing an attacker-side barren plateau.
- (Optional) **adversarial training** with white-box examples [5], to test whether it transfers to hard-label attacks.

The scientific point: all three were validated against gradient-based adversaries. Our attacker never computes a gradient of the classifier. **This is the first gradient-free test of these defenses.**

---

## 5. Datasets and models

- **Primary:** downscaled MNIST binary pairs (3-vs-5, 0-vs-1) — the field-standard QML benchmark used in [1,5,8], enabling direct comparison.
- **Secondary:** Fashion-MNIST binary pair (robustness generalization, per [18]).
- **Structured control:** a synthetic linearly/periodically separable dataset with an analytically known boundary — lets us check attack-found perturbations against ground-truth boundary distance (sanity test T3).
- **Encodings:** angle (primary), amplitude, data re-uploading.
- **Circuit sizes:** `n ∈ {4, 6, 8, 10, 12}` qubits; ansatz depth `L ∈ {2, 5, 10}`.

Rationale for staying ≤12 qubits: exact statevector simulation with shot sampling is tractable; the concentration effect [25,26] is already visible in this range; and the whole study runs on a workstation, matching QMI's typical simulation-scale papers.

---

## 6. Statistical design (Phase 2, Step 4)

- **Seeds:** ≥5 independent `(θ, train/test split)` realizations per configuration; report **mean ± std**.
- **Self-averaging indicator:** report normalized dispersion `σ/|mean|` for the headline metrics (median perturbation norm, median queries-to-success).
- **Primary metrics:**
  - median ℓ₂ perturbation at fixed budget `T`;
  - **queries-to-success** and **shots-to-success** at fixed success threshold;
  - attack success rate at fixed perturbation budget `ε`;
  - **robustness reported relative to clean accuracy above chance** (RQ5 guardrail).
- **Budget curves:** perturbation vs. `T`, and perturbation vs. shot-allocation ratio `S/T` at fixed `T` (the RQ2 interior-optimum plot).
- **Correlation claims:** any `p_flip`-model-vs-empirical fit must report Pearson `r` with p-value; `|r| > 0.9` requires `p < 0.001`.
- **Scaling:** apparent-robustness-vs-`n` fitted against the concentration model; if extrapolated, use `1/n` scaling with `R² > 0.95` required before any claim.

---

## 7. Sanity checks — defined *before* code (Phase 2, Step 5)

| ID | Test | Expected result |
|---|---|---|
| **T1** | `p_flip(x,S)` numeric vs. §4.2 closed form, swept over `m ∈ [0,1]`, `S ∈ {10,…,10⁴}` | Match within Monte-Carlo error; limits `S→∞ ⇒ 0`, `m→0 ⇒ 1/2` |
| **T2** | Infinite-shot limit of the attack = deterministic HSJA | Calibrated attack → standard HSJA trajectory as `S→∞` |
| **T3** | Attack on synthetic dataset with known boundary | Recovered perturbation norm ≈ true point-to-boundary distance (within tol) |
| **T4** | Zero-perturbation input | Attack returns the input unchanged when it is already adversarial (trivial-case guard) |
| **T5** | Budget monotonicity | Larger `T` never yields a *worse* median perturbation (within noise) |
| **T6** | Concentration guard | For global `Z^{⊗n}` at large `n`, clean accuracy → chance ⟹ flagged as "trivially robust", not "defended" |

Code will not be trusted until T1–T4 pass.

---

## 8. Implementation plan (Phase 3)

**Toolchain:** Python 3.10+, **PennyLane** (classifier + shot-based sampling via `qml.device("default.qubit", shots=S)`), **PyTorch** (training + classical anchor), NumPy/SciPy, Matplotlib.

**Modules:**
1. `classifier.py` — VQC definition, training loop, encoding switch, observable switch. Each function docstring states the unitary/loss it implements; `# VERIFY:` on every non-trivial numeric step; asserts for known limits.
2. `oracle.py` — the stochastic label oracle; the `p_flip` closed form and its Monte-Carlo validator (T1); the adaptive shot controller `S(δ, m̂)`.
3. `attacks/` — `hsja_quantum.py` (calibrated, M1+M2), `hsja_fixed.py` (naive port), `popskipjump.py` (baseline [15]), `pgd_whitebox.py` (reference [1,6]), `transfer.py` (surrogate baseline [19]).
4. `defenses/` — noise defense [16], randomized-encoding defense [17].
5. `experiments/` — `argparse` for every hyperparameter and seed; `config.json` dumped at start; `results.json` as `{"metric": {"mean":…, "std":…, "seeds":[…]}}`; figures to `figures/`.
6. `analysis.py` — budget curves, concentration diagnostics, correlation tests with p-values.

**Reproducibility block** (in every script): the standard `set_seed(np, torch, random)`.

**Pre-return checklist** (per skill): density-matrix validity where applicable, observable values in `[−1,1]`, loss decreasing in first 10 steps, latent/decision values non-degenerate, distance metric sanity on a test triplet.

---

## 9. Milestones

| Phase | Work | Output |
|---|---|---|
| **P0 (wk 1)** | Forward-citation crawl of [1],[11],[15]; read [8],[20] in full | Final novelty confirmation (closes R1–R2) |
| **P1 (wk 1–2)** | `classifier.py`, `oracle.py`; pass T1, T2 | Verified stochastic oracle + `p_flip` model |
| **P2 (wk 2–3)** | `hsja_quantum.py` (M1+M2); pass T3, T4 | Working calibrated attack |
| **P3 (wk 3–4)** | Baselines + ablations A/B/C; budget curves | RQ1–RQ3 results |
| **P4 (wk 4–5)** | Defenses [16,17] vs. hard-label attack; concentration control | RQ4–RQ5 results |
| **P5 (wk 5–6)** | (Optional) one IBM-hardware validation run at `n=4` | Real-device data point |
| **P6 (wk 6–7)** | Writing, figures, `sn-jnl` formatting | Submission draft |

---

## 10. Anticipated referee objections and pre-emptive answers

1. *"Isn't this just HopSkipJump with more shots?"* — No. The naive fixed-shot port is an explicit baseline that we show underperforms; the contribution is the Born-rule-calibrated query/shot budget allocation (M1+M2) and its analysis, benchmarked against PopSkipJump [15].
2. *"Quantum noise already defends against attacks [16]."* — Only against gradient-based attacks. RQ4 is the first gradient-free test; either outcome is a result.
3. *"Robustness gains might be exponential concentration."* — Explicitly controlled (RQ5, T6); robustness reported relative to above-chance accuracy.
4. *"Why not hardware?"* — Simulation with shot sampling faithfully reproduces the stochastic oracle; §5 justifies ≤12 qubits; optional hardware run included. This matches QMI's simulation-scale norms.
5. *"Encoding choice drives robustness [8] — did you control it?"* — Yes, Ablation C sweeps angle/amplitude/re-uploading.

---

## 11. Contribution checklist (what is genuinely new)

- [x] First decision-based **hard-label** adversarial attack for quantum classifiers.
- [x] Closed-form **Born-rule flip-probability model** linking shot budget to oracle reliability.
- [x] A **shot-budget-aware query allocation** with an interior optimum (query/shot economics — a genuinely quantum resource question).
- [x] First **gradient-free** evaluation of quantum-noise [16] and randomized-encoding [17] defenses.
- [x] A **concentration-vs-robustness** guardrail metric for the QML-security literature.

---

## References

*(All entries below were surfaced and cross-checked during the literature pass. Bracketed numbers are cited in-text above. Items marked ⚠ still require a full-text read or forward-citation check before submission, per the closing statement.)*

**Quantum adversarial ML — foundations & attacks**
[1] S. Lu, L.-M. Duan, D.-L. Deng. *Quantum adversarial machine learning.* Phys. Rev. Research 2, 033212 (2020).
[2] N. Liu, P. Wittek. *Vulnerability of quantum classification to adversarial perturbations.* Phys. Rev. A 101, 062331 (2020).
[3] W. Gong, D.-L. Deng. *Universal adversarial examples and perturbations for quantum classifiers.* National Science Review 9(6), nwab130 (2022).
[4] Y.-Z. Qiu. *Universal adversarial perturbations for multiple classification tasks with quantum classifiers.* Mach. Learn.: Sci. Technol. 4(4), 045009 (2023).
[5] W. Ren et al. *Experimental quantum adversarial learning with programmable superconducting qubits.* Nature Computational Science 2, 711 (2022).
[6] M. Pan, W. Liu, F. Yang, et al. *Momentum-based quantum adversarial attack algorithm.* Quantum Machine Intelligence (2026), art. 62. ⚠ full-text
[7] *Adversarially robust quantum transfer learning.* (2025), arXiv:2510.16301. ⚠ full-text
[8] *SoK: Critical Evaluation of Quantum Machine Learning for Adversarial Robustness.* (2025), arXiv:2511.14989. ⚠ full-text (competitor — confirm no decision-based evasion)
[19] *A black-box attack method of machine learning algorithms based on quantum autoencoders.* Physica A (2025), S0378437125006855. ⚠ full-text
[20] *Entangled Threats: A Unified Kill Chain Model for Quantum Machine Learning Security.* (2025), arXiv:2507.08623. ⚠ full-text
[27] X. Hou, R. Wu, Z. Wang, X. Wang. *Quantum adversarial attack generation algorithm based on variational quantum classifiers.* J. UESTC 52(2), 162 (2023).
[28] H. Liao, I. Convy, W. J. Huggins, K. B. Whaley. *Robust in practice: Adversarial attacks on quantum machine learning.* Phys. Rev. A 103, 042427 (2021).
[29] P. Georgiou, S. T. Jose, O. Simeone. *Adversarial quantum machine learning: an information-theoretic generalization analysis.* (2024).
[30] *On the generalization of adversarially trained quantum classifiers.* (2025), arXiv:2504.17690.

**Quantum defenses & certification**
[16] Y. Du, M.-H. Hsieh, T. Liu, D. Tao, N. Liu. *Quantum noise protects quantum classifiers against adversaries.* Phys. Rev. Research 3, 023153 (2021).
[17] W. Gong, D. Yuan, W. Li, D.-L. Deng. *Enhancing quantum adversarial robustness by randomized encodings.* Phys. Rev. Research 6, 023020 (2024).
[18] D. Winderl, N. Franco, J. M. Lorenz. *Quantum neural networks under depolarization noise: exploring white-box attacks and defenses.* Quantum Machine Intelligence 6(2), 83 (2024).
[21] J. Guan, W. Fang, M. Ying. *Robustness verification of quantum classifiers.* CAV 2021, LNCS, 151–174.
[22] M. Weber, N. Liu, B. Li, C. Zhang, Z. Zhao. *Optimal provable robustness of quantum classification via quantum hypothesis testing.* npj Quantum Information 7, 76 (2021).
[31] Y. Lin, J. Guan, W. Fang, M. Ying, Z. Su. *A robustness verification tool for quantum machine learning models.* Formal Methods 2024, LNCS 14933, 403–421.
[32] *Verifying Adversarial Robustness in Quantum Machine Learning: from theory to physical validation via a software tool.* (2026), arXiv:2605.29877. ⚠ full-text
[33] H.-F. Zhang et al. *Experimental robustness benchmarking of quantum neural networks on a superconducting quantum processor.* Sci. China Phys. Mech. Astron. 69, 260315 (2026). ⚠ full-text
[34] N. Franco et al. *Predominant Aspects on Security for Quantum Machine Learning* / randomized-smoothing QML (OpenReview, 2022).
[35] J. Berberich, D. Fink, C. Holm. *Training robust and generalizable quantum models.* Phys. Rev. Research (2024), arXiv:2311.11871.
[36] M. T. West et al. *Towards quantum enhanced adversarial robustness in machine learning.* Nature Machine Intelligence 5, 581 (2023).
[37] M. T. West et al. *Benchmarking adversarially robust quantum machine learning at scale.* (2022), arXiv:2211.12681.

**Classical decision-based / hard-label attacks (methods being ported)**
[9] I. Goodfellow, J. Shlens, C. Szegedy. *Explaining and harnessing adversarial examples.* ICLR 2015.
[10] W. Brendel, J. Rauber, M. Bethge. *Decision-based adversarial attacks: reliable attacks against black-box ML models.* ICLR 2018.
[11] J. Chen, M. I. Jordan, M. J. Wainwright. *HopSkipJumpAttack: a query-efficient decision-based attack.* IEEE S&P 2020, 1277–1294.
[12] M. Cheng, S. Singh, P.-H. Chen, P.-Y. Chen, S. Liu, C.-J. Hsieh. *Sign-OPT: a query-efficient hard-label adversarial attack.* ICLR 2020.
[13] Y. Liu, S.-M. Moosavi-Dezfooli, P. Frossard. *A geometry-inspired decision-based attack (qFool).* ICCV 2019.
[14] P.-Y. Chen, H. Zhang, Y. Sharma, J. Yi, C.-J. Hsieh. *ZOO: zeroth-order optimization based black-box attacks.* AISec 2017.
[15] C.-J. Simon-Gabriel, N. A. Sheikh, A. Krause. *PopSkipJump: decision-based attack for probabilistic classifiers.* ICML 2021. **(key baseline)**
[38] A. Ilyas, L. Engstrom, A. Athalye, J. Lin. *Black-box adversarial attacks with limited queries and information.* ICML 2018.
[39] J. Uesato, B. O'Donoghue, A. van den Oord, P. Kohli. *Adversarial risk and the dangers of evaluating against weak attacks (SPSA).* ICML 2018.
[40] N. Carlini, D. Wagner. *Towards evaluating the robustness of neural networks.* IEEE S&P 2017.
[41] A. Madry, A. Makelov, L. Schmidt, D. Tsipras, A. Vladu. *Towards deep learning models resistant to adversarial attacks (PGD).* ICLR 2018.
[42] C. Szegedy et al. *Intriguing properties of neural networks.* ICLR 2014.

**Gradient masking / evaluation methodology (for RQ4 framing)**
[23] A. Athalye, N. Carlini, D. Wagner. *Obfuscated gradients give a false sense of security.* ICML 2018.
[43] F. Tramèr, N. Carlini, W. Brendel, A. Madry. *On adaptive attacks to adversarial example defenses.* NeurIPS 2020.
[44] N. Papernot et al. *Practical black-box attacks against machine learning (transferability).* AsiaCCS 2017.

**Trainability / concentration (for RQ5 confound & H4)**
[24] A. Arrasmith, M. Cerezo, P. Czarnik, L. Cincio, P. J. Coles. *Effect of barren plateaus on gradient-free optimization.* Quantum 5, 558 (2021). **(key for H4)**
[25] J. R. McClean, S. Boixo, V. N. Smelyanskiy, R. Babbush, H. Neven. *Barren plateaus in quantum neural network training landscapes.* Nat. Commun. 9, 4812 (2018).
[26] S. Thanasilp, S. Wang, M. Cerezo, Z. Holmes. *Exponential concentration in quantum kernel methods.* Nat. Commun. 15, 5200 (2024).
[45] M. Larocca et al. *A review of barren plateaus in variational quantum computing.* (2024), arXiv:2405.00781.
[46] M. Cerezo et al. *Does provable absence of barren plateaus imply classical simulability?* (2023), arXiv:2312.09121.

**QML background / methods**
[47] M. Schuld, N. Killoran. *Quantum machine learning in feature Hilbert spaces.* Phys. Rev. Lett. 122, 040504 (2019).
[48] V. Havlíček et al. *Supervised learning with quantum-enhanced feature spaces.* Nature 567, 209 (2019).
[49] M. Schuld, A. Bocharov, K. Svore, N. Wiebe. *Circuit-centric quantum classifiers.* Phys. Rev. A 101, 032308 (2020).
[50] A. Pérez-Salinas, A. Cervera-Lierta, E. Gil-Fuster, J. I. Latorre. *Data re-uploading for a universal quantum classifier.* Quantum 4, 226 (2020).
[51] M. Benedetti, E. Lloyd, S. Sack, M. Fiorentini. *Parameterized quantum circuits as machine learning models.* Quantum Sci. Technol. 4, 043001 (2019).
[52] K. Mitarai, M. Negoro, M. Kitagawa, K. Fujii. *Quantum circuit learning.* Phys. Rev. A 98, 032309 (2018).
[53] M. Cerezo et al. *Variational quantum algorithms.* Nat. Rev. Phys. 3, 625 (2021).
[54] J. Preskill. *Quantum computing in the NISQ era and beyond.* Quantum 2, 79 (2018).
[55] V. Bergholm et al. *PennyLane: automatic differentiation of hybrid quantum-classical computations.* (2018/2022), arXiv:1811.04968.
[56] E. Farhi, H. Neven. *Classification with quantum neural networks on near term processors.* (2018), arXiv:1802.06002.
[57] Y. Du, M.-H. Hsieh, T. Liu, D. Tao. *Expressive power of parameterized quantum circuits.* Phys. Rev. Research 2, 033125 (2020).

**Classical robustness context**
[58] N. Papernot, P. McDaniel, S. Jha, M. Fredrikson, Z. B. Celik, A. Swami. *The limitations of deep learning in adversarial settings.* IEEE EuroS&P 2016.
[59] J. Cohen, E. Rosenfeld, Z. Kolter. *Certified adversarial robustness via randomized smoothing.* ICML 2019.
[60] M. Lécuyer, V. Atlidakis, R. Geambasu, D. Hsu, S. Jana. *Certified robustness to adversarial examples with differential privacy.* IEEE S&P 2019.

---

### Closing statement (per methodology)

> **Verification.** The stochastic-oracle model in §4.2 was checked against three analytic limits (`S→∞`, `m→0`, monotonicity) and is scheduled as sanity test T1 before any attack code is trusted. Each cited pre-emption was cross-checked against ≥2 independent sources during the search (e.g., HopSkipJump via the primary IEEE S&P paper, the author's page, and multiple citing surveys; PopSkipJump via the ICML PDF and its abstract; the Deng-group defenses via Phys. Rev. Research and the 2026 QAML survey).
>
> **Literature.** Searched across four bodies of work: (i) quantum adversarial ML attacks [1–8,19,20,27–30], (ii) quantum defenses/certification [16–18,21,22,31–37], (iii) classical decision-based/hard-label attacks [9–15,38–44], (iv) trainability/concentration [24–26,45,46]. **Found:** no decision-based hard-label attack for quantum classifiers; the closest adjacency is PopSkipJump [15] (classical, constant-noise) which becomes the key baseline. **Not found:** any gradient-free evaluation of the quantum-noise [16] or randomized-encoding [17] defenses.
>
> **Flags.** (1) **Forward-citation traversals not yet run** on [1], [11], [15] — this is the single highest-value remaining check and is milestone P0; the Deng group is the most likely pre-emptor (R1). (2) Items marked ⚠ in the reference list — especially the SoK [8] and kill-chain [20] — require full-text reads to confirm neither implements decision-based evasion (R2). (3) The `p_flip` closed form uses a CLT/Gaussian approximation; for very small `S` (≲10) a binomial treatment may be needed — to be confirmed empirically in T1. (4) The interior-optimum hypothesis (H2/RQ2) is a conjecture; the budget-curve experiment could return a monotone (boundary) optimum instead, which would still be a reportable result. (5) Reference bibliographic details (volumes, article numbers for 2026 items) should be re-verified against publisher pages at write-up.
