"""Sanity checks T1-T6, defined before the code (plan Sec. 7).

The double-step validation gate: T1-T4 must pass before any attack result is trusted.

  T1  Born-rule flip model vs Monte-Carlo, and vs PennyLane's own shot sampling
  T2  infinite-shot limit of the calibrated attack -> deterministic HSJA
  T3  attack on a synthetic dataset with an analytically known boundary
  T4  already-adversarial input -> attack returns it unchanged
  T5  budget monotonicity: larger T never yields a worse perturbation
  T6  concentration guard: at-chance clean accuracy is flagged "trivially robust"

Run:  python experiments/run_sanity.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hlq.analysis import monotone_non_increasing, save_json, validate_p_flip_on_model
from hlq.attacks import CalibratedHSJA, Domain, FixedShotHSJA
from hlq.classifier import VQC, train_or_load
from hlq.classifier import weight_shape as vqc_weight_shape
from hlq.concentration import margin_stats, trivially_robust
from hlq.config import AttackConfig, ClassifierConfig
from hlq.data import load_dataset, make_linear_boundary
from hlq.metrics import verify
from hlq.oracle import StochasticOracle, sample_label, validate_p_flip
from hlq.seeds import set_seed

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# --------------------------------------------------------------------------- #
class LinearOracleModel:
    """Analytic classifier with an exactly known boundary: f(x) = tanh(w.x + b).

    Used by T3: the true point-to-boundary distance of the *sign* boundary is
    |w.x + b| / ||w||, so the attack's recovered perturbation has a ground truth.
    """

    def __init__(self, w, b, scale=1.0):
        self.w = np.asarray(w, float)
        self.b = float(b)
        self.scale = float(scale)

    def decision_function(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        return np.tanh(self.scale * (X @ self.w + self.b))

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1).astype(np.int64)

    def true_distance(self, x):
        return abs(float(x @ self.w + self.b)) / float(np.linalg.norm(self.w))


# --------------------------------------------------------------------------- #
def t1_flip_model(quick=False) -> dict:
    """T1: closed form vs Monte-Carlo, plus a cross-check against PennyLane shots."""
    out = validate_p_flip(trials=4000 if quick else 20000, seed=0)

    # Independent cross-check: our exact-Binomial oracle must reproduce PennyLane's
    # own shot-based sampling for the same state (they must agree in distribution).
    import pennylane as qml

    n, S, reps = 3, 200, 400
    rng = np.random.default_rng(0)
    dev_a = qml.device("default.qubit", wires=n, shots=S)
    dev_e = qml.device("default.qubit", wires=n)
    angles = rng.uniform(0, np.pi, size=n)

    def _c():
        qml.AngleEmbedding(angles, wires=range(n), rotation="Y")
        return qml.expval(qml.PauliZ(0))

    f_exact = float(qml.QNode(_c, dev_e)())
    pl_labels = np.array([1 if float(qml.QNode(_c, dev_a)()) >= 0 else -1 for _ in range(reps)])
    our_labels = sample_label(np.full(reps, f_exact), S, rng)
    p_pl = float(np.mean(pl_labels == 1))
    p_our = float(np.mean(our_labels == 1))
    se = float(np.sqrt(max(p_pl * (1 - p_pl), 1e-9) / reps) + np.sqrt(max(p_our * (1 - p_our), 1e-9) / reps))
    out["pennylane_crosscheck"] = {
        "f_exact": f_exact, "p_plus_pennylane": p_pl, "p_plus_ours": p_our,
        "abs_diff": abs(p_pl - p_our), "tolerance_3se": 3 * se + 0.02,
        "agrees": bool(abs(p_pl - p_our) <= 3 * se + 0.02),
    }
    out["passed"] = bool(out["passed"] and out["pennylane_crosscheck"]["agrees"])
    return out


def t2_infinite_shot_limit(quick=False) -> dict:
    """T2: as S -> infinity the calibrated attack reproduces deterministic HSJA.

    A single global shot-scale sigma raises BOTH shot budgets that make the oracle
    stochastic -- the boundary-decision cap (which sets the resolution tau ~ z/sqrt(cap))
    and the probe shots -- while the probe COUNT and the total budget are held fixed, so
    sigma is the only thing varying.  The budget is deliberately unbounded here: this
    test isolates the S->inf limit, not budget economics (that is RQ2), and a finite cap
    would truncate the walk and confound the two.
    """
    cfg = ClassifierConfig(n_qubits=4, n_layers=2, dataset="synthetic", epochs=12, seed=0)
    b = load_dataset(cfg)
    vqc, _ = train_or_load(cfg, b)
    dom = Domain("box", 0.0, float(np.pi))
    idxs = np.where(vqc.predict(b.X_test) == b.y_test)[0][: (4 if quick else 8)]
    UNBOUNDED = 10 ** 15
    B_PROBES = 100

    def _run(mk, acfg):
        """Per-image perturbations (paired across conditions on the same x0)."""
        ps = []
        for idx in idxs:
            x0, y0 = b.X_test[idx], int(b.y_test[idx])
            pool = b.X_train[b.y_train != y0][:30]
            rng = set_seed(0)
            orc = StochasticOracle(vqc, rng, budget=UNBOUNDED)
            r = mk(acfg).attack(orc, x0, y0, pool, rng, dom)
            v = verify(vqc, x0, r.x_adv, y0)
            ps.append(v["verified_perturbation"])
        return np.array(ps, float)

    def _paired_gap(a, ref):
        """Median |a - ref| / ref over images where BOTH succeeded.

        Paired per-image: the perturbation varies by ~2x across x0, so comparing
        medians of different small samples measures image variance, not convergence.
        """
        m = np.isfinite(a) & np.isfinite(ref)
        if not m.any():
            return float("nan")
        return float(np.median(np.abs(a[m] - ref[m]) / np.maximum(ref[m], 1e-9)))

    # Reference: the deterministic (infinite-shot) oracle -- standard HSJA.
    det_cfg = AttackConfig(iterations=12, total_budget=UNBOUNDED, fixed_shots=None,
                           init_eval_shots=None, num_probes=B_PROBES, probe_shots=None,
                           seed=0)
    det_per_image = _run(FixedShotHSJA, det_cfg)
    det_ok = det_per_image[np.isfinite(det_per_image)]
    det_pert = float(np.median(det_ok)) if len(det_ok) else float("nan")
    det_succ = float(np.mean(np.isfinite(det_per_image)))

    # The scales must be large: a probe sits ~delta from the boundary where |f| ~ 0.03,
    # and p_flip(0.03, S) only approaches 0 for S >> 1/|f|^2 ~ 1e3.  Shots are free in the
    # analytic Born oracle (one Binomial draw per query), so sigma can sweep decades.
    scales = [1, 100, 10_000] if quick else [1, 10, 100, 1_000, 10_000]
    caps, probe_shots, perts, succs, gaps = [], [], [], [], []
    for s in scales:
        cap = 500 * s
        sp = 100 * s
        acfg = AttackConfig(iterations=12, total_budget=UNBOUNDED, delta_decision=0.05,
                            fixed_shots=100, num_probes=B_PROBES, probe_shots=sp, seed=0)

        def _mk(c, cap=cap, sp=sp):
            a = CalibratedHSJA(c, decision_cap=cap)
            a.force_probe_shots = sp              # scale probe shots with sigma too
            a.force_num_probes = B_PROBES         # hold the probe COUNT fixed
            return a

        per_image = _run(_mk, acfg)
        ok = per_image[np.isfinite(per_image)]
        caps.append(cap)
        probe_shots.append(sp)
        perts.append(float(np.median(ok)) if len(ok) else float("nan"))
        succs.append(float(np.mean(np.isfinite(per_image))))
        gaps.append(_paired_gap(per_image, det_per_image))

    return {
        "test": "T2_infinite_shot_limit",
        "deterministic_hsja": {"median_perturbation": det_pert, "success_rate": det_succ,
                               "per_image": det_per_image.tolist()},
        "shot_scales": scales, "decision_caps": caps, "probe_shots": probe_shots,
        "calibrated_median_perturbation": perts, "calibrated_success_rate": succs,
        "paired_relative_gap": gaps,
        "gap_shrinks_with_shots": bool(gaps[-1] <= gaps[0] + 1e-6),
        # paired per-image convergence to the deterministic limit within 15%
        "passed": bool(np.isfinite(gaps[-1]) and gaps[-1] < 0.15
                       and gaps[-1] <= gaps[0] + 1e-6 and succs[-1] >= 0.7),
    }


def t3_known_boundary(quick=False) -> dict:
    """T3: recovered perturbation ~ true point-to-boundary distance (analytic model)."""
    n = 4
    w, b0 = make_linear_boundary(n, seed=0)
    model = LinearOracleModel(w, b0, scale=3.0)
    dom = Domain("box", 0.0, float(np.pi))
    rng0 = np.random.default_rng(1)
    X = rng0.uniform(0, np.pi, size=(40, n))
    X = X[np.abs(X @ w + b0) > 0.2][: (4 if quick else 10)]

    acfg = AttackConfig(iterations=25, total_budget=10 ** 12, fixed_shots=None,
                        init_eval_shots=None, num_probes=150, probe_shots=None, seed=0)
    rec, rel = [], []
    for x0 in X:
        y0 = int(model.predict(x0.reshape(1, -1))[0])
        pool = rng0.uniform(0, np.pi, size=(400, n))
        pool = pool[model.predict(pool) != y0][:30]
        rng = set_seed(0)
        orc = StochasticOracle(model, rng, budget=10 ** 12)
        r = FixedShotHSJA(acfg).attack(orc, x0, y0, pool, rng, dom)
        truth = model.true_distance(x0)
        rec.append({"recovered": float(r.perturbation), "true_distance": float(truth),
                    "rel_error": float(abs(r.perturbation - truth) / max(truth, 1e-9))})
        rel.append(rec[-1]["rel_error"])
    med_rel = float(np.median(rel))
    return {"test": "T3_known_boundary", "per_point": rec,
            "median_relative_error": med_rel, "tolerance": 0.15,
            "passed": bool(med_rel < 0.15)}


def t4_already_adversarial(quick=False) -> dict:
    """T4: an input that is already adversarial must be returned unchanged."""
    cfg = ClassifierConfig(n_qubits=4, n_layers=2, dataset="synthetic", epochs=12, seed=0)
    b = load_dataset(cfg)
    vqc, _ = train_or_load(cfg, b)
    dom = Domain("box", 0.0, float(np.pi))
    # points the model gets WRONG: relative to the true label they are already adversarial
    wrong = np.where(vqc.predict(b.X_test) != b.y_test)[0][: (3 if quick else 8)]
    acfg = AttackConfig(iterations=10, total_budget=200000, delta_decision=0.05,
                        fixed_shots=100, num_probes=60, probe_shots=60, seed=0)
    perts, flags = [], []
    for idx in wrong:
        x0, y0 = b.X_test[idx], int(b.y_test[idx])      # y0 = TRUE label; model disagrees
        pool = b.X_train[b.y_train != y0][:20]
        rng = set_seed(0)
        orc = StochasticOracle(vqc, rng, budget=200000)
        r = CalibratedHSJA(acfg).attack(orc, x0, y0, pool, rng, dom)
        perts.append(float(r.perturbation))
        flags.append(bool(r.meta.get("already_adversarial", False)))
    return {"test": "T4_already_adversarial", "n_points": int(len(wrong)),
            "perturbations": perts, "flagged_trivial": flags,
            "all_zero_perturbation": bool(all(p == 0.0 for p in perts)),
            "passed": bool(len(wrong) > 0 and all(flags) and all(p == 0.0 for p in perts))}


def t5_budget_monotonicity(quick=False) -> dict:
    """T5: a larger measurement budget must never give a worse median perturbation."""
    cfg = ClassifierConfig(n_qubits=4, n_layers=2, dataset="synthetic", epochs=12, seed=0)
    b = load_dataset(cfg)
    vqc, _ = train_or_load(cfg, b)
    dom = Domain("box", 0.0, float(np.pi))
    idxs = np.where(vqc.predict(b.X_test) == b.y_test)[0][: (4 if quick else 10)]
    budgets = [20000, 60000] if quick else [10000, 30000, 100000, 300000]
    per_budget = []
    for T in budgets:
        acfg = AttackConfig(iterations=15, total_budget=T, delta_decision=0.05,
                            fixed_shots=100, num_probes=80, probe_shots=60, seed=0)
        ps = []
        for idx in idxs:
            x0, y0 = b.X_test[idx], int(b.y_test[idx])
            pool = b.X_train[b.y_train != y0][:30]
            rng = set_seed(0)
            orc = StochasticOracle(vqc, rng, budget=T)
            r = CalibratedHSJA(acfg).attack(orc, x0, y0, pool, rng, dom)
            v = verify(vqc, x0, r.x_adv, y0)
            ps.append(v["verified_perturbation"])
        per_budget.append(np.array(ps, float))

    # PAIRED over a COMMON success set: perturbation varies ~10x across images, and the
    # set of images that succeed changes with T, so medians of different subsets measure
    # image variance and selection, not the budget trend.
    M = np.vstack(per_budget)
    common = np.all(np.isfinite(M), axis=0)
    meds_all = [float(np.median(p[np.isfinite(p)])) if np.isfinite(p).any() else float("nan")
                for p in per_budget]
    meds = ([float(np.median(p[common])) for p in per_budget] if common.any()
            else meds_all)
    tol = 0.10 * float(np.nanmax(meds)) if np.isfinite(meds).any() else 0.05
    return {"test": "T5_budget_monotonicity", "budgets": budgets,
            "median_perturbation_common_set": meds,
            "median_perturbation_all_successes": meds_all,
            "n_common_images": int(common.sum()), "n_images": int(len(idxs)),
            "tolerance": tol,
            "monotone_non_increasing": monotone_non_increasing(budgets, meds, tol=tol),
            "passed": bool(common.sum() >= 2
                           and monotone_non_increasing(budgets, meds, tol=tol))}


def t6_concentration_guard(quick=False) -> dict:
    """T6: the RQ5 guardrail must fire on a concentrated, at-chance model -- and only then.

    Exponential concentration is a property of RANDOM/deep circuits, not of a trained
    model on a learnable task (training actively fights it), so the guard is exercised
    against randomly-initialised deep circuits with the global observable Z^{otimes n} --
    the named concentration source.  Their decision value must collapse with n and their
    accuracy fall to chance, at which point the model must be reported "trivially robust"
    rather than "defended".  A trained, above-chance model must NOT be flagged.
    Whether *trained* models concentrate is an empirical question -- that is RQ5, not a
    sanity check.
    """
    ns = [4, 6, 8] if quick else [4, 6, 8, 10, 12]
    rows = []
    for n in ns:
        cfg = ClassifierConfig(n_qubits=n, n_layers=10, dataset="synthetic",
                               observable="global_z", epochs=1, seed=0)
        b = load_dataset(cfg)
        vqc = VQC(cfg)                                     # RANDOM (untrained) weights
        rng = np.random.default_rng(0)
        vqc.weights = rng.uniform(0, 2 * np.pi, size=vqc_weight_shape(cfg))
        acc = vqc.clean_accuracy(b.X_test[:400], b.y_test[:400])
        ms = margin_stats(vqc, b.X_test[:400])
        rows.append({"n_qubits": n, "observable": "global_z", "regime": "random_deep",
                     "clean_accuracy": float(acc),
                     "trivially_robust": trivially_robust(acc), **ms})

    # control: a trained, genuinely above-chance model must NOT be flagged
    cfg_t = ClassifierConfig(n_qubits=4, n_layers=2, dataset="synthetic", epochs=12, seed=0)
    bt = load_dataset(cfg_t)
    vqc_t, _ = train_or_load(cfg_t, bt)
    acc_t = vqc_t.clean_accuracy(bt.X_test, bt.y_test)
    trained = {"n_qubits": 4, "observable": "local_z", "regime": "trained",
               "clean_accuracy": float(acc_t), "trivially_robust": trivially_robust(acc_t),
               **margin_stats(vqc_t, bt.X_test[:400])}

    var = [r["var_f"] for r in rows]
    fired = bool(any(r["trivially_robust"] for r in rows))
    decays = bool(var[-1] < var[0])
    return {"test": "T6_concentration_guard", "random_deep_global_z": rows,
            "trained_control": trained,
            "var_f_collapses_with_n": decays, "var_f": var,
            "guard_fires_on_concentrated_model": fired,
            "guard_silent_on_trained_model": bool(not trained["trivially_robust"]),
            "passed": bool(decays and fired and not trained["trivially_robust"])}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Sanity tests T1-T6")
    ap.add_argument("--quick", action="store_true", help="smaller sweeps for a fast gate")
    ap.add_argument("--only", default=None, help="comma-separated subset, e.g. T1,T3")
    ap.add_argument("--out", default=os.path.join(RESULTS, "sanity.json"))
    args = ap.parse_args()

    tests = {"T1": t1_flip_model, "T2": t2_infinite_shot_limit, "T3": t3_known_boundary,
             "T4": t4_already_adversarial, "T5": t5_budget_monotonicity,
             "T6": t6_concentration_guard}
    if args.only:
        tests = {k: v for k, v in tests.items() if k in args.only.split(",")}

    results = {}
    if args.only and os.path.exists(args.out):     # merge, don't clobber other tests
        with open(args.out) as fh:
            results = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    for name, fn in tests.items():
        print(f"[sanity] running {name} ...", flush=True)
        try:
            results[name] = fn(args.quick)
            print(f"[sanity] {name}: passed={results[name].get('passed')}", flush=True)
        except Exception as exc:                     # keep going; record the failure
            results[name] = {"test": name, "passed": False, "error": repr(exc)}
            print(f"[sanity] {name}: ERROR {exc!r}", flush=True)

    gate = all(results[t].get("passed") for t in ("T1", "T2", "T3", "T4") if t in results)
    results["_summary"] = {
        "passed": {k: bool(v.get("passed")) for k, v in results.items()},
        "trust_gate_T1_T4": bool(gate),
        "note": "Plan Sec. 7: code is not trusted until T1-T4 pass.",
    }
    save_json(results, args.out)
    print(f"\n[sanity] wrote {args.out}")
    print(f"[sanity] T1-T4 trust gate: {'PASS' if gate else 'FAIL'}")
    for k, v in results["_summary"]["passed"].items():
        print(f"   {k}: {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
