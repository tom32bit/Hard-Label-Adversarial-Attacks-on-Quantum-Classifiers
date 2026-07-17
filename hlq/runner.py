"""The experiment cell: one (model x defense x attack x seed) -> verified metrics.

Every research question in the plan is a *composition* of cells, so this is the only
place that knows how to run an attack end-to-end.  Cells are the unit of parallelism
(the driver maps them across processes); they take plain configs, so workers rebuild
from cache rather than pickling live QNodes.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .attacks import BUILDERS, Domain, pgd_attack, transfer_attack
from .classical import MatchedClassicalNN
from .classifier import train_or_load
from .concentration import margin_stats, robustness_guardrail
from .config import AttackConfig, ClassifierConfig, DefenseConfig
from .data import load_dataset
from .defenses import wrap_defense
from .metrics import (queries_to_threshold, robust_accuracy, shots_to_threshold,
                      summarize_attack, verify)
from .oracle import StochasticOracle
from .seeds import set_seed


def build_domain(cfg: ClassifierConfig) -> Domain:
    """Valid input region for the encoder: angles live in [0, pi]; amplitudes on S^{d-1}."""
    if cfg.encoding == "amplitude":
        return Domain("sphere")
    return Domain("box", 0.0, float(np.pi))


def _verify(clf, x0, x_adv, y0, stochastic: bool, n_samples: int = 25) -> dict:
    """Ground-truth check. For a per-query-stochastic defense the deployed decision is
    the expectation over the randomisation, so average f over ``n_samples`` draws."""
    if not stochastic:
        return verify(clf, x0, x_adv, y0)
    f = float(np.mean([clf.decision_function(np.asarray(x_adv).reshape(1, -1))[0]
                       for _ in range(n_samples)]))
    truly_adv = (1 if f >= 0 else -1) != int(y0)
    return {"verified_success": bool(truly_adv),
            "verified_perturbation": (float(np.linalg.norm(np.asarray(x_adv) - np.asarray(x0)))
                                      if truly_adv else float("inf")),
            "exact_margin_at_adv": abs(f)}


def select_attack_images(clf, bundle, n_images: int, stochastic=False) -> np.ndarray:
    """Indices of the first ``n_images`` test points the DEPLOYED model gets right.

    Attacking already-misclassified points is meaningless, so the evaluation set is
    always clean-correct under the model actually being attacked (defense included).
    """
    pred = clf.predict(bundle.X_test)
    idx = np.where(pred == bundle.y_test)[0]
    return idx[:n_images]


def run_cell(clf_cfg: ClassifierConfig, def_cfg: DefenseConfig, atk_cfg: AttackConfig,
             n_images: int = 250, eps: float = 0.5, force_probe_shots=None,
             verbose: bool = False) -> dict:
    """Run one full experiment cell and return a JSON-ready record."""
    bundle = load_dataset(clf_cfg)
    domain = build_domain(clf_cfg)

    # -- model under attack (classical anchor swaps in a matched NN) ---------- #
    if atk_cfg.name == "classical_hsja":
        model = MatchedClassicalNN(clf_cfg)
        info = model.fit(bundle.X_train, bundle.y_train)
        info["test_acc"] = model.clean_accuracy(bundle.X_test, bundle.y_test)
        deployed, stochastic = model, False
        # A classical NN returns a DETERMINISTIC label: there is no Born rule and no
        # shot noise to sample. Sampling labels from its output would invent a
        # stochastic oracle that does not exist and destroy the anchor's meaning --
        # the anchor exists precisely to show what the walk costs WITHOUT shot noise.
        atk_cfg = replace(atk_cfg, fixed_shots=None, probe_shots=None,
                          init_eval_shots=None, total_budget=10 ** 15)
    else:
        model, info = train_or_load(clf_cfg, bundle)
        deployed, stochastic = wrap_defense(model, def_cfg)

    clean_acc = deployed.clean_accuracy(bundle.X_test, bundle.y_test)
    idxs = select_attack_images(deployed, bundle, n_images, stochastic)

    records, perts, succs = [], [], []
    for i, idx in enumerate(idxs):
        x0, y0 = bundle.X_test[idx], int(bundle.y_test[idx])
        pool = bundle.X_train[bundle.y_train != y0][:40]
        rng = set_seed(atk_cfg.seed * 100003 + int(idx))       # per-image reproducibility

        if atk_cfg.name == "pgd_whitebox":
            if not hasattr(deployed, "decision_function_torch"):
                continue                                        # white-box needs gradients
            res = pgd_attack(deployed, x0, y0, domain, atk_cfg)
            v = _verify(deployed, x0, res["x_adv"], y0, stochastic)
            rec = {"queries": res["queries"], "shots": res["shots"],
                   "grad_steps": res.get("grad_steps", 0)}
        elif atk_cfg.name == "transfer":
            orc = StochasticOracle(deployed, rng, stochastic=stochastic)
            res = transfer_attack(deployed, orc, x0, y0, bundle.X_train[:300],
                                  domain, atk_cfg, rng)
            v = _verify(deployed, x0, res["x_adv"], y0, stochastic)
            rec = {"queries": res["queries"], "shots": res["shots"]}
        else:
            orc = StochasticOracle(deployed, rng, stochastic=stochastic,
                                   budget=atk_cfg.total_budget)
            atk = BUILDERS[atk_cfg.name](atk_cfg)
            if force_probe_shots is not None and hasattr(atk, "force_probe_shots"):
                atk.force_probe_shots = force_probe_shots       # RQ2 allocation sweep
            out = atk.attack(orc, x0, y0, pool, rng, domain)
            v = _verify(deployed, x0, out.x_adv, y0, stochastic)
            rec = out.to_record()
            rec["circuit_evals"] = int(orc.n_circuit_evals)
            # cost-to-reach-eps, computed from the trajectory before it is discarded
            rec["queries_to_eps"] = queries_to_threshold(rec, v["verified_success"], eps)
            rec["shots_to_eps"] = shots_to_threshold(rec, v["verified_success"], eps)
            if not verbose:      # trajectories are large; keep them only when asked
                for k in ("dist_trajectory", "query_trajectory", "shot_trajectory",
                          "m2_log"):
                    rec.pop(k, None)

        rec.update(image_index=int(idx), **v)
        records.append(rec)
        perts.append(v["verified_perturbation"])
        succs.append(v["verified_success"])

    def _median_finite(key):
        v = np.array([r.get(key, np.inf) for r in records], float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else float("nan")

    summary = summarize_attack(perts, succs, cap=None)
    summary.update(
        robust_accuracy_at_eps=robust_accuracy(perts, succs, eps),
        eps=eps,
        # plan Sec. 6: cost to REACH the eps threshold (inf/excluded if never reached)
        median_queries_to_eps=_median_finite("queries_to_eps"),
        median_shots_to_eps=_median_finite("shots_to_eps"),
        frac_reaching_eps=float(np.mean([np.isfinite(r.get("queries_to_eps", np.inf))
                                         for r in records])) if records else float("nan"),
        clean_accuracy=clean_acc,
        train_test_acc=info.get("test_acc", float("nan")),
        median_queries=float(np.median([r["queries"] for r in records])) if records else float("nan"),
        median_shots=float(np.median([r["shots"] for r in records])) if records else float("nan"),
    )
    summary.update(robustness_guardrail(clean_acc, summary["median_perturbation"]))
    summary.update(margin_stats(deployed if not stochastic else model, bundle.X_test[:400]))
    return {
        "classifier": clf_cfg.to_dict(), "defense": def_cfg.to_dict(),
        "attack": atk_cfg.to_dict(), "n_images": int(len(idxs)),
        "force_probe_shots": force_probe_shots,
        "summary": summary, "per_image": records,
    }
