"""Experiment driver: composes cells into the plan's research questions.

Every RQ is a set of :func:`hlq.runner.run_cell` calls that vary exactly one axis with
principled defaults for the rest (the plan's ablation design, Sec. 4.4).  Cells are the
unit of parallelism.  ``config.json`` is dumped at the start of every run and results
land in ``results/<rq>.json`` with the plan's schema
``{"metric": {"mean":..., "std":..., "seeds":[...]}}``.

Run:  python experiments/driver.py --rq rq1 --preset smoke
      python experiments/driver.py --rq all --preset full --jobs 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib
import json

from hlq.analysis import _default, budget_curve, find_interior_optimum, save_json
from hlq.classifier import train_or_load
from hlq.concentration import fit_exponential_concentration
from hlq.config import AttackConfig, ClassifierConfig, DefenseConfig
from hlq.data import load_dataset
from hlq.metrics import seed_aggregate
from hlq.runner import run_cell

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# --------------------------------------------------------------------------- #
# Presets: scale knobs only -- the experimental design is identical across them.
# --------------------------------------------------------------------------- #
PRESETS = {
    "smoke":  dict(n_images=4,   seeds=(0, 1),                   iterations=8,  budget=20_000),
    "medium": dict(n_images=40,  seeds=(0, 1, 2),                iterations=15, budget=60_000),
    # "kaggle": tuned to finish RQ4+RQ5 inside a single ~12h CPU session, with real
    # statistics (3 seeds).  The heavy blocks additionally cap their most expensive axis
    # internally (RQ4 -> n=4 density-matrix; RQ5 -> n<=10 by default).
    "kaggle": dict(n_images=60,  seeds=(0, 1, 2),                iterations=20, budget=80_000),
    "full":   dict(n_images=250, seeds=(0, 1, 2, 3, 4, 5, 6, 7), iterations=30, budget=200_000),
}

# RQ5 qubit ladder. n=12 statevector TRAINING dominates the whole run (tens of minutes
# per model), so it is opt-in via HLQ_RQ5_MAX_N; the default stops at 10.
_RQ5_MAX_N = int(os.environ.get("HLQ_RQ5_MAX_N", "10"))

# Principled defaults held fixed while one axis varies (plan Sec. 4.1/5).
DEFAULT_N, DEFAULT_L, DEFAULT_ENC, DEFAULT_OBS = 8, 5, "angle", "local_z"
DEFAULT_DATASET = "mnist_3v5"


def _clf(seed, **kw) -> ClassifierConfig:
    base = dict(n_qubits=DEFAULT_N, n_layers=DEFAULT_L, encoding=DEFAULT_ENC,
                observable=DEFAULT_OBS, dataset=DEFAULT_DATASET, seed=seed)
    base.update(kw)
    return ClassifierConfig(**base)


def _atk(name, seed, P, **kw) -> AttackConfig:
    base = dict(name=name, seed=seed, iterations=P["iterations"],
                total_budget=P["budget"], delta_decision=0.05, fixed_shots=100,
                num_probes=80, probe_shots=60)
    base.update(kw)
    return AttackConfig(**base)


# --------------------------------------------------------------------------- #
def warmup_models(clf_cfgs, verbose=True):
    """Train/cache every unique classifier serially BEFORE parallel cells run.

    Parallel workers would otherwise race to write the same model-cache file.
    """
    seen = {}
    for c in clf_cfgs:
        if c.key() in seen or c.dataset == "synthetic" and False:
            continue
        seen[c.key()] = c
    for i, (k, c) in enumerate(seen.items(), 1):
        t0 = time.time()
        b = load_dataset(c)
        _, info = train_or_load(c, b)
        if verbose:
            print(f"  [warmup {i}/{len(seen)}] {k}  test_acc={info.get('test_acc', float('nan')):.3f}"
                  f"  ({time.time()-t0:.1f}s)", flush=True)
    return list(seen.values())


# --------------------------------------------------------------------------- #
# Per-cell checkpointing + resume.  A cell is the unit of work AND of durability:
# each completed cell is written to results/checkpoints/<rq>/<id>.json the instant it
# finishes, so a session that hits Kaggle's 12h wall keeps every cell it computed, and
# re-running skips them.  Workers write distinct files, so there is no contention.
# --------------------------------------------------------------------------- #
_CKPT_DIR = None


def _cell_id(task) -> str:
    payload = {"clf": task["clf_cfg"].to_dict(), "def": task["def_cfg"].to_dict(),
               "atk": task["atk_cfg"].to_dict(), "n_images": task.get("n_images"),
               "force_probe_shots": task.get("force_probe_shots")}
    return hashlib.md5(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _run_cell_ckpt(task, ckpt_dir):
    """Run one cell, or load it from a checkpoint if it was already computed."""
    path = os.path.join(ckpt_dir, _cell_id(task) + ".json")
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception:
            pass                                   # corrupt/partial -> recompute
    res = run_cell(**task)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, default=_default)
    os.replace(tmp, path)                          # atomic publish
    return res


def _run_all(jobs, tasks):
    """Execute cells (parallel when jobs>1), checkpointing each when _CKPT_DIR is set."""
    ckpt = _CKPT_DIR
    if ckpt:
        os.makedirs(ckpt, exist_ok=True)
        fn = lambda t: _run_cell_ckpt(t, ckpt)
        done = sum(os.path.exists(os.path.join(ckpt, _cell_id(t) + ".json")) for t in tasks)
        if done:
            print(f"  [resume] {done}/{len(tasks)} cells already checkpointed", flush=True)
    else:
        fn = lambda t: run_cell(**t)
    if jobs <= 1:
        return [fn(t) for t in tasks]
    from joblib import Parallel, delayed
    if ckpt:
        return Parallel(n_jobs=jobs, backend="loky", verbose=5)(
            delayed(_run_cell_ckpt)(t, ckpt) for t in tasks)
    return Parallel(n_jobs=jobs, backend="loky", verbose=5)(
        delayed(run_cell)(**t) for t in tasks)


def _agg(cells, metric):
    return seed_aggregate([c["summary"].get(metric, float("nan")) for c in cells])


def _group(cells, keyfn):
    out = {}
    for c in cells:
        out.setdefault(keyfn(c), []).append(c)
    return out


METRICS = ("success_rate", "median_perturbation", "median_queries", "median_shots",
           "robust_accuracy_at_eps", "clean_accuracy")


def _summarize(groups) -> dict:
    """Aggregate each condition's cells over seeds into the plan's results schema."""
    return {k: {m: _agg(v, m) for m in METRICS} for k, v in groups.items()}


# --------------------------------------------------------------------------- #
# RQ1: feasibility -- can a hard-label attack fool a VQC, and at what query cost?
# --------------------------------------------------------------------------- #
def rq1(P, jobs):
    attacks = ["calibrated_hsja", "fixed_hsja", "popskipjump", "pgd_whitebox",
               "transfer", "classical_hsja"]
    tasks, clfs = [], []
    for s in P["seeds"]:
        c = _clf(s)
        clfs.append(c)
        for a in attacks:
            tasks.append(dict(clf_cfg=c, def_cfg=DefenseConfig("none"),
                              atk_cfg=_atk(a, s, P), n_images=P["n_images"]))
    warmup_models(clfs)
    cells = _run_all(jobs, tasks)
    return {"cells": cells,
            "aggregated": _summarize(_group(cells, lambda c: c["attack"]["name"]))}


# --------------------------------------------------------------------------- #
# RQ2: query/shot economics -- budget curve, probe-allocation sweep, M1/M2 split
# --------------------------------------------------------------------------- #
def rq2(P, jobs):
    out = {}
    clfs = [_clf(s) for s in P["seeds"]]
    warmup_models(clfs)

    # (a) budget curve: perturbation vs total measurement budget T
    budgets = [P["budget"] // 8, P["budget"] // 4, P["budget"] // 2, P["budget"],
               P["budget"] * 2]
    tasks = [dict(clf_cfg=_clf(s), def_cfg=DefenseConfig("none"),
                  atk_cfg=_atk("calibrated_hsja", s, P, total_budget=T),
                  n_images=P["n_images"])
             for s in P["seeds"] for T in budgets]
    cells = _run_all(jobs, tasks)
    g = _group(cells, lambda c: c["attack"]["total_budget"])
    agg = _summarize(g)
    out["budget_curve"] = {
        "budgets": budgets, "aggregated": agg,
        "curve": budget_curve([{"total_budget": int(k),
                                "median_perturbation": v["median_perturbation"]["mean"]}
                               for k, v in agg.items()]),
        "cells": cells,
    }

    # (b) allocation sweep: perturbation vs FORCED probe shots S_probe at fixed T
    #     (tests H2's interior optimum against the plateau the theory predicts)
    s_grid = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    tasks = [dict(clf_cfg=_clf(s), def_cfg=DefenseConfig("none"),
                  atk_cfg=_atk("calibrated_hsja", s, P), n_images=P["n_images"],
                  force_probe_shots=sp)
             for s in P["seeds"] for sp in s_grid]
    cells = _run_all(jobs, tasks)
    g = _group(cells, lambda c: c["force_probe_shots"])
    agg = _summarize(g)
    xs = sorted(g.keys())
    ys = [agg[x]["median_perturbation"]["mean"] for x in xs]
    out["allocation_sweep"] = {
        "probe_shots": xs, "median_perturbation": ys, "aggregated": agg,
        "interior_optimum": find_interior_optimum(xs, ys), "cells": cells,
    }

    # (c) M1/M2 split: fraction of the per-iteration budget given to normal estimation
    fracs = [0.1, 0.25, 0.5, 0.75, 0.9]
    tasks = [dict(clf_cfg=_clf(s), def_cfg=DefenseConfig("none"),
                  atk_cfg=_atk("calibrated_hsja", s, P, grad_budget_frac=fr),
                  n_images=P["n_images"])
             for s in P["seeds"] for fr in fracs]
    cells = _run_all(jobs, tasks)
    g = _group(cells, lambda c: c["attack"]["grad_budget_frac"])
    agg = _summarize(g)
    xs = sorted(g.keys())
    ys = [agg[x]["median_perturbation"]["mean"] for x in xs]
    out["m1_m2_split"] = {
        "grad_budget_frac": xs, "median_perturbation": ys, "aggregated": agg,
        "interior_optimum": find_interior_optimum(xs, ys), "cells": cells,
    }
    return out


# --------------------------------------------------------------------------- #
# RQ3: does Born-rule calibration beat a constant-noise treatment at equal budget?
# --------------------------------------------------------------------------- #
def rq3(P, jobs):
    methods = ["calibrated_hsja", "popskipjump", "fixed_hsja"]
    budgets = [P["budget"] // 4, P["budget"], P["budget"] * 2]
    clfs = [_clf(s) for s in P["seeds"]]
    warmup_models(clfs)
    tasks = [dict(clf_cfg=_clf(s), def_cfg=DefenseConfig("none"),
                  atk_cfg=_atk(m, s, P, total_budget=T), n_images=P["n_images"])
             for s in P["seeds"] for m in methods for T in budgets]
    cells = _run_all(jobs, tasks)
    g = _group(cells, lambda c: (c["attack"]["name"], c["attack"]["total_budget"]))
    return {"cells": cells,
            "aggregated": {f"{k[0]}@T={k[1]}": {m: _agg(v, m) for m in METRICS}
                           for k, v in g.items()},
            "methods": methods, "budgets": budgets}


# --------------------------------------------------------------------------- #
# RQ4: first gradient-free evaluation of the quantum-noise / randomized defenses
# --------------------------------------------------------------------------- #
def rq4(P, jobs):
    defenses = [DefenseConfig("none"),
                DefenseConfig("depolarizing", depolarizing_p=0.05),
                DefenseConfig("randomized_encoding", randomized_strength=0.30)]
    attacks = ["calibrated_hsja", "pgd_whitebox"]
    # The depolarizing defense uses density-matrix simulation (O(4^n), no broadcasting),
    # which dominates cost, so RQ4 runs at n=4 (256-dim) -- ~16x faster than n=6 -- with a
    # reduced image count and a lighter attack budget. The mechanism (margin collapse ->
    # gradient-free robustness) is qubit-count-independent; RQ5 handles the n sweep.
    n_def = min(DEFAULT_N, 4)
    n_img = max(8, P["n_images"] // 3)
    def4 = dict(iterations=min(P["iterations"], 15),
                total_budget=min(P["budget"], 40_000))
    clfs = [_clf(s, n_qubits=n_def) for s in P["seeds"]]
    warmup_models(clfs)
    tasks = [dict(clf_cfg=_clf(s, n_qubits=n_def), def_cfg=d,
                  atk_cfg=_atk(a, s, P, **def4), n_images=n_img)
             for s in P["seeds"] for d in defenses for a in attacks]
    cells = _run_all(jobs, tasks)
    g = _group(cells, lambda c: (c["defense"]["name"], c["attack"]["name"]))
    return {"cells": cells, "n_qubits": n_def,
            "aggregated": {f"{k[0]}|{k[1]}": {m: _agg(v, m) for m in METRICS}
                           for k, v in g.items()},
            "note": "n capped for the density-matrix (default.mixed) noise defense; "
                    "attacked-image count reduced for these O(4^n) cells."}


# --------------------------------------------------------------------------- #
# RQ5: is apparent robustness really exponential concentration?
# --------------------------------------------------------------------------- #
def rq5(P, jobs):
    ns = [n for n in (4, 6, 8, 10, 12) if n <= _RQ5_MAX_N]     # n=12 opt-in (see top)
    obs = ["local_z", "global_z"]
    n_img = max(8, P["n_images"] // 4)

    # Training the large-n models is the cost; fewer epochs still exhibit the
    # concentration effect this RQ measures (var[f] collapse for the global observable).
    def _c5(s, n, o):
        return _clf(s, n_qubits=n, observable=o, epochs=min(40, 25 if n >= 8 else 40))

    clfs = [_c5(s, n, o) for s in P["seeds"] for n in ns for o in obs]
    warmup_models(clfs)
    tasks = [dict(clf_cfg=_c5(s, n, o), def_cfg=DefenseConfig("none"),
                  atk_cfg=_atk("calibrated_hsja", s, P), n_images=n_img)
             for s in P["seeds"] for n in ns for o in obs]
    cells = _run_all(jobs, tasks)
    g = _group(cells, lambda c: (c["classifier"]["observable"], c["classifier"]["n_qubits"]))
    agg = {f"{k[0]}|n={k[1]}": {m: _agg(v, m) for m in METRICS} for k, v in g.items()}
    fits = {}
    for o in obs:
        xs = [n for n in ns]
        var = [float(np.mean([c["summary"]["var_f"] for c in g[(o, n)]])) for n in ns]
        fits[o] = {"n_qubits": xs, "var_f": var,
                   "exponential_fit": fit_exponential_concentration(xs, var)}
    return {"cells": cells, "aggregated": agg, "concentration_fits": fits,
            "n_qubits": ns, "observables": obs}


# --------------------------------------------------------------------------- #
# Ablations: encoding (C) and ansatz depth
# --------------------------------------------------------------------------- #
def ablation_encoding(P, jobs):
    encs = ["angle", "amplitude", "reuploading"]
    clfs = [_clf(s, encoding=e) for s in P["seeds"] for e in encs]
    warmup_models(clfs)
    tasks = [dict(clf_cfg=_clf(s, encoding=e), def_cfg=DefenseConfig("none"),
                  atk_cfg=_atk("calibrated_hsja", s, P), n_images=P["n_images"])
             for s in P["seeds"] for e in encs]
    cells = _run_all(jobs, tasks)
    return {"cells": cells,
            "aggregated": _summarize(_group(cells, lambda c: c["classifier"]["encoding"]))}


def ablation_depth(P, jobs):
    Ls = [2, 5, 10]
    clfs = [_clf(s, n_layers=L) for s in P["seeds"] for L in Ls]
    warmup_models(clfs)
    tasks = [dict(clf_cfg=_clf(s, n_layers=L), def_cfg=DefenseConfig("none"),
                  atk_cfg=_atk("calibrated_hsja", s, P), n_images=P["n_images"])
             for s in P["seeds"] for L in Ls]
    cells = _run_all(jobs, tasks)
    return {"cells": cells,
            "aggregated": _summarize(_group(cells, lambda c: c["classifier"]["n_layers"]))}


def ablation_dataset(P, jobs):
    ds = ["mnist_3v5", "mnist_0v1", "fashion_mnist"]
    clfs = [_clf(s, dataset=d) for s in P["seeds"] for d in ds]
    warmup_models(clfs)
    tasks = [dict(clf_cfg=_clf(s, dataset=d), def_cfg=DefenseConfig("none"),
                  atk_cfg=_atk("calibrated_hsja", s, P), n_images=P["n_images"])
             for s in P["seeds"] for d in ds]
    cells = _run_all(jobs, tasks)
    return {"cells": cells,
            "aggregated": _summarize(_group(cells, lambda c: c["classifier"]["dataset"]))}


RQS = {"rq1": rq1, "rq2": rq2, "rq3": rq3, "rq4": rq4, "rq5": rq5,
       "ablation_encoding": ablation_encoding, "ablation_depth": ablation_depth,
       "ablation_dataset": ablation_dataset}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Hard-label quantum attack experiments")
    ap.add_argument("--rq", required=True, choices=list(RQS) + ["all"])
    ap.add_argument("--preset", default="smoke", choices=list(PRESETS))
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--images", type=int, default=None, help="override images per cell")
    ap.add_argument("--seeds", type=int, default=None, help="override number of seeds")
    ap.add_argument("--out", default=RESULTS)
    ap.add_argument("--force", action="store_true",
                    help="recompute even if results/<rq>.json already exists")
    ap.add_argument("--no-checkpoint", action="store_true",
                    help="disable per-cell checkpointing/resume")
    args = ap.parse_args()

    P = dict(PRESETS[args.preset])
    if args.images:
        P["n_images"] = args.images
    if args.seeds:
        P["seeds"] = tuple(range(args.seeds))

    targets = list(RQS) if args.rq == "all" else [args.rq]
    os.makedirs(args.out, exist_ok=True)
    save_json({"preset": args.preset, "params": {**P, "seeds": list(P["seeds"])},
               "defaults": {"n_qubits": DEFAULT_N, "n_layers": DEFAULT_L,
                            "encoding": DEFAULT_ENC, "observable": DEFAULT_OBS,
                            "dataset": DEFAULT_DATASET},
               "jobs": args.jobs},
              os.path.join(args.out, "config.json"))

    global _CKPT_DIR
    for name in targets:
        path = os.path.join(args.out, f"{name}.json")
        if os.path.exists(path) and not args.force:
            print(f"\n=== {name}: results/{name}.json exists -> skip (use --force to redo) ===",
                  flush=True)
            continue
        _CKPT_DIR = None if args.no_checkpoint else os.path.join(args.out, "checkpoints", name)
        t0 = time.time()
        print(f"\n=== {name} (preset={args.preset}, images={P['n_images']}, "
              f"seeds={len(P['seeds'])}) ===", flush=True)
        res = RQS[name](P, args.jobs)
        res["_meta"] = {"rq": name, "preset": args.preset,
                        "params": {**P, "seeds": list(P["seeds"])},
                        "runtime_s": round(time.time() - t0, 1)}
        save_json(res, path)
        print(f"=== {name} done in {res['_meta']['runtime_s']}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
