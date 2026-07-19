"""Rebuild an RQ2 result file from per-cell checkpoints of an interrupted run.

When a session hits a wall-clock limit mid-RQ, the completed cells still exist in
``results/checkpoints/rq2/``. This reconstructs the same JSON structure the driver
would have written from whatever finished, so a truncated run is not wasted, and
records explicitly which sweeps are complete vs partial.

Run:  python experiments/salvage_rq2.py --checkpoints <dir> --out results/rq2_partial.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hlq.analysis import budget_curve, find_interior_optimum, save_json
from hlq.metrics import seed_aggregate

METRICS = ("success_rate", "median_perturbation", "median_queries", "median_shots",
           "robust_accuracy_at_eps", "clean_accuracy")


def _agg(cells, metric):
    return seed_aggregate([c["summary"].get(metric, float("nan")) for c in cells])


def _group(cells, keyfn):
    out = {}
    for c in cells:
        out.setdefault(keyfn(c), []).append(c)
    return out


def _summarize(groups):
    return {k: {m: _agg(v, m) for m in METRICS} for k, v in groups.items()}


def salvage(ckpt_dir: str, expected_seeds: int = 8) -> dict:
    cells = [json.load(open(f)) for f in glob.glob(os.path.join(ckpt_dir, "*.json"))]
    alloc = [c for c in cells if c.get("force_probe_shots") is not None]
    rest = [c for c in cells if c.get("force_probe_shots") is None]
    out = {"_salvaged": True, "n_cells_recovered": len(cells)}

    # (a) budget curve: default grad fraction, total_budget varying
    bud = [c for c in rest if c["attack"]["grad_budget_frac"] == 0.5]
    if bud:
        g = _group(bud, lambda c: c["attack"]["total_budget"])
        agg = _summarize(g)
        xs = sorted(g)
        out["budget_curve"] = {
            "budgets": xs, "aggregated": agg,
            "curve": budget_curve([{"total_budget": int(k),
                                    "median_perturbation": v["median_perturbation"]["mean"]}
                                   for k, v in agg.items()]),
            "seeds_per_point": {str(k): len(v) for k, v in g.items()},
            "complete": all(len(v) >= expected_seeds for v in g.values()),
            "cells": bud,
        }

    # (b) allocation sweep over forced probe shots
    if alloc:
        g = _group(alloc, lambda c: c["force_probe_shots"])
        agg = _summarize(g)
        xs = sorted(g)
        ys = [agg[x]["median_perturbation"]["mean"] for x in xs]
        out["allocation_sweep"] = {
            "probe_shots": xs, "median_perturbation": ys, "aggregated": agg,
            "interior_optimum": find_interior_optimum(xs, ys),
            "seeds_per_point": {str(k): len(v) for k, v in g.items()},
            "complete": all(len(v) >= expected_seeds for v in g.values()),
            "cells": alloc,
        }

    # (c) M1/M2 split: total budget fixed, grad fraction varying
    split = [c for c in rest if c["attack"]["grad_budget_frac"] != 0.5]
    if split:
        base = max(c["attack"]["total_budget"] for c in split)
        sc = [c for c in rest if c["attack"]["total_budget"] == base]
        g = _group(sc, lambda c: c["attack"]["grad_budget_frac"])
        agg = _summarize(g)
        xs = sorted(g)
        ys = [agg[x]["median_perturbation"]["mean"] for x in xs]
        out["m1_m2_split"] = {
            "grad_budget_frac": xs, "median_perturbation": ys, "aggregated": agg,
            "interior_optimum": find_interior_optimum(xs, ys),
            "seeds_per_point": {str(k): len(v) for k, v in g.items()},
            "complete": all(len(v) >= expected_seeds for v in g.values()),
            "cells": sc,
        }

    out["missing"] = [k for k in ("budget_curve", "allocation_sweep", "m1_m2_split")
                      if k not in out]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected-seeds", type=int, default=8)
    args = ap.parse_args()

    res = salvage(args.checkpoints, args.expected_seeds)
    res["_meta"] = {"rq": "rq2", "salvaged_from": args.checkpoints,
                    "expected_seeds": args.expected_seeds}
    save_json(res, args.out)

    print(f"[salvage] recovered {res['n_cells_recovered']} cells -> {args.out}")
    for k in ("budget_curve", "allocation_sweep", "m1_m2_split"):
        if k in res:
            sp = res[k]["seeds_per_point"]
            print(f"  {k}: {'COMPLETE' if res[k]['complete'] else 'PARTIAL'} "
                  f"(seeds/point: min={min(sp.values())}, max={max(sp.values())})")
    if res["missing"]:
        print(f"  missing entirely: {res['missing']}")


if __name__ == "__main__":
    main()
