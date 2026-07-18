"""Generate the self-contained Kaggle/Colab notebook from the package sources.

The notebook is *generated*, never hand-maintained, so it cannot drift from the
modules: every ``hlq`` file is embedded verbatim via ``%%writefile``, then the
notebook runs the same sanity gate, driver and figure code as the repo.

Run:  python experiments/make_notebook.py
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "hard_label_quantum_attacks.ipynb")

MODULES = [
    "hlq/__init__.py", "hlq/seeds.py", "hlq/config.py", "hlq/data.py",
    "hlq/classifier.py", "hlq/classical.py", "hlq/oracle.py", "hlq/budget.py",
    "hlq/metrics.py", "hlq/concentration.py", "hlq/analysis.py",
    "hlq/attacks/__init__.py", "hlq/attacks/base.py", "hlq/attacks/hsja_fixed.py",
    "hlq/attacks/hsja_quantum.py", "hlq/attacks/popskipjump.py",
    "hlq/attacks/pgd_whitebox.py", "hlq/attacks/momentum.py", "hlq/attacks/transfer.py",
    "hlq/defenses/__init__.py", "hlq/defenses/noise.py",
    "hlq/defenses/randomized_encoding.py", "hlq/runner.py",
    "experiments/run_sanity.py", "experiments/driver.py", "experiments/make_figures.py",
    "experiments/significance.py",
]


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.splitlines(True)}


def build():
    cells = [md("""# Hard-Label Adversarial Attacks on Quantum Classifiers

**A Shot-Budget-Aware, Born-Rule-Calibrated Decision-Based Attack for Variational Quantum Classifiers**

Self-contained companion notebook. It writes the full `hlq` package to disk, runs the
sanity gate (T1-T6), executes the experiments for RQ1-RQ5 plus the ablations, and
renders the figures. Everything is reproducible from the seeds in the configs.

**The idea.** Every published adversarial attack on quantum classifiers is white-box and
gradient-based. A deployed quantum classifier returns a *measured class label* - and that
label is a Bernoulli sample whose flip probability is set by the Born-rule margin
`|<M>|` and the shot budget `S`. So the oracle is natively stochastic, and decision-based
attacks are known to break on stochastic oracles. This notebook builds a
**Born-rule-calibrated, shot-budget-aware** hard-label attack and studies its query/shot
economics.

**Runtime notes**
* No GPU needed. These circuits (<=12 qubits) are tiny; the cost is the *number* of
  circuit evaluations, which is CPU- and Python-bound. A Kaggle **CPU** session is a fine
  target - a GPU will not speed this up.
* **Enable Internet** (Notebook settings -> Internet on) so the real MNIST /
  Fashion-MNIST data can download. Alternatively attach the MNIST dataset via *Add Data*;
  the loader will find it.
* Start with `PRESET = "smoke"` to verify the pipeline end-to-end (minutes), then move to
  `"medium"` or `"full"`. `"full"` is the heavier-statistics scope (250 attacked images
  per cell, 8 seeds) and is a multi-hour job - run it in stages with the `RQS` list below.
"""),
             md("## 1. Environment"),
             code("""# PennyLane is the only dependency Kaggle usually lacks.
import importlib, subprocess, sys
for pkg, mod in [("pennylane", "pennylane"), ("pennylane-lightning", "pennylane_lightning")]:
    if importlib.util.find_spec(mod) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

import pennylane as qml, torch, numpy, scipy, sklearn, matplotlib
print("pennylane   ", qml.__version__)
print("torch       ", torch.__version__)
print("numpy/scipy ", numpy.__version__, scipy.__version__)
print("devices     ", [d for d in qml.plugin_devices] if hasattr(qml, "plugin_devices") else "-")
"""),
             code("""import os
os.makedirs("hlq/attacks", exist_ok=True)
os.makedirs("hlq/defenses", exist_ok=True)
os.makedirs("experiments", exist_ok=True)
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
print("workspace ready")
"""),
             md("## 2. The `hlq` package\n\nEvery module is written verbatim from the "
                "repository sources, so the notebook and the repo cannot diverge.")]

    for rel in MODULES:
        path = os.path.join(ROOT, rel)
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        cells.append(code(f"%%writefile {rel}\n{src}"))

    cells += [
        md("## 3. Sanity gate (T1-T6)\n\nDefined *before* the code, per the plan. "
           "**T1-T4 must pass before any attack number is trusted.**\n\n"
           "* **T1** Born-rule flip model vs Monte-Carlo, and vs PennyLane's own shot sampling\n"
           "* **T2** infinite-shot limit -> deterministic HopSkipJump (paired per-image)\n"
           "* **T3** attack vs an analytically known boundary\n"
           "* **T4** already-adversarial input returned unchanged\n"
           "* **T5** budget monotonicity\n"
           "* **T6** concentration guardrail fires on a concentrated model, not a trained one"),
        code("""!python experiments/run_sanity.py --quick        # drop --quick for the full gate"""),
        code("""import json
s = json.load(open("results/sanity.json"))
for k, v in s["_summary"]["passed"].items():
    print(f"{k}: {'PASS' if v else 'FAIL'}")
print("\\nT1-T4 trust gate:", "PASS" if s["_summary"]["trust_gate_T1_T4"] else "FAIL")
"""),
        md("## 4. Experiments\n\nEach research question varies exactly one axis with "
           "principled defaults for the rest.\n\n"
           "| RQ | Question |\n|---|---|\n"
           "| RQ1 | Can a hard-label attack fool a VQC, and at what query cost? |\n"
           "| RQ2 | How should a fixed measurement budget `T = Q x S` be allocated? |\n"
           "| RQ3 | Does Born-rule calibration beat a constant-noise treatment (PopSkipJump)? |\n"
           "| RQ4 | Do the quantum-noise / randomized-encoding defenses survive a *gradient-free* attacker? |\n"
           "| RQ5 | Is apparent robustness really exponential concentration? |\n\n"
           "`smoke` proves the pipeline; `medium` gives real trends; `full` is the "
           "heavier-statistics scope from the plan (250 images/cell, 8 seeds)."),
        code('''# PUBLICATION-GRADE STATISTICS (8 seeds). Because the head-to-head blocks and the
# training-heavy RQ5 don't both fit one 12h CPU session, run this in TWO sessions:
#
#   RUN 1 (head-to-head, ~8-9h):  RQS = RUN1   (rq1/rq3/rq4/rq2 at 8 seeds)
#   RUN 2 (concentration, ~4h):   RQS = RUN2   (rq5, self-capped to 5 seeds)
#
# Each run is self-contained and downloads its own results zip (last cell). Merge both
# results/ folders locally and regenerate figures. Every cell is CHECKPOINTED, so a
# timeout inside a session loses nothing and re-running resumes.
PRESET = "kaggle8"         # 8 seeds. Use "kaggle" (3 seeds) for a faster first pass.
JOBS   = 4                 # parallel cells; set to the session's core count
RUN1   = ["rq1", "rq3", "rq4", "rq2"]     # rq2 last: its allocation sweep is the slowest
RUN2   = ["rq5"]
RQS    = RUN1              # <- switch to RUN2 for the second session

# RQ4 across qubit count (tests Theorem 4's exponential scaling): os.environ["HLQ_RQ4_NS"]="4,6"
# RQ5 n=12 tier (tens of min/model training):                     os.environ["HLQ_RQ5_MAX_N"]="12"
ALL_RQS = ["rq1", "rq2", "rq3", "rq4", "rq5", "ablation_encoding",
           "ablation_depth", "ablation_dataset"]

import subprocess, sys
for rq in RQS:
    print(f"\\n########## {rq} ##########", flush=True)
    subprocess.run([sys.executable, "experiments/driver.py", "--rq", rq,
                    "--preset", PRESET, "--jobs", str(JOBS)], check=False)
'''),
        md("## 5. Figures\n\nResearch figures rendered from the results JSON: a validated "
           "colorblind-safe categorical palette in fixed order, always with a second "
           "encoding (marker + linestyle); single-hue sequential ramps for magnitude; "
           "no dual axes anywhere."),
        code("""!python experiments/make_figures.py"""),
        code("""from IPython.display import Image, display
import glob, os
for p in sorted(glob.glob("figures/*.png")):
    print("\\n" + os.path.basename(p))
    display(Image(p))
"""),
        md("## 5b. Significance tests\n\nBootstrap CIs + Mann-Whitney / two-proportion "
           "tests on the per-image data, so each head-to-head claim carries a p-value and "
           "an effect size (plan Sec. 6). Runs only for whichever RQs are present."),
        code("""!python experiments/significance.py"""),
        md("## 6. Results summary\n\nThe headline numbers, straight out of the JSON."),
        code('''import json, os, numpy as np

def show(rq):
    p = f"results/{rq}.json"
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    agg = d.get("aggregated")
    if not agg:
        return
    print(f"\\n=== {rq} ===")
    for cond, mets in agg.items():
        sr = mets.get("success_rate", {})
        mp = mets.get("median_perturbation", {})
        print(f"  {str(cond):42s} success={sr.get('mean', float('nan')):.3f}"
              f" +-{sr.get('std', float('nan')):.3f}   median_pert="
              f"{mp.get('mean', float('nan')):.4f} +-{mp.get('std', float('nan')):.4f}")

for rq in ["rq1", "rq3", "rq4", "rq5", "ablation_encoding", "ablation_depth",
           "ablation_dataset"]:
    show(rq)

if os.path.exists("results/rq2.json"):
    d = json.load(open("results/rq2.json"))
    print("\\n=== rq2: allocation (H2 interior optimum?) ===")
    print(" ", d["allocation_sweep"]["interior_optimum"])
    print("=== rq2: M1/M2 split ===")
    print(" ", d["m1_m2_split"]["interior_optimum"])
'''),
        md("## 7. Collect outputs\n\nAll results are JSON; figures are PNG + vector PDF."),
        code("""import shutil, os
shutil.make_archive("hlq_results", "zip", ".", "results")
shutil.make_archive("hlq_figures", "zip", ".", "figures")
print("wrote hlq_results.zip and hlq_figures.zip")
print("results:", sorted(os.listdir("results")))
print("figures:", sorted(os.listdir("figures")))
"""),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1)
    print(f"wrote {OUT}  ({len(cells)} cells, {len(MODULES)} modules embedded)")


if __name__ == "__main__":
    build()
