"""Momentum-based quantum adversarial attack -- the QMI precedent [6].

The direct comparison the plan requires: a white-box, momentum-boosted iterative attack
(MI-FGSM, Dong et al. 2018) differentiated *through the circuit* (backprop /
parameter-shift). Like the PGD reference it is a white-box upper bound -- it computes a
gradient of the classifier, which the realistic hard-label threat model forbids -- but it
is the state-of-the-art gradient attacker for VQCs and so anchors where the calibrated
hard-label attack sits relative to the strongest published quantum adversary.

Momentum accumulates an L1-normalised gradient velocity
``g <- mu * g + grad / ||grad||_1``, which stabilises the direction and escapes poor
local optima -- the paper's advantage over plain (P)GD. We report the *minimal-norm*
adversarial (descend ``y0 * f`` under momentum until the label flips, then bisect back
toward ``x0`` with the exact sign), identically to :func:`hlq.attacks.pgd_whitebox.pgd_attack`
so the two white-box references are measured on the same footing.
"""
from __future__ import annotations

import numpy as np


def momentum_attack(vqc, x0, y0, domain, cfg) -> dict:
    import torch

    x0 = np.asarray(x0, dtype=np.float64)
    x0t = torch.tensor(x0, dtype=torch.float64)
    y0 = int(y0)
    x = x0t.clone()
    g = torch.zeros_like(x0t)                       # momentum (velocity) accumulator
    mu = float(getattr(cfg, "momentum_mu", 1.0))
    step = cfg.pgd_step_size
    grad_steps = 0
    x_adv = None

    # Phase 1: momentum gradient descent on y0 * f until the label flips.
    for _ in range(cfg.pgd_steps):
        x = x.detach().requires_grad_(True)
        f = vqc.decision_function_torch(x.unsqueeze(0))[0]
        (y0 * f).backward()
        grad_steps += 1
        gr = x.grad.detach()
        g = mu * g + gr / (gr.abs().sum() + 1e-12)   # MI-FGSM: L1-normalised accumulation
        x = x.detach() - step * g / (g.norm() + 1e-12)
        x = torch.tensor(domain.project(x.numpy()), dtype=torch.float64)
        f_now = float(vqc.decision_function(x.numpy().reshape(1, -1))[0])
        if (1 if f_now >= 0 else -1) != y0:
            x_adv = x.numpy().copy()
            break

    if x_adv is None:
        return {"success": False, "perturbation": float("inf"), "x_adv": x0.copy(),
                "queries": 0, "shots": 0, "grad_steps": grad_steps}

    # Phase 2: exact-sign bisection toward x0 for the minimal-norm crossing.
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        xm = domain.project(x0 + mid * (x_adv - x0))
        f = float(vqc.decision_function(xm.reshape(1, -1))[0])
        if (1 if f >= 0 else -1) != y0:
            hi = mid
        else:
            lo = mid
    x_b = domain.project(x0 + hi * (x_adv - x0))
    return {"success": True, "perturbation": float(np.linalg.norm(x_b - x0)),
            "x_adv": x_b, "queries": 0, "shots": 0, "grad_steps": grad_steps}
