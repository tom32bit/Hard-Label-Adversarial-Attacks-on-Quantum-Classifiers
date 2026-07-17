"""White-box PGD reference (plan Sec. 4.4, upper-bound attacker).

Differentiates the classifier through the circuit (parameter-shift / backprop), so it
is the strongest possible adversary and yields a *lower bound* on the achievable
perturbation against which the gradient-free hard-label attacks are measured.  We
report the minimal-norm adversarial: descend ``y0 * f`` until the label flips, then
bisect back toward ``x0`` using the exact (infinite-shot) sign to minimise ‖x'-x0‖.
"""
from __future__ import annotations

import numpy as np


def pgd_attack(vqc, x0, y0, domain, cfg) -> dict:
    import torch

    x0 = np.asarray(x0, dtype=np.float64)
    x0t = torch.tensor(x0, dtype=torch.float64)
    y0 = int(y0)
    x = x0t.clone()
    step = cfg.pgd_step_size
    grad_steps = 0
    x_adv = None

    # Phase 1: gradient descent on y0 * f until the label flips (find any adversarial).
    for _ in range(cfg.pgd_steps):
        x = x.detach().requires_grad_(True)
        f = vqc.decision_function_torch(x.unsqueeze(0))[0]
        (y0 * f).backward()
        grad_steps += 1
        g = x.grad.detach()
        g = g / (g.norm() + 1e-12)
        x = x.detach() - step * g
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
    dist = float(np.linalg.norm(x_b - x0))
    return {"success": True, "perturbation": dist, "x_adv": x_b,
            "queries": 0, "shots": 0, "grad_steps": grad_steps}
