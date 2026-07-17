"""Classical-surrogate transfer attack (plan Sec. 4.4, alternative black-box route).

Query the VQC to label a pool, fit a small classical MLP surrogate, run white-box
PGD on the surrogate, and transfer the perturbation to the VQC.  Contextualises the
query cost of the decision-based attacks: transfer is cheap in queries but its
perturbations are typically larger / less reliable because the surrogate only
approximates the quantum boundary.
"""
from __future__ import annotations

import numpy as np


def _train_surrogate(X, y01, seed=0, epochs=150):
    import torch

    torch.manual_seed(seed)
    d = X.shape[1]
    net = torch.nn.Sequential(
        torch.nn.Linear(d, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 1),
    ).double()
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y01, dtype=torch.float64).reshape(-1, 1)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    for _ in range(epochs):
        opt.zero_grad()
        logit = net(Xt)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt)
        loss.backward()
        opt.step()
    return net


def transfer_attack(vqc, oracle, x0, y0, X_pool, domain, cfg, rng) -> dict:
    import torch

    x0 = np.asarray(x0, dtype=np.float64)
    y0 = int(y0)

    # 1. label the pool with hard-label VQC queries (charged)
    y_pool = np.array([oracle.label(xp, cfg.fixed_shots) for xp in X_pool])
    pool_queries = oracle.n_queries
    net = _train_surrogate(X_pool, (y_pool + 1) // 2, seed=cfg.seed)

    # 2. white-box PGD on the surrogate to flip x0's surrogate label
    x = torch.tensor(x0, dtype=torch.float64)
    target_orig = torch.tensor([float((y0 + 1) // 2)], dtype=torch.float64)
    for _ in range(cfg.pgd_steps):
        x = x.detach().requires_grad_(True)
        logit = net(x.unsqueeze(0))[0, 0]
        # ascend the loss wrt the ORIGINAL label -> push x away from y0 (flip it)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit.unsqueeze(0),
                                                                     target_orig)
        loss.backward()
        g = x.grad.detach()
        x = x.detach() + cfg.pgd_step_size * g / (g.norm() + 1e-12)
        x = torch.tensor(domain.project(x.numpy()), dtype=torch.float64)

    x_pert = x.numpy()
    # 3. transfer + exact-sign bisection toward x0 for the minimal transferable crossing
    f = float(vqc.decision_function(x_pert.reshape(1, -1))[0])
    oracle.n_queries += 1
    if (1 if f >= 0 else -1) == y0:                           # transfer failed to flip
        return {"success": False, "perturbation": float("inf"), "x_adv": x0.copy(),
                "queries": oracle.n_queries, "shots": oracle.n_shots,
                "pool_queries": int(pool_queries)}
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        xm = domain.project(x0 + mid * (x_pert - x0))
        fm = float(vqc.decision_function(xm.reshape(1, -1))[0])
        oracle.n_queries += 1
        if (1 if fm >= 0 else -1) != y0:
            hi = mid
        else:
            lo = mid
    x_b = domain.project(x0 + hi * (x_pert - x0))
    return {"success": True, "perturbation": float(np.linalg.norm(x_b - x0)),
            "x_adv": x_b, "queries": oracle.n_queries, "shots": oracle.n_shots,
            "pool_queries": int(pool_queries)}
