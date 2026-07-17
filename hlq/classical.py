"""Parameter-matched classical NN -- the classical anchor (plan Sec. 4.4).

HSJA on this model calibrates whether the quantum attack's query cost is "normal":
the same boundary-walk skeleton runs against a *deterministic* oracle (S=None), so
any extra cost the VQC imposes is attributable to its stochastic Born-rule oracle
rather than to the attack.  Hidden width is chosen so the parameter count matches the
VQC's ansatz (``L * n * 3`` for StronglyEntanglingLayers).
"""
from __future__ import annotations

import numpy as np

from .config import ClassifierConfig


def matched_hidden_width(cfg: ClassifierConfig) -> int:
    """Hidden width h giving ~ the VQC's parameter count for input dim d."""
    d = cfg.n_features
    target = (cfg.reupload_layers if cfg.encoding == "reuploading" else cfg.n_layers) \
        * cfg.n_qubits * 3
    h = int(round(target / max(d + 2, 1)))
    return max(2, h)


class MatchedClassicalNN:
    """Tanh-output MLP exposing the same interface as :class:`hlq.classifier.VQC`."""

    def __init__(self, cfg: ClassifierConfig):
        self.cfg = cfg
        self.hidden = matched_hidden_width(cfg)
        self.net = None

    def _build(self):
        import torch

        torch.manual_seed(self.cfg.seed)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.cfg.n_features, self.hidden), torch.nn.Tanh(),
            torch.nn.Linear(self.hidden, 1), torch.nn.Tanh(),      # output in [-1,1]
        ).double()
        return self.net

    def n_parameters(self) -> int:
        if self.net is None:
            self._build()
        return int(sum(p.numel() for p in self.net.parameters()))

    def fit(self, X, y, epochs=200, lr=0.01):
        import torch

        net = self._build()
        Xt = torch.tensor(np.asarray(X), dtype=torch.float64)
        yt = torch.tensor((np.asarray(y) + 1) / 2.0, dtype=torch.float64).reshape(-1, 1)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            f = net(Xt)
            p = torch.clamp((1.0 + f) / 2.0, 1e-6, 1 - 1e-6)
            loss = torch.nn.functional.binary_cross_entropy(p, yt)
            loss.backward()
            opt.step()
        return {"train_acc": self.clean_accuracy(X, y), "n_parameters": self.n_parameters(),
                "hidden": self.hidden}

    def decision_function(self, X):
        import torch

        if self.net is None:
            self._build()
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        with torch.no_grad():
            out = self.net(torch.tensor(X, dtype=torch.float64)).numpy().reshape(-1)
        return np.clip(out, -1.0, 1.0)

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1).astype(np.int64)

    def clean_accuracy(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def margins(self, X):
        return np.abs(self.decision_function(X))
