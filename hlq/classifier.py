"""Variational quantum classifier under attack (plan Sec. 4.1).

    y_hat(x) = sign(f_theta(x)),
    f_theta(x) = <0| E(x)^dag W(theta)^dag M W(theta) E(x) |0>,  f in [-1, 1].

One ``_circuit`` definition is shared across three QNodes so the *same* unitary is
used everywhere (single source of truth):

* inference QNode  (lightning.qubit, batched)  -> exact ``f`` for the oracle;
* training QNode   (default.qubit, torch backprop) -> gradient-based fitting;
* the training QNode is reused for white-box input gradients (PGD reference).

VERIFY invariants (asserted): |f| <= 1 for every input; loss decreases over the
first epochs; clean decision values are non-degenerate (checked by callers/T-tests).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pennylane as qml

from .config import ClassifierConfig

_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models_cache")


# --------------------------------------------------------------------------- #
# Circuit building blocks (shared by all QNodes)
# --------------------------------------------------------------------------- #
def _observable(cfg: ClassifierConfig):
    """Hermitian measurement M with eigenvalues in {-1,+1} (diagonal in Z basis)."""
    if cfg.observable == "local_z":
        return qml.PauliZ(0)
    op = qml.PauliZ(0)                     # global parity Z^{otimes n} (concentration source)
    for i in range(1, cfg.n_qubits):
        op = op @ qml.PauliZ(i)
    return op


def weight_shape(cfg: ClassifierConfig):
    if cfg.encoding == "reuploading":
        return (cfg.reupload_layers, cfg.n_qubits, 3)
    return qml.StronglyEntanglingLayers.shape(n_layers=cfg.n_layers, n_wires=cfg.n_qubits)


def apply_encoding(x, cfg: ClassifierConfig):
    """Data-encoding unitary E(x) (single application; broadcasts over batch)."""
    wires = range(cfg.n_qubits)
    if cfg.encoding == "amplitude":
        qml.AmplitudeEmbedding(x, wires=wires, normalize=True, pad_with=0.0)
    else:                                         # angle / reuploading base
        qml.AngleEmbedding(x, wires=wires, rotation="Y")


def apply_ansatz(weights, cfg: ClassifierConfig):
    qml.StronglyEntanglingLayers(weights, wires=range(cfg.n_qubits))


def circuit_body(x, weights, cfg: ClassifierConfig):
    """E(x) then W(theta), with re-uploading interleaving encode/ansatz per layer."""
    wires = range(cfg.n_qubits)
    if cfg.encoding == "reuploading":
        for r in range(cfg.reupload_layers):      # data re-uploading (Perez-Salinas)
            qml.AngleEmbedding(x, wires=wires, rotation="Y")
            qml.StronglyEntanglingLayers(weights[r : r + 1], wires=wires)
    else:
        apply_encoding(x, cfg)
        apply_ansatz(weights, cfg)


def _circuit(x, weights, cfg: ClassifierConfig):
    """E(x) then W(theta); returns <M>. Broadcasts over a leading batch dim of x."""
    circuit_body(x, weights, cfg)
    return qml.expval(_observable(cfg))


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
class VQC:
    """Trainable VQC with a fast batched exact-inference path for the oracle."""

    def __init__(self, cfg: ClassifierConfig):
        self.cfg = cfg
        self.weights: Optional[np.ndarray] = None
        self._infer_qnode = None
        self._torch_qnode = None

    # -- QNode construction ------------------------------------------------- #
    def _build_infer(self):
        if self._infer_qnode is None:
            try:
                dev = qml.device("lightning.qubit", wires=self.cfg.n_qubits)
            except Exception:
                dev = qml.device("default.qubit", wires=self.cfg.n_qubits)
            cfg = self.cfg
            self._infer_qnode = qml.QNode(lambda x, w: _circuit(x, w, cfg), dev)
        return self._infer_qnode

    def _build_torch(self):
        if self._torch_qnode is None:
            dev = qml.device("default.qubit", wires=self.cfg.n_qubits)
            cfg = self.cfg
            self._torch_qnode = qml.QNode(
                lambda x, w: _circuit(x, w, cfg), dev,
                interface="torch", diff_method="backprop",
            )
        return self._torch_qnode

    # -- Inference ---------------------------------------------------------- #
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Exact infinite-shot decision value f_theta(x), batched. No gradients."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        qn = self._build_infer()
        out = np.asarray(qn(X, self.weights), dtype=np.float64).reshape(-1)
        # VERIFY: observable value must lie in [-1, 1] (numerical guard).
        assert np.all(np.abs(out) <= 1.0 + 1e-6), f"|f|>1: max={np.abs(out).max()}"
        return np.clip(out, -1.0, 1.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        f = self.decision_function(X)
        return np.where(f >= 0, 1, -1).astype(np.int64)

    def clean_accuracy(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def margins(self, X) -> np.ndarray:
        """|f_theta(x)| -- used for the concentration diagnostic (RQ5)."""
        return np.abs(self.decision_function(X))

    # -- Differentiable path (white-box PGD reference) ---------------------- #
    def decision_function_torch(self, X_torch):
        import torch

        qn = self._build_torch()
        w = torch.as_tensor(self.weights, dtype=torch.float64)
        out = qn(X_torch, w)
        return out.reshape(-1)

    # -- Training ----------------------------------------------------------- #
    def fit(self, X, y, X_test=None, y_test=None, *, verbose=False, use_cache=True) -> dict:
        """Train theta by Adam on binary cross-entropy of p=(1+f)/2.

        Always returns the same info dict and handles its own weight cache, so a cache
        hit and a fresh fit are indistinguishable to callers. Models are keyed by the
        full config, so one trained VQC is reused across every attack, defense and RQ.
        """
        import torch

        os.makedirs(_MODELS, exist_ok=True)
        cache = os.path.join(_MODELS, self.cfg.key() + ".npz")
        if use_cache and os.path.exists(cache):
            d = np.load(cache)
            self.weights = d["weights"]
            return {"train_acc": float(d["train_acc"]), "test_acc": float(d["test_acc"]),
                    "loss_curve": d["loss_curve"].tolist(), "cached": True}

        torch.manual_seed(self.cfg.seed)
        shape = weight_shape(self.cfg)
        w = torch.nn.Parameter(0.1 * torch.randn(*shape, dtype=torch.float64))
        qn = self._build_torch()
        Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64)
        yt = torch.as_tensor((np.asarray(y) + 1) // 2, dtype=torch.float64)   # {0,1}
        opt = torch.optim.Adam([w], lr=self.cfg.lr)

        n = len(yt)
        bs = self.cfg.batch_size
        loss_curve = []
        for epoch in range(self.cfg.epochs):
            perm = torch.randperm(n)
            ep_loss = 0.0
            for i in range(0, n, bs):
                idx = perm[i : i + bs]
                opt.zero_grad()
                f = qn(Xt[idx], w).reshape(-1)
                p = torch.clamp((1.0 + f) / 2.0, 1e-6, 1 - 1e-6)
                loss = torch.nn.functional.binary_cross_entropy(p, yt[idx])
                loss.backward()
                opt.step()
                ep_loss += float(loss) * len(idx)
            loss_curve.append(ep_loss / n)
            if verbose and (epoch % 5 == 0 or epoch == self.cfg.epochs - 1):
                self.weights = w.detach().numpy()
                print(f"  epoch {epoch:3d} loss={loss_curve[-1]:.4f} "
                      f"acc={self.clean_accuracy(X, y):.3f}")

        self.weights = w.detach().numpy()
        # VERIFY: the loss must fall over the first epochs (guards a degenerate fit)
        if len(loss_curve) >= 10:
            assert loss_curve[9] <= loss_curve[0] + 1e-6, "loss did not decrease early"

        train_acc = self.clean_accuracy(X, y)
        test_acc = (self.clean_accuracy(X_test, y_test)
                    if X_test is not None else float("nan"))
        np.savez(cache, weights=self.weights, train_acc=train_acc, test_acc=test_acc,
                 loss_curve=np.asarray(loss_curve))
        return {"train_acc": float(train_acc), "test_acc": float(test_acc),
                "loss_curve": loss_curve, "cached": False}


def train_or_load(cfg: ClassifierConfig, bundle, verbose=False):
    """Train (or load cached) a VQC on a DataBundle. Returns (vqc, info)."""
    vqc = VQC(cfg)
    info = vqc.fit(bundle.X_train, bundle.y_train, bundle.X_test, bundle.y_test,
                   verbose=verbose)
    return vqc, info
