"""Quantum-noise defense: depolarizing channels injected at inference [16].

Faithful density-matrix model (``default.mixed``): a depolarizing channel of
strength ``p`` follows every ansatz layer (and the encoding).  Noise acts *during*
computation, so ``f_noisy(x)`` is not a global rescaling of ``f_ideal`` -- it
reshapes the boundary and shrinks margins non-uniformly, which is exactly the effect
whose gradient-free robustness we test (RQ4) and whose margin-collapse must be told
apart from genuine defense (RQ5).

Cost note: density-matrix simulation is ``O(4**n)``; the defense experiments cap ``n``.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml

from ..classifier import _observable, apply_encoding
from ..config import DefenseConfig


class NoisyVQC:
    def __init__(self, base_vqc, defense: DefenseConfig):
        self.cfg = base_vqc.cfg
        self.weights = base_vqc.weights
        self.p = float(defense.depolarizing_p)
        self._qnode = None
        self._batched = None            # broadcasting support auto-detected on first call

    def _build(self):
        if self._qnode is not None:
            return self._qnode
        cfg, p = self.cfg, self.p
        dev = qml.device("default.mixed", wires=cfg.n_qubits)

        def circ(x, w):
            wires = range(cfg.n_qubits)
            if cfg.encoding == "reuploading":
                for r in range(cfg.reupload_layers):
                    qml.AngleEmbedding(x, wires=wires, rotation="Y")
                    qml.StronglyEntanglingLayers(w[r : r + 1], wires=wires)
                    for i in wires:
                        qml.DepolarizingChannel(p, wires=i)
            else:
                apply_encoding(x, cfg)
                for l in range(w.shape[0]):                 # per-layer noise injection
                    qml.StronglyEntanglingLayers(w[l : l + 1], wires=wires)
                    for i in wires:
                        qml.DepolarizingChannel(p, wires=i)
            return qml.expval(_observable(cfg))

        self._qnode = qml.QNode(circ, dev)
        return self._qnode

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        qn = self._build()
        if self._batched is None:                           # detect broadcasting once
            try:
                out = np.asarray(qn(X, self.weights), dtype=np.float64).reshape(-1)
                self._batched = len(out) == len(X)
                if self._batched:
                    return np.clip(out, -1.0, 1.0)
            except Exception:
                self._batched = False
        if self._batched:
            out = np.asarray(qn(X, self.weights), dtype=np.float64).reshape(-1)
        else:
            out = np.array([float(qn(x, self.weights)) for x in X])
        return np.clip(out, -1.0, 1.0)

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1).astype(np.int64)

    def clean_accuracy(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def margins(self, X):
        return np.abs(self.decision_function(X))
