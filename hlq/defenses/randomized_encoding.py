"""Randomized-encoding defense: fresh random rotations per query [17].

A data-independent random rotation layer is inserted after the encoding and
resampled on every query.  For the legitimate user the effect averages out; for the
attacker each query's decision value ``f`` is perturbed by an *independent* random
amount, so the boundary the attacker probes jitters from call to call -- an
attacker-side barren plateau on top of the shot noise.  ``randomized_strength``
tunes the plateau intensity (and trades against clean accuracy).

This is pure-state (fast, lightning), so it scales to all ``n``.  The per-query
randomness makes the oracle mark the classifier ``stochastic`` (no caching of ``f``).
"""
from __future__ import annotations

import numpy as np
import pennylane as qml

from ..classifier import _observable, apply_ansatz, apply_encoding
from ..config import DefenseConfig


class RandomizedEncodingVQC:
    def __init__(self, base_vqc, defense: DefenseConfig):
        self.cfg = base_vqc.cfg
        self.weights = base_vqc.weights
        self.strength = float(defense.randomized_strength)
        self.per_query = bool(defense.randomized_per_query)
        self._rng = np.random.default_rng(self.cfg.seed + 4242)
        self._fixed = None                    # used when per_query is False
        self._qnode = None

    def _build(self):
        if self._qnode is not None:
            return self._qnode
        cfg = self.cfg
        try:
            dev = qml.device("lightning.qubit", wires=cfg.n_qubits)
        except Exception:
            dev = qml.device("default.qubit", wires=cfg.n_qubits)

        def circ(x, w, rand):
            wires = range(cfg.n_qubits)
            if cfg.encoding == "reuploading":
                for r in range(cfg.reupload_layers):
                    qml.AngleEmbedding(x, wires=wires, rotation="Y")
                    qml.StronglyEntanglingLayers(w[r : r + 1], wires=wires)
            else:
                apply_encoding(x, cfg)
            for i in wires:                                  # random per-query layer
                qml.RY(rand[..., i], wires=i)
            if cfg.encoding != "reuploading":
                apply_ansatz(w, cfg)
            return qml.expval(_observable(cfg))

        self._qnode = qml.QNode(circ, dev)
        return self._qnode

    def _sample_rand(self, batch):
        if self.per_query:
            return self._rng.normal(0.0, self.strength, size=(batch, self.cfg.n_qubits))
        if self._fixed is None:
            self._fixed = self._rng.normal(0.0, self.strength, size=(self.cfg.n_qubits,))
        return np.broadcast_to(self._fixed, (batch, self.cfg.n_qubits))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        qn = self._build()
        rand = self._sample_rand(len(X))
        out = np.asarray(qn(X, self.weights, rand), dtype=np.float64).reshape(-1)
        return np.clip(out, -1.0, 1.0)

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1).astype(np.int64)

    def clean_accuracy(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def margins(self, X):
        return np.abs(self.decision_function(X))
