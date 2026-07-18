"""Typed configuration objects and experiment presets.

Every experiment cell is fully described by a :class:`ClassifierConfig` (the model
under attack), an :class:`AttackConfig` (the adversary), and a
:class:`DefenseConfig` (optional inference-time defense). These are JSON-round-
trippable so ``config.json`` can be dumped at the start of every run (plan Sec. 8).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Optional


# --------------------------------------------------------------------------- #
# Vocabulary (kept as plain strings so configs serialize cleanly to JSON)
# --------------------------------------------------------------------------- #
ENCODINGS = ("angle", "amplitude", "reuploading")
OBSERVABLES = ("local_z", "global_z")          # Z_0  vs  Z^{otimes n}
DATASETS = ("mnist_3v5", "mnist_0v1", "fashion_mnist", "synthetic")
DEFENSES = ("none", "depolarizing", "randomized_encoding")
ATTACKS = (
    "calibrated_hsja",   # ours (M1 + M2)
    "fixed_hsja",        # naive port
    "popskipjump",       # constant-flip baseline
    "pgd_whitebox",      # upper-bound reference
    "momentum",          # momentum-based quantum attack (QMI precedent [6])
    "transfer",          # classical-surrogate transfer
    "classical_hsja",    # HSJA on a matched classical NN (anchor)
)


@dataclass
class ClassifierConfig:
    """Specification of the variational quantum classifier under attack."""

    n_qubits: int = 8
    n_layers: int = 5
    encoding: str = "angle"
    observable: str = "local_z"
    dataset: str = "mnist_3v5"
    reupload_layers: int = 3          # only used by the re-uploading encoding
    epochs: int = 40
    # Large batches are near-free: PennyLane broadcasts the batch through one
    # statevector call, so wall-clock tracks epochs, not steps (~6 s/epoch on the full
    # 11.5k-sample MNIST 3-vs-5 pair at n=8, vs ~10x slower at batch 32).
    batch_size: int = 256
    lr: float = 0.05
    train_size: int = 0               # 0 => use the full available binary-pair train set
    test_size: int = 0                # 0 => use the full available test set
    seed: int = 0

    def __post_init__(self) -> None:
        assert self.encoding in ENCODINGS, self.encoding
        assert self.observable in OBSERVABLES, self.observable
        assert self.dataset in DATASETS, self.dataset

    @property
    def n_features(self) -> int:
        """Input dimensionality fed to the encoder."""
        if self.encoding == "amplitude":
            return 2 ** self.n_qubits
        return self.n_qubits

    def key(self) -> str:
        """Stable identifier used to cache trained models / results."""
        return (
            f"{self.dataset}_n{self.n_qubits}_L{self.n_layers}"
            f"_{self.encoding}_{self.observable}_s{self.seed}"
        )

    to_dict = asdict


@dataclass
class DefenseConfig:
    name: str = "none"
    depolarizing_p: float = 0.05          # per-qubit depolarizing strength
    randomized_strength: float = 0.30     # std of the random-encoder rotations (rad)
    randomized_per_query: bool = True     # resample the random encoder each query

    def __post_init__(self) -> None:
        assert self.name in DEFENSES, self.name

    to_dict = asdict


@dataclass
class AttackConfig:
    """Decision-based attack hyper-parameters (HSJA family)."""

    name: str = "calibrated_hsja"
    iterations: int = 30                  # boundary-walk steps
    total_budget: int = 200_000           # T = total measurement shots for the attack
    delta_decision: float = 0.05          # target per-decision error for M1 (calibrated)
    fixed_shots: int = 100                # S for fixed-shot variants / probe default
    grad_budget_frac: float = 0.5         # fraction of per-iter budget for normal est.
    num_probes: int = 100                 # B (fixed variants / upper cap)
    probe_shots: int = 100                # S_probe (fixed variants)
    init_eval_shots: int = 200            # shots for the (rare) initialisation queries
    bin_search_tol: float = 1e-4          # theta-tolerance of geometric binary search
    norm: str = "l2"                      # perturbation norm ("l2" only for now)
    # PopSkipJump: the CONSTANT flip rate it assumes (oracle-agnostic -- it does not
    # know the Born-rule margin structure, so it applies one repeat count everywhere).
    psj_assumed_p0: float = 0.30
    # PGD reference
    pgd_steps: int = 100
    pgd_step_size: float = 0.02
    pgd_epsilon: float = 0.8
    momentum_mu: float = 1.0              # decay factor for the momentum attack [6]
    seed: int = 0

    def __post_init__(self) -> None:
        assert self.name in ATTACKS, self.name

    to_dict = asdict


@dataclass
class ExperimentConfig:
    """A full experiment cell: model x defense x attack, plus evaluation scope."""

    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    n_attack_images: int = 250            # attacked test images per cell (heavier stats)
    seeds: tuple = (0, 1, 2, 3, 4, 5, 6, 7)   # >=5; 8 for heavier statistics

    def to_dict(self) -> dict:
        return {
            "classifier": self.classifier.to_dict(),
            "defense": self.defense.to_dict(),
            "attack": self.attack.to_dict(),
            "n_attack_images": self.n_attack_images,
            "seeds": list(self.seeds),
        }


# --------------------------------------------------------------------------- #
# Presets: fast smoke-scale vs. full "heavier statistics" scope
# --------------------------------------------------------------------------- #
SMOKE = dict(n_attack_images=6, seeds=(0, 1), attack_iterations=8, total_budget=20_000)
FULL = dict(n_attack_images=250, seeds=(0, 1, 2, 3, 4, 5, 6, 7), attack_iterations=30,
            total_budget=200_000)


def dataclass_from_dict(cls, d: dict):
    """Rebuild a dataclass from a plain dict (JSON round-trip helper)."""
    fields = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in fields})
