"""Real datasets (full binary-pair subsets) and the synthetic known-boundary control.

No mock/synthetic stand-ins for real data: MNIST and Fashion-MNIST are downloaded
in full via torchvision and used at their *complete* binary-pair scope.  The only
synthetic dataset is a deliberate control with an *analytically known* decision
boundary, required by sanity test T3 (plan Sec. 5, Sec. 7).

Feature maps
------------
* angle / reuploading : PCA to ``n_qubits`` components (fit on train), then
  min-max scaled to ``[0, pi]`` so features are valid rotation angles.
* amplitude           : each image resized to ``2**n_qubits`` pixels (2-D area
  interpolation, structure preserved), flattened and L2-normalised -> a valid
  quantum state amplitude vector.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import ClassifierConfig

_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_cache")

# dataset key -> (torchvision class, (class_neg, class_pos))
_SPEC = {
    "mnist_3v5": ("MNIST", (3, 5)),
    "mnist_0v1": ("MNIST", (0, 1)),
    "fashion_mnist": ("FashionMNIST", (0, 6)),   # T-shirt vs Shirt: a genuinely hard pair
}


@dataclass
class DataBundle:
    """Everything an experiment needs from a dataset, with fitted transforms."""

    X_train: np.ndarray          # (N_tr, n_features) encoder inputs
    y_train: np.ndarray          # (N_tr,) in {-1, +1}
    X_test: np.ndarray           # (N_te, n_features)
    y_test: np.ndarray           # (N_te,) in {-1, +1}
    encoding: str
    n_features: int
    meta: dict                   # provenance + fitted-transform parameters


# --------------------------------------------------------------------------- #
# Raw image loading (full data)
# --------------------------------------------------------------------------- #
def _load_raw_images(dataset: str):
    """Return (Xtr_img, ytr, Xte_img, yte) with images in [0,1], FULL real dataset.

    Tries several providers so the same code runs locally and on Kaggle (whose free
    tier may have Internet disabled): torchvision -> keras -> a mounted /kaggle/input
    copy.  Real data only -- there is no synthetic fallback for MNIST/Fashion-MNIST.
    """
    tv_name, _ = _SPEC[dataset]
    os.makedirs(_CACHE, exist_ok=True)
    errors = []

    try:                                   # 1. torchvision (cached to data_cache/)
        import torchvision

        cls = getattr(torchvision.datasets, tv_name)
        tr = cls(_CACHE, train=True, download=True)
        te = cls(_CACHE, train=False, download=True)
        return (tr.data.numpy().astype(np.float64) / 255.0, np.asarray(tr.targets),
                te.data.numpy().astype(np.float64) / 255.0, np.asarray(te.targets))
    except Exception as e:                 # pragma: no cover - provider fallback
        errors.append(f"torchvision: {e!r}")

    try:                                   # 2. keras bundled loaders
        from tensorflow import keras

        loader = (keras.datasets.mnist if tv_name == "MNIST"
                  else keras.datasets.fashion_mnist)
        (Xtr, ytr), (Xte, yte) = loader.load_data()
        return (Xtr.astype(np.float64) / 255.0, ytr,
                Xte.astype(np.float64) / 255.0, yte)
    except Exception as e:                 # pragma: no cover
        errors.append(f"keras: {e!r}")

    # 3. a dataset mounted into a Kaggle notebook (Add Data -> MNIST / Fashion-MNIST)
    for base in ("/kaggle/input", os.path.join(_CACHE, "kaggle")):
        hit = _try_kaggle_idx(base, tv_name)
        if hit is not None:
            return hit
    raise RuntimeError(
        f"Could not load real {dataset}. Enable Internet in the notebook settings, or "
        f"attach the dataset. Attempts: " + " | ".join(errors))


def _try_kaggle_idx(base: str, tv_name: str):
    """Look for raw idx-ubyte files in a mounted directory tree."""
    if not os.path.isdir(base):
        return None
    want = {"train_x": "train-images", "train_y": "train-labels",
            "test_x": "t10k-images", "test_y": "t10k-labels"}
    found = {}
    for root, _dirs, files in os.walk(base):
        for f in files:
            for k, pat in want.items():
                if pat in f and "idx" in f and k not in found:
                    found[k] = os.path.join(root, f)
    if len(found) < 4:
        return None
    Xtr = _read_idx(found["train_x"]).astype(np.float64) / 255.0
    Xte = _read_idx(found["test_x"]).astype(np.float64) / 255.0
    return Xtr, _read_idx(found["train_y"]), Xte, _read_idx(found["test_y"])


def _read_idx(path: str) -> np.ndarray:
    """Minimal IDX (MNIST binary format) reader, transparently gunzipping."""
    import gzip
    import struct

    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rb") as fh:
        magic, ndim = struct.unpack(">HBB", fh.read(4))[1:]
        shape = struct.unpack(">" + "I" * ndim, fh.read(4 * ndim))
        return np.frombuffer(fh.read(), dtype=np.uint8).reshape(shape)


def _binary_filter(X, y, classes):
    neg, pos = classes
    mask = (y == neg) | (y == pos)
    Xb = X[mask]
    yb = np.where(y[mask] == pos, 1, -1).astype(np.int64)
    return Xb, yb


# --------------------------------------------------------------------------- #
# Feature maps
# --------------------------------------------------------------------------- #
def _resize_to_pow2(imgs: np.ndarray, n_qubits: int) -> np.ndarray:
    """Area-resize (N,H,W) images to 2**n pixels, preserving 2-D structure."""
    import torch
    import torch.nn.functional as F

    h = 2 ** ((n_qubits + 1) // 2)
    w = 2 ** (n_qubits // 2)
    t = torch.from_numpy(imgs).unsqueeze(1).float()          # (N,1,H,W)
    t = F.interpolate(t, size=(h, w), mode="area")
    return t.squeeze(1).reshape(imgs.shape[0], h * w).numpy().astype(np.float64)


def _make_features(Xtr_img, Xte_img, cfg: ClassifierConfig):
    """Build (Xtr, Xte, meta) for the requested encoding. Transforms fit on train."""
    n = cfg.n_qubits
    if cfg.encoding == "amplitude":
        Ftr = _resize_to_pow2(Xtr_img, n)
        Fte = _resize_to_pow2(Xte_img, n)
        ntr = np.linalg.norm(Ftr, axis=1, keepdims=True)
        nte = np.linalg.norm(Fte, axis=1, keepdims=True)
        ntr[ntr == 0] = 1.0
        nte[nte == 0] = 1.0
        Ftr, Fte = Ftr / ntr, Fte / nte
        meta = {"map": "resize_l2norm", "dim": 2 ** n}
        return Ftr, Fte, meta

    # angle / reuploading: PCA -> min-max to [0, pi]
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler

    flat_tr = Xtr_img.reshape(Xtr_img.shape[0], -1)
    flat_te = Xte_img.reshape(Xte_img.shape[0], -1)
    pca = PCA(n_components=n, random_state=0).fit(flat_tr)
    Ztr, Zte = pca.transform(flat_tr), pca.transform(flat_te)
    scaler = MinMaxScaler((0.0, np.pi)).fit(Ztr)
    Ftr = scaler.transform(Ztr)
    Fte = np.clip(scaler.transform(Zte), 0.0, np.pi)         # keep test angles valid
    meta = {
        "map": "pca_minmax",
        "pca_explained_var": float(pca.explained_variance_ratio_.sum()),
        "n_components": n,
    }
    return Ftr, Fte, meta


def _feature_cache_path(cfg: ClassifierConfig) -> str:
    os.makedirs(_CACHE, exist_ok=True)
    return os.path.join(
        _CACHE,
        f"feat_{cfg.dataset}_n{cfg.n_qubits}_{cfg.encoding}"
        f"_tr{cfg.train_size}_te{cfg.test_size}_s{cfg.seed}.npz",
    )


def load_dataset(cfg: ClassifierConfig, use_cache: bool = True) -> DataBundle:
    """Load a real binary-pair dataset (full scope) as encoder-ready features.

    Features are cached to disk: the PCA/resize transform is deterministic given the
    config, so parallel workers reuse it instead of recomputing it per process.
    """
    if cfg.dataset == "synthetic":
        return _load_synthetic(cfg)

    cache = _feature_cache_path(cfg)
    if use_cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return DataBundle(d["X_train"], d["y_train"], d["X_test"], d["y_test"],
                          cfg.encoding, cfg.n_features, json.loads(str(d["meta"])))

    Xtr_img, ytr_all, Xte_img, yte_all = _load_raw_images(cfg.dataset)
    Xtr_img, ytr = _binary_filter(Xtr_img, ytr_all, _SPEC[cfg.dataset][1])
    Xte_img, yte = _binary_filter(Xte_img, yte_all, _SPEC[cfg.dataset][1])

    # Optional caps (train_size/test_size == 0 => full data)
    rng = np.random.default_rng(cfg.seed)
    if cfg.train_size and cfg.train_size < len(ytr):
        idx = rng.permutation(len(ytr))[: cfg.train_size]
        Xtr_img, ytr = Xtr_img[idx], ytr[idx]
    if cfg.test_size and cfg.test_size < len(yte):
        idx = rng.permutation(len(yte))[: cfg.test_size]
        Xte_img, yte = Xte_img[idx], yte[idx]

    Xtr, Xte, meta = _make_features(Xtr_img, Xte_img, cfg)
    meta.update(dataset=cfg.dataset, classes=list(_SPEC[cfg.dataset][1]),
                n_train=int(len(ytr)), n_test=int(len(yte)))
    if use_cache:
        np.savez_compressed(cache, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
                            meta=json.dumps(meta))
    return DataBundle(Xtr, ytr, Xte, yte, cfg.encoding, cfg.n_features, meta)


# --------------------------------------------------------------------------- #
# Synthetic control with analytically known boundary (T3)
# --------------------------------------------------------------------------- #
def _load_synthetic(cfg: ClassifierConfig) -> DataBundle:
    """Linearly separable data in feature space with a known margin boundary.

    Feature space matches the angle encoder ([0, pi]); label = sign(w.x + b).
    The exact point-to-boundary distance is |w.x + b| / ||w||, used by T3.
    """
    n = cfg.n_qubits
    rng = np.random.default_rng(cfg.seed)
    w, b = make_linear_boundary(n, seed=cfg.seed)
    n_tr, n_te = 4000, 2000

    def _sample(m):
        X = rng.uniform(0.0, np.pi, size=(m, n))
        s = X @ w + b
        keep = np.abs(s) > 0.15                      # small margin band removed
        X, s = X[keep], s[keep]
        y = np.where(s > 0, 1, -1).astype(np.int64)
        return X, y

    Xtr, ytr = _sample(int(n_tr * 1.4))
    Xte, yte = _sample(int(n_te * 1.4))
    Xtr, ytr = Xtr[:n_tr], ytr[:n_tr]
    Xte, yte = Xte[:n_te], yte[:n_te]
    meta = {"map": "synthetic_linear", "w": w.tolist(), "b": float(b),
            "dataset": "synthetic", "n_train": len(ytr), "n_test": len(yte)}
    return DataBundle(Xtr, ytr, Xte, yte, "angle", n, meta)


def make_linear_boundary(n: int, seed: int = 0):
    """A fixed random hyperplane through the centre of the [0, pi]^n box."""
    rng = np.random.default_rng(seed + 777)
    w = rng.normal(size=n)
    w /= np.linalg.norm(w)
    b = -float(w @ (np.full(n, np.pi / 2)))          # boundary passes box centre
    return w, b
