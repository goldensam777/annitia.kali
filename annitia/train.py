"""
train.py — Boucle d'entraînement Python orchestrant libannitia.so.
"""

import numpy as np
from typing import Optional, Callable
from .model   import AnnitiaModel
from .dataset import MasldDataset
from .metrics import c_index, trustii_score


def train(
    model:       AnnitiaModel,
    ds_train:    MasldDataset,
    ds_val:      Optional[MasldDataset] = None,
    epochs:      int   = 50,
    batch_size:  int   = 32,
    seed:        int   = 42,
    checkpoint:  Optional[str] = None,
    verbose:     bool  = True,
    on_epoch:    Optional[Callable] = None,
) -> list[dict]:
    """
    Boucle d'entraînement complète.

    Retourne l'historique : liste de dicts par époque
    {'epoch', 'loss', 'ci_hep', 'ci_dth', 'score'}.

    on_epoch(epoch, metrics) — callback optionnel (pour plots live, early stopping…).
    """
    rng = np.random.default_rng(seed)
    N   = len(ds_train)
    T   = ds_train.T
    F   = ds_train.F

    # Batch de validation (chargé une seule fois)
    val_batch = ds_val.get_all() if ds_val and len(ds_val) > 0 else None

    best_score = -1.0
    history    = []

    if verbose:
        if val_batch:
            print(f"Train: {N} | Val: {len(ds_val)}")
            print("\nEpoch | Loss       | C-idx Hep | C-idx Dth | Score")
            print("------|------------|-----------|-----------|------")
        else:
            print(f"Train: {N} (pas de validation)")
            print("\nEpoch | Loss")
            print("------|----------")

    for ep in range(1, epochs + 1):
        # Shuffle
        perm = rng.permutation(N)

        total_loss = 0.0
        n_batches  = 0

        for start in range(0, N - batch_size + 1, batch_size):
            idx   = perm[start : start + batch_size].astype(np.uintp)
            batch = _get_batch_idx(ds_train, idx)
            total_loss += model.train_step(batch)
            n_batches  += 1

        avg_loss = total_loss / n_batches if n_batches else 0.0

        metrics = {"epoch": ep, "loss": avg_loss,
                   "ci_hep": None, "ci_dth": None, "score": None}

        if val_batch:
            rh, rd = model.forward(val_batch)
            ci_hep = c_index(rh, val_batch.time_hepatic, val_batch.event_hepatic)
            ci_dth = c_index(rd, val_batch.time_death,   val_batch.event_death)
            score  = trustii_score(ci_hep, ci_dth)
            metrics.update(ci_hep=ci_hep, ci_dth=ci_dth, score=score)

            flag = ""
            if score > best_score:
                best_score = score
                if checkpoint:
                    model.save(checkpoint)
                flag = " *"

            if verbose:
                print(f"{ep:5d} | {avg_loss:10.4f} | {ci_hep:9.4f} | "
                      f"{ci_dth:9.4f} | {score:.4f}{flag}")
        else:
            if checkpoint:
                model.save(checkpoint)
            if verbose:
                print(f"{ep:5d} | {avg_loss:10.4f}")

        history.append(metrics)
        if on_epoch:
            on_epoch(ep, metrics)

    if verbose and val_batch:
        print(f"\nBest: {best_score:.4f}" +
              (f" — {checkpoint}" if checkpoint else ""))

    return history


def _get_batch_idx(ds: MasldDataset, idx: np.ndarray):
    """Charge un batch dans un ordre arbitraire (shuffle)."""
    import ctypes
    from ._lib import lib

    size = len(idx)
    idx_c = idx.astype(np.uintp)
    batch_ptr = lib.masld_batch_alloc(size, ds.T, ds.F)
    lib.masld_get_batch_idx(
        ds._ptr,
        batch_ptr,
        idx_c.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
        ctypes.c_size_t(size),
    )
    from .dataset import _Batch
    return _Batch(batch_ptr, size, ds.T, ds.F, owned=True)
