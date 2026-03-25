"""
model.py — Wrapper Python pour AnnitiaModel.
"""

import ctypes
import numpy as np
from ._lib import lib
from ._structs import AnnitiaConfig, MBOptimConfig, MAX_TIMESTEPS, N_FEATURES


class AnnitiaModel:
    """
    Wrapper haut niveau pour AnnitiaModel* C.

    Exemple :
        model = AnnitiaModel(dim=64, state=16, layers=2)
        model.init(seed=42)
        model.enable_training(lr=3e-4, wd=1e-4)
        loss = model.train_step(batch)
        rh, rd = model.forward(batch)
        model.save("checkpoint.bin")
    """

    def __init__(
        self,
        dim:       int   = 64,
        state:     int   = 16,
        layers:    int   = 2,
        mimo_rank: int   = 1,
        n_features: int  = N_FEATURES,
        seq_len:   int   = MAX_TIMESTEPS,
        dt_scale:  float = 1.0,
        dt_min:    float = 0.001,
        dt_max:    float = 0.1,
    ):
        cfg = AnnitiaConfig(
            n_features = n_features,
            dim        = dim,
            state_size = state,
            seq_len    = seq_len,
            n_layers   = layers,
            mimo_rank  = mimo_rank,
            dt_scale   = dt_scale,
            dt_min     = dt_min,
            dt_max     = dt_max,
        )
        ptr = lib.annitia_create(ctypes.byref(cfg))
        if not ptr:
            raise RuntimeError("annitia_create a échoué")
        self._ptr = ptr
        self.dim = dim
        self._training = False

    # ------------------------------------------------------------------
    def init(self, seed: int = 42):
        lib.annitia_init(self._ptr, ctypes.c_ulong(seed))

    # ------------------------------------------------------------------
    def enable_training(
        self,
        lr:           float = 1e-3,
        wd:           float = 1e-4,
        mu:           float = 0.9,
        beta2:        float = 0.999,
        eps:          float = 1e-8,
        clip_norm:    float = 1.0,
    ):
        opt = MBOptimConfig(
            lr           = lr,
            mu           = mu,
            beta2        = beta2,
            eps          = eps,
            clip_norm    = clip_norm,
            weight_decay = wd,
        )
        lib.annitia_enable_training(
            self._ptr, ctypes.byref(opt),
            ctypes.c_float(lr), ctypes.c_float(wd)
        )
        self._training = True

    # ------------------------------------------------------------------
    def forward(self, batch) -> tuple[np.ndarray, np.ndarray]:
        """
        Retourne (risk_hepatic, risk_death) — arrays numpy float32 [B].
        batch : _Batch ou pointeur brut C.
        """
        N = batch.batch_size if hasattr(batch, "batch_size") else 1
        rh = np.zeros(N, dtype=np.float32)
        rd = np.zeros(N, dtype=np.float32)
        lib.annitia_forward(
            self._ptr,
            batch.ptr() if hasattr(batch, "ptr") else batch,
            rh.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            rd.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        return rh, rd

    # ------------------------------------------------------------------
    def train_step(self, batch) -> float:
        """Un pas de gradient. Retourne la loss."""
        return lib.annitia_train_step(
            self._ptr,
            batch.ptr() if hasattr(batch, "ptr") else batch,
        )

    # ------------------------------------------------------------------
    def save(self, path: str) -> int:
        return lib.annitia_save(self._ptr, path.encode())

    # ------------------------------------------------------------------
    @classmethod
    def load(
        cls,
        path:         str,
        for_training: bool  = False,
        lr:           float = 1e-3,
        wd:           float = 1e-4,
    ) -> "AnnitiaModel":
        opt = MBOptimConfig(lr=lr, mu=0.9, beta2=0.999, eps=1e-8,
                             clip_norm=1.0, weight_decay=wd)
        ptr = lib.annitia_load(
            path.encode(),
            ctypes.c_int(1 if for_training else 0),
            ctypes.byref(opt) if for_training else None,
            ctypes.c_float(lr),
            ctypes.c_float(wd),
        )
        if not ptr:
            raise IOError(f"annitia_load: impossible de charger {path}")
        obj = cls.__new__(cls)
        obj._ptr = ptr
        obj._training = for_training
        return obj

    # ------------------------------------------------------------------
    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            lib.annitia_free(self._ptr)
            self._ptr = None
