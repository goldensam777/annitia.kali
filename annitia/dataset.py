"""
dataset.py — Wrapper Python pour MasldDataset.
"""

import ctypes
import numpy as np
from ._lib import lib
from ._structs import MAX_TIMESTEPS, N_FEATURES


class MasldDataset:
    """Wraps MasldDataset* C."""

    def __init__(self, path: str):
        ptr = lib.masld_load(path.encode())
        if not ptr:
            raise IOError(f"masld_load: impossible d'ouvrir {path}")
        self._ptr = ptr
        self.n_patients = lib.masld_n_patients(ptr)
        self.T = MAX_TIMESTEPS
        self.F = N_FEATURES

    def __len__(self):
        return self.n_patients

    def get_batch(self, start: int, size: int) -> "_Batch":
        size = min(size, self.n_patients - start)
        batch_ptr = lib.masld_batch_alloc(size, self.T, self.F)
        lib.masld_get_batch(self._ptr, batch_ptr, start, size)
        return _Batch(batch_ptr, size, self.T, self.F, owned=True)

    def get_all(self) -> "_Batch":
        return self.get_batch(0, self.n_patients)

    def __del__(self):
        if hasattr(self, "_ptr") and self._ptr:
            lib.masld_free(self._ptr)
            self._ptr = None


class _Batch:
    """Wraps SurvivalBatch* C — fournit des vues numpy zero-copy."""

    def __init__(self, ptr, batch_size, T, F, owned=True):
        self._ptr = ptr
        self.batch_size = batch_size
        self.T = T
        self.F = F
        self._owned = owned

        # Vues numpy sur les buffers C (zero-copy)
        TF = T * F
        float_p = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_void_p))

        # On accède aux champs via la structure SurvivalBatch
        from ._structs import SurvivalBatch
        self._sb = SurvivalBatch.from_address(ptr)

        self.features      = self._arr(self._sb.features,      batch_size * TF, np.float32)
        self.mask          = self._arr(self._sb.mask,           batch_size * TF, np.float32)
        self.time_gaps     = self._arr(self._sb.time_gaps,      batch_size * T,  np.float32)
        self.n_visits      = self._arr(self._sb.n_visits,       batch_size,      np.int32)
        self.time_hepatic  = self._arr(self._sb.time_hepatic,   batch_size,      np.float32)
        self.event_hepatic = self._arr(self._sb.event_hepatic,  batch_size,      np.uint8)
        self.time_death    = self._arr(self._sb.time_death,     batch_size,      np.float32)
        self.event_death   = self._arr(self._sb.event_death,    batch_size,      np.uint8)

        # Reshape pratique
        self.features = self.features.reshape(batch_size, T, F)
        self.mask      = self.mask.reshape(batch_size, T, F)
        self.time_gaps = self.time_gaps.reshape(batch_size, T)

    @staticmethod
    def _arr(ptr, n, dtype):
        """Vue numpy sur un pointeur C."""
        ctype = {
            np.float32: ctypes.c_float,
            np.int32:   ctypes.c_int,
            np.uint8:   ctypes.c_uint8,
        }[dtype]
        return np.ctypeslib.as_array(
            ctypes.cast(ptr, ctypes.POINTER(ctype)), shape=(n,)
        ).view(dtype)

    def ptr(self):
        return self._ptr

    def __del__(self):
        if self._owned and hasattr(self, "_ptr") and self._ptr:
            lib.masld_batch_free(self._ptr)
            self._ptr = None
