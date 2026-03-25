"""
metrics.py — C-index et score Trustii.
"""

import ctypes
import numpy as np
from ._lib import lib


def c_index(risks: np.ndarray, times: np.ndarray, events: np.ndarray) -> float:
    """Concordance index (C-index) — appelle l'implémentation C."""
    risks  = np.ascontiguousarray(risks,  dtype=np.float32)
    times  = np.ascontiguousarray(times,  dtype=np.float32)
    events = np.ascontiguousarray(events, dtype=np.uint8)
    n = len(risks)
    return lib.c_index(
        risks.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        times.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        events.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_size_t(n),
    )


def trustii_score(ci_hepatic: float, ci_death: float) -> float:
    """Score final Trustii = 0.7 * C-index_hepatic + 0.3 * C-index_death."""
    return 0.7 * ci_hepatic + 0.3 * ci_death
