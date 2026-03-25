"""
_lib.py — Chargement de libannitia.so via ctypes.

Cherche la lib dans :
  1. ANNITIA_LIB_PATH (variable d'env)
  2. build/libannitia.so  (développement local)
  3. LD_LIBRARY_PATH standard
"""

import ctypes
import ctypes.util
import os
import pathlib

def _find_lib() -> str:
    # 1. Variable d'environnement explicite
    env = os.environ.get("ANNITIA_LIB_PATH")
    if env and os.path.exists(env):
        return env

    # 2. Chemin relatif depuis ce fichier (package installé ou dev)
    here = pathlib.Path(__file__).parent
    candidates = [
        here.parent / "build" / "libannitia.so",
        here.parent / "libannitia.so",
        here / "libannitia.so",
    ]
    for p in candidates:
        if p.exists():
            return str(p)

    # 3. ctypes.util.find_library
    found = ctypes.util.find_library("annitia")
    if found:
        return found

    raise FileNotFoundError(
        "libannitia.so introuvable. "
        "Compilez avec `cmake --build build` ou définissez ANNITIA_LIB_PATH."
    )


lib = ctypes.CDLL(_find_lib())

# ---------------------------------------------------------------------------
# Types C de base
# ---------------------------------------------------------------------------
c_float_p  = ctypes.POINTER(ctypes.c_float)
c_uint8_p  = ctypes.POINTER(ctypes.c_uint8)
c_int_p    = ctypes.POINTER(ctypes.c_int)
c_size_t_p = ctypes.POINTER(ctypes.c_size_t)
c_void_p   = ctypes.c_void_p

# ---------------------------------------------------------------------------
# Signatures des fonctions exportées
# ---------------------------------------------------------------------------

# --- AnnitiaModel ---
lib.annitia_create.restype  = c_void_p
lib.annitia_create.argtypes = [ctypes.c_void_p]   # const AnnitiaConfig*

lib.annitia_free.restype  = None
lib.annitia_free.argtypes = [c_void_p]

lib.annitia_init.restype  = None
lib.annitia_init.argtypes = [c_void_p, ctypes.c_ulong]

lib.annitia_enable_training.restype  = None
lib.annitia_enable_training.argtypes = [
    c_void_p,           # AnnitiaModel*
    ctypes.c_void_p,    # const MBOptimConfig*
    ctypes.c_float,     # lr
    ctypes.c_float,     # weight_decay
]

lib.annitia_forward.restype  = None
lib.annitia_forward.argtypes = [
    c_void_p,    # AnnitiaModel*
    c_void_p,    # const SurvivalBatch*
    c_float_p,   # float* risk_hepatic
    c_float_p,   # float* risk_death
]

lib.annitia_train_step.restype  = ctypes.c_float
lib.annitia_train_step.argtypes = [c_void_p, c_void_p]  # model, batch

lib.annitia_save.restype  = ctypes.c_int
lib.annitia_save.argtypes = [c_void_p, ctypes.c_char_p]

lib.annitia_load.restype  = c_void_p
lib.annitia_load.argtypes = [
    ctypes.c_char_p,    # path
    ctypes.c_int,       # for_training
    ctypes.c_void_p,    # const MBOptimConfig* (NULL ok)
    ctypes.c_float,     # lr
    ctypes.c_float,     # weight_decay
]

# --- MasldDataset ---
lib.masld_load.restype  = c_void_p
lib.masld_load.argtypes = [ctypes.c_char_p]

lib.masld_free.restype  = None
lib.masld_free.argtypes = [c_void_p]

lib.masld_n_patients.restype  = ctypes.c_size_t
lib.masld_n_patients.argtypes = [c_void_p]

lib.masld_batch_alloc.restype  = c_void_p
lib.masld_batch_alloc.argtypes = [
    ctypes.c_size_t,  # batch_size
    ctypes.c_size_t,  # seq_len
    ctypes.c_size_t,  # n_features
]

lib.masld_batch_free.restype  = None
lib.masld_batch_free.argtypes = [c_void_p]

lib.masld_get_batch.restype  = None
lib.masld_get_batch.argtypes = [c_void_p, c_void_p, ctypes.c_size_t, ctypes.c_size_t]

lib.masld_get_batch_idx.restype  = None
lib.masld_get_batch_idx.argtypes = [c_void_p, c_void_p, c_size_t_p, ctypes.c_size_t]

# --- Métriques ---
lib.c_index.restype  = ctypes.c_float
lib.c_index.argtypes = [c_float_p, c_float_p, c_uint8_p, ctypes.c_size_t]
