"""
_structs.py — Structures C mappées en ctypes.

Doit rester en sync avec include/annitia.h.
"""

import ctypes

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MAX_TIMESTEPS = 22
N_FEATURES    = 18
DYN_FEATURES  = 12
STAT_FEATURES = 6


# ---------------------------------------------------------------------------
# AnnitiaConfig
# ---------------------------------------------------------------------------
class AnnitiaConfig(ctypes.Structure):
    _fields_ = [
        ("n_features", ctypes.c_size_t),
        ("dim",        ctypes.c_size_t),
        ("state_size", ctypes.c_size_t),
        ("seq_len",    ctypes.c_size_t),
        ("n_layers",   ctypes.c_size_t),
        ("mimo_rank",  ctypes.c_size_t),
        ("dt_scale",   ctypes.c_float),
        ("dt_min",     ctypes.c_float),
        ("dt_max",     ctypes.c_float),
        ("use_conv2d", ctypes.c_int),
        ("conv2d_K",   ctypes.c_size_t),
    ]


# ---------------------------------------------------------------------------
# MBOptimConfig  (depuis kmamba.h)
# ---------------------------------------------------------------------------
class MBOptimConfig(ctypes.Structure):
    _fields_ = [
        ("lr",           ctypes.c_float),
        ("mu",           ctypes.c_float),
        ("beta2",        ctypes.c_float),
        ("eps",          ctypes.c_float),
        ("clip_norm",    ctypes.c_float),
        ("weight_decay", ctypes.c_float),
    ]


# ---------------------------------------------------------------------------
# SurvivalBatch  — layout doit correspondre à annitia.h
# ---------------------------------------------------------------------------
class SurvivalBatch(ctypes.Structure):
    _fields_ = [
        ("features",       ctypes.POINTER(ctypes.c_float)),
        ("mask",           ctypes.POINTER(ctypes.c_float)),
        ("time_gaps",      ctypes.POINTER(ctypes.c_float)),
        ("n_visits",       ctypes.POINTER(ctypes.c_int)),
        ("time_hepatic",   ctypes.POINTER(ctypes.c_float)),
        ("event_hepatic",  ctypes.POINTER(ctypes.c_uint8)),
        ("time_death",     ctypes.POINTER(ctypes.c_float)),
        ("event_death",    ctypes.POINTER(ctypes.c_uint8)),
        ("batch_size",     ctypes.c_size_t),
        ("seq_len",        ctypes.c_size_t),
        ("n_features",     ctypes.c_size_t),
    ]
