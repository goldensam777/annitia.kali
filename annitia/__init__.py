"""
annitia — Python wrapper for libannitia.so (K-Mamba SSM for MASLD survival).

Usage rapide :
    from annitia import AnnitiaModel, MasldDataset
    from annitia.train import train
    from annitia.metrics import c_index, trustii_score
"""

from .model    import AnnitiaModel
from .dataset  import MasldDataset
from .metrics  import c_index, trustii_score
from .ensemble import ensemble_predictions
from ._structs import (
    MAX_TIMESTEPS, N_FEATURES, DYN_FEATURES, STAT_FEATURES,
    AnnitiaConfig, MBOptimConfig,
)

__all__ = [
    "AnnitiaModel",
    "MasldDataset",
    "c_index",
    "trustii_score",
    "ensemble_predictions",
    "MAX_TIMESTEPS",
    "N_FEATURES",
    "DYN_FEATURES",
    "STAT_FEATURES",
    "AnnitiaConfig",
    "MBOptimConfig",
]
