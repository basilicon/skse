"""Stochastic k-Simplex Embedding (SKSE)."""
from .loss import metric_skse_loss, sample_simplices, simplex_volume, universal_simplex_volume
from .model import SKSE

__version__ = "0.1.0"

__all__ = [
    "SKSE",
    "metric_skse_loss",
    "sample_simplices",
    "simplex_volume",
    "universal_simplex_volume",
]
