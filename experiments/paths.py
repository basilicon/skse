"""Default locations for data, embeddings, and results.

Every path can be overridden with an environment variable, e.g.
    SKSE_DATA_DIR=/mnt/datasets python experiments/pipeline.py
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_path(var, default):
    return Path(os.environ.get(var, default))


DATA_DIR = _env_path("SKSE_DATA_DIR", REPO_ROOT / "data")
PROCESSED_DIR = DATA_DIR / "processed"
EMBED_DIR = _env_path("SKSE_EMBED_DIR", REPO_ROOT / "embed")
RESULTS_DIR = _env_path("SKSE_RESULTS_DIR", REPO_ROOT / "results")
FIGURES_DIR = _env_path("SKSE_FIGURES_DIR", RESULTS_DIR / "figures")

# Datasets used in the paper
DATASETS = ["arcene", "GDS4278", "GDS4516"]
