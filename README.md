# SKSE: Stochastic k-Simplex Embedding

SKSE is a dimensionality reduction method that preserves the **volumes of k-simplices** (triangles, tetrahedra, ...) instead of pairwise distances. An MLP is trained so that randomly sampled simplices have the same volume in the embedding as in the input space. Sampling keeps training tractable on large datasets.

This repository contains the `skse` Python package and the code used for the experiments in:

> E. D. Curtis, P. Billion-Polak, and T. M. Khoshgoftaar, *Stochastic k-Simplex Embedding in High-Dimensional Cancer Classification*, ICTAI.

## Install

```bash
pip install -e .
```

## Usage

```python
import numpy as np
from skse import SKSE

X = np.random.randn(500, 1000).astype(np.float32)

reducer = SKSE(n_components=8, n_points=3, epochs=200, random_state=42)
Y = reducer.fit_transform(X)        # (500, 8)
Y_new = reducer.transform(X[:10])   # embed new points with the trained MLP
```

Main parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `n_components` | 2 | Embedding dimension |
| `n_points` | 3 | Points per sampled simplex |
| `epochs` | 1000 | Training epochs |
| `lr` | 1e-4 | AdamW learning rate |
| `batch_size` | 32 | Mini-batch size |
| `n_simplices` | 16 | Simplices sampled per mini-batch |
| `normalize` | `"batch"` | Volume normalization: `"batch"`, `"quality"`, or `None` |
| `random_state` | `None` | Seed for reproducibility |
| `device` | auto | `"cpu"`, `"cuda"`, ... |

The loss and sampler are also exported (`metric_skse_loss`, `sample_simplices`, `simplex_volume`) if you want to use SKSE as a term in your own model.

## Reproducing the paper

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the versions the experiments were tested with. `pip install -e ".[experiments]"` installs the same packages without pins. For GPU training, install the CUDA build of PyTorch first (see the note in `requirements.txt`).

1. Download the raw datasets into `data/`:
   - **ARCENE** (UCI): put the files in `data/arcene/ARCENE/` (`arcene_train.data`, `arcene_train.labels`, `arcene_valid.data`, `arcene_valid.labels`).
   - **GDS4278** and **GDS4516** (NCBI GEO): put the SOFT files at `data/GDS4278_full.soft/GDS4278_full.soft` and `data/GDS4516_full.soft/GDS4516_full.soft`.
2. Run, from the repository root:

```bash
python experiments/preprocess.py                 # -> data/processed/<dataset>/{X,y}.csv
python experiments/pipeline.py                   # embeddings, classifiers, reports
python experiments/eval_unreduced_baseline.py    # classifiers on raw features
python experiments/generate_polished_figures.py  # paper figures -> results/figures/
python experiments/generate_qualitative_2d_plots.py
```

`pipeline.py` caches embeddings in `embed/` and skips any that already exist. Use `--datasets arcene` to run a subset.

Paths can be changed with environment variables: `SKSE_DATA_DIR`, `SKSE_EMBED_DIR`, `SKSE_RESULTS_DIR`, `SKSE_FIGURES_DIR`, `SKSE_ARCENE_DIR`.

The classification results reported in the paper are in `results/`.

## Layout

```
src/skse/       the pip-installable package (loss + SKSE estimator)
experiments/    pipeline, baselines (AE, Isomap, t-SNE, UMAP), classifiers, reporting
results/        result CSVs and LaTeX tables from the paper runs
tests/          unit tests (pytest)
```
