"""Scikit-learn style estimator for Stochastic k-Simplex Embedding."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .loss import metric_skse_loss, sample_simplices


class MLP(nn.Module):
    """Three-layer MLP that maps inputs to the embedding space."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.fc(x)


class SKSE:
    """Stochastic k-Simplex Embedding.

    Trains an MLP so that the volumes of randomly sampled simplices in the
    embedding match their volumes in the input space.

    Parameters
    ----------
    n_components : int
        Dimension of the embedding.
    n_points : int
        Number of points per sampled simplex (vertices).
    hidden_dim : int
        Width of the MLP hidden layers.
    epochs : int
        Number of passes over the training data.
    lr : float
        AdamW learning rate.
    batch_size : int
        Mini-batch size; simplices are sampled within each mini-batch.
    n_simplices : int
        Number of simplices sampled per mini-batch.
    normalize : {"batch", "quality", None}
        How simplex volumes are normalized before comparison.
    random_state : int or None
        Seed for weight init, shuffling, and simplex sampling.
    device : str or None
        Torch device. Defaults to CUDA when available.
    verbose : bool
        Print the loss after each epoch.
    """

    def __init__(
        self,
        n_components: int = 2,
        n_points: int = 3,
        hidden_dim: int = 128,
        epochs: int = 1000,
        lr: float = 1e-4,
        batch_size: int = 32,
        n_simplices: int = 16,
        normalize: str | None = "batch",
        random_state: int | None = None,
        device: str | None = None,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.n_points = n_points
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.n_simplices = n_simplices
        self.normalize = normalize
        self.random_state = random_state
        self.device = device
        self.verbose = verbose
        self.model_ = None

    def _device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X, y=None) -> "SKSE":
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape {X.shape}")
        if len(X) < self.n_points:
            raise ValueError(f"Need at least n_points={self.n_points} samples, got {len(X)}")

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        device = self._device()
        self.model_ = MLP(X.shape[1], self.hidden_dim, self.n_components).to(device)
        optimizer = optim.AdamW(self.model_.parameters(), lr=self.lr)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        dataset = TensorDataset(X_tensor, torch.zeros(len(X_tensor)))
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(X)), shuffle=True)

        for epoch in range(self.epochs):
            self.model_.train()
            total = 0.0
            for data, _ in loader:
                n = data.size(0)
                data = data.to(device)
                optimizer.zero_grad()
                y_pred = self.model_(data)
                simplices = sample_simplices(n, batch_size=self.n_simplices, points=self.n_points).to(device)
                loss = metric_skse_loss(data, y_pred, simplices, normalize=self.normalize)
                loss.backward()
                optimizer.step()
                total += loss.item()
            if self.verbose:
                print(f"epoch {epoch + 1}/{self.epochs}  loss {total:.6g}")
        return self

    def transform(self, X) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("SKSE is not fitted yet. Call fit() first.")
        self.model_.eval()
        device = next(self.model_.parameters()).device
        X_tensor = torch.tensor(np.asarray(X, dtype=np.float32)).to(device)
        with torch.no_grad():
            return self.model_(X_tensor).cpu().numpy()

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X).transform(X)

    def __repr__(self) -> str:
        return (
            f"SKSE(n_components={self.n_components}, n_points={self.n_points}, "
            f"epochs={self.epochs}, lr={self.lr}, random_state={self.random_state})"
        )
