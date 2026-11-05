import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.manifold import Isomap, TSNE
from umap import UMAP

from .base import DimensionalityReducer
from skse import SKSE
from skse.model import MLP

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bottleneck_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

class SimplexReducer(DimensionalityReducer):
    """Thin wrapper around skse.SKSE with the settings used in the paper."""

    def __init__(self, output_dim, n_points, random_state=42, epochs=1000, lr=1e-4, batch_size=32):
        super().__init__(output_dim, random_state)
        self.n_points = n_points
        self.skse = SKSE(
            n_components=output_dim,
            n_points=n_points,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            n_simplices=16,
            random_state=random_state,
        )

    def fit(self, X):
        self.skse.fit(X)
        return self

    def transform(self, X):
        return self.skse.transform(X)

class AutoencoderReducer(DimensionalityReducer):
    def __init__(self, output_dim, random_state=42, epochs=2000, lr=1e-4, batch_size=32):
        super().__init__(output_dim, random_state)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(self, X):
        torch.manual_seed(self.random_state)
        self.model = Autoencoder(X.shape[1], 128, self.output_dim).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        train_dataset = TensorDataset(X_tensor, torch.zeros(len(X_tensor)))
        loader_batch_size = min(self.batch_size, len(X))
        train_loader = DataLoader(train_dataset, batch_size=loader_batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            self.model.train()
            for data, _ in train_loader:
                data = data.to(self.device)
                optimizer.zero_grad()
                recon, _ = self.model(data)
                loss = criterion(recon, data)
                loss.backward()
                optimizer.step()
        return self

    def transform(self, X):
        if self.model is None:
            raise ValueError("Model is not fitted yet!")
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, Y = self.model(X_tensor)
        return Y.cpu().numpy()

class IsomapReducer(DimensionalityReducer):
    def __init__(self, output_dim, random_state=42, n_neighbors=5):
        super().__init__(output_dim, random_state)
        self.n_neighbors = n_neighbors
        self.isomap = None
        
    def fit(self, X):
        n_neighbors = max(1, min(self.n_neighbors, len(X) - 1))
        self.isomap = Isomap(n_neighbors=n_neighbors, n_components=self.output_dim)
        try:
            self.isomap.fit(X)
        except Exception as e:
            print(f"Isomap fit failed: {e}")
            self.isomap = None
        return self

    def transform(self, X):
        if self.isomap is not None:
            try:
                return self.isomap.transform(X)
            except Exception as e:
                print(f"Isomap transform failed: {e}")
        return np.zeros((len(X), self.output_dim))

class TSNEReducer(DimensionalityReducer):
    def __init__(self, output_dim, random_state=42, perplexity=10):
        super().__init__(output_dim, random_state)
        self.perplexity = perplexity
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def fit(self, X):
        # Limit perplexity to be at most (len(X) - 1) / 3
        perp = min(self.perplexity, max(1.0, (len(X) - 1) / 3.0))
        tsne_method = 'barnes_hut' if self.output_dim < 4 else 'exact'
        try:
            tsne = TSNE(n_components=self.output_dim, perplexity=perp, 
                        random_state=self.random_state, method=tsne_method)
            Y_tsne = tsne.fit_transform(X)
        except Exception as e:
            print(f"t-SNE fit_transform failed: {e}")
            Y_tsne = np.zeros((len(X), self.output_dim))
            
        torch.manual_seed(self.random_state)
        self.model = MLP(X.shape[1], 128, self.output_dim).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        Y_tensor = torch.tensor(Y_tsne, dtype=torch.float32).to(self.device)
        
        dataset = TensorDataset(X_tensor, Y_tensor)
        batch_size = min(32, len(X))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(500):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def transform(self, X):
        if self.model is None:
            raise ValueError("Model is not fitted yet!")
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            Y = self.model(X_tensor).cpu().numpy()
        return Y

class UMAPReducer(DimensionalityReducer):
    def __init__(self, output_dim, random_state=42, n_neighbors=10):
        super().__init__(output_dim, random_state)
        self.n_neighbors = n_neighbors
        self.umap_model = None
        
    def fit(self, X):
        n_neighbors = max(2, min(self.n_neighbors, len(X) - 1))
        self.umap_model = UMAP(n_components=self.output_dim, n_neighbors=n_neighbors, random_state=self.random_state)
        try:
            self.umap_model.fit(X)
        except Exception as e:
            print(f"UMAP fit failed: {e}")
            self.umap_model = None
        return self

    def transform(self, X):
        if self.umap_model is not None:
            try:
                return self.umap_model.transform(X)
            except Exception as e:
                print(f"UMAP transform failed: {e}")
        return np.zeros((len(X), self.output_dim))
