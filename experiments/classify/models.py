import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from .base import Classifier

class MLPClassifierImpl(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
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

class MLPClassifier(Classifier):
    def __init__(self, random_state=42, epochs=1000, lr=1e-4, batch_size=32, hidden_dim=64):
        super().__init__(random_state)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 2

    def fit(self, X_train, y_train):
        torch.manual_seed(self.random_state)
        input_dim = X_train.shape[1]
        self.num_classes = len(np.unique(y_train))
        output_dim = 1 if self.num_classes == 2 else self.num_classes
        
        self.model = MLPClassifierImpl(input_dim=input_dim, hidden_dim=self.hidden_dim, output_dim=output_dim).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        
        if self.num_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
            y_t = torch.tensor(y_train, dtype=torch.float32, device=self.device).unsqueeze(1)
        else:
            criterion = nn.CrossEntropyLoss()
            y_t = torch.tensor(y_train, dtype=torch.long, device=self.device)

        X_t = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        
        train_ds = TensorDataset(X_t, y_t)
        loader_batch_size = min(self.batch_size, len(X_train))
        dataloader = DataLoader(train_ds, batch_size=loader_batch_size, shuffle=True)

        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_t)
            loss = criterion(outputs, y_t)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        probs = self.predict_proba(X)
        if self.num_classes == 2:
            return (probs > 0.5).astype(float)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(X_t)
            if self.num_classes == 2:
                probs = torch.sigmoid(out).cpu().numpy().flatten()
            else:
                probs = torch.softmax(out, dim=1).cpu().numpy()
        return probs

class SVMClassifier(Classifier):
    def __init__(self, random_state=42, n_estimators=10, max_samples=0.8):
        super().__init__(random_state)
        base_svm = SVC(probability=True, kernel='rbf', random_state=random_state)
        self.model = BaggingClassifier(estimator=base_svm, n_estimators=n_estimators, 
                                       random_state=random_state, max_samples=max_samples)
        self.num_classes = 2

    def fit(self, X_train, y_train):
        self.num_classes = len(np.unique(y_train))
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        probs = self.model.predict_proba(X)
        if self.num_classes == 2:
            return probs[:, 1]
        return probs

class GBDTClassifier(Classifier):
    def __init__(self, random_state=42, n_estimators=100, learning_rate=0.1, subsample=0.8):
        super().__init__(random_state)
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            subsample=subsample,
            max_features='sqrt',
            random_state=random_state
        )
        self.num_classes = 2

    def fit(self, X_train, y_train):
        self.num_classes = len(np.unique(y_train))
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        probs = self.model.predict_proba(X)
        if self.num_classes == 2:
            return probs[:, 1]
        return probs

class KNNClassifier(Classifier):
    def __init__(self, random_state=42, n_neighbors=5):
        super().__init__(random_state)
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.num_classes = 2

    def fit(self, X_train, y_train):
        self.num_classes = len(np.unique(y_train))
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        probs = self.model.predict_proba(X)
        if self.num_classes == 2:
            return probs[:, 1]
        return probs
