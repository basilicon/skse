import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from paths import PROCESSED_DIR, RESULTS_DIR, DATASETS

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.fc(x)

def train_eval_mlp(X_train, y_train, X_test, y_test, epochs=100, lr=1e-4, batch_size=32, device='cuda'):
    torch.manual_seed(42)
    model = SimpleMLP(X_train.shape[1], 64, 1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X_tr_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    
    ds = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(ds, batch_size=min(batch_size, len(X_train)), shuffle=True)

    model.train()
    for epoch in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    model.eval()
    X_te_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(X_te_t)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    preds = (probs > 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    auprc = average_precision_score(y_test, probs)
    return acc, auprc

def evaluate_unreduced_dataset(name, X, y):
    print(f"\n--- Evaluating Unreduced Baseline on {name} (X: {X.shape}) ---", flush=True)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {
        'KNN': {'acc': [], 'auprc': []},
        'SVM': {'acc': [], 'auprc': []},
        'GBDT': {'acc': [], 'auprc': []},
        'MLP': {'acc': [], 'auprc': []},
    }
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        # 1. KNN
        t0 = time.time()
        knn = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
        p_knn = knn.predict_proba(X_te)[:, 1]
        results['KNN']['acc'].append(accuracy_score(y_te, knn.predict(X_te)))
        results['KNN']['auprc'].append(average_precision_score(y_te, p_knn))
        print(f"  Fold {fold} KNN: {time.time()-t0:.1f}s", flush=True)
        
        # 2. SVM
        t0 = time.time()
        svm = SVC(probability=True, kernel='rbf', cache_size=1000, random_state=42).fit(X_tr, y_tr)
        p_svm = svm.predict_proba(X_te)[:, 1]
        results['SVM']['acc'].append(accuracy_score(y_te, svm.predict(X_te)))
        results['SVM']['auprc'].append(average_precision_score(y_te, p_svm))
        print(f"  Fold {fold} SVM: {time.time()-t0:.1f}s", flush=True)
        
        # 3. GBDT
        t0 = time.time()
        gbdt = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_features='sqrt', random_state=42).fit(X_tr, y_tr)
        p_gbdt = gbdt.predict_proba(X_te)[:, 1]
        results['GBDT']['acc'].append(accuracy_score(y_te, gbdt.predict(X_te)))
        results['GBDT']['auprc'].append(average_precision_score(y_te, p_gbdt))
        print(f"  Fold {fold} GBDT: {time.time()-t0:.1f}s", flush=True)
        
        # 4. MLP
        t0 = time.time()
        m_acc, m_auprc = train_eval_mlp(X_tr, y_tr, X_te, y_te, device='cuda')
        results['MLP']['acc'].append(m_acc)
        results['MLP']['auprc'].append(m_auprc)
        print(f"  Fold {fold} MLP: {time.time()-t0:.1f}s", flush=True)
        
    summary = []
    for clf in ['KNN', 'SVM', 'GBDT', 'MLP']:
        summary.append({
            'dataset': name,
            'classifier': clf,
            'mean_acc': np.mean(results[clf]['acc']),
            'std_acc': np.std(results[clf]['acc']),
            'mean_auprc': np.mean(results[clf]['auprc']),
            'std_auprc': np.std(results[clf]['auprc'])
        })
        print(f"  {clf} Mean ACC: {np.mean(results[clf]['acc']):.4f}, Mean AUPRC: {np.mean(results[clf]['auprc']):.4f}", flush=True)
    return summary

def main():
    all_summaries = []
    for ds in DATASETS:
        t_load = time.time()
        X = pd.read_csv(PROCESSED_DIR / ds / 'X.csv', header=None).values.astype(np.float32)
        y = pd.read_csv(PROCESSED_DIR / ds / 'y.csv', header=None).values.flatten().astype(int)
        print(f"Loaded {ds} in {time.time()-t_load:.1f}s", flush=True)
        summ = evaluate_unreduced_dataset(ds, X, y)
        all_summaries.extend(summ)
        
    df = pd.DataFrame(all_summaries)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(RESULTS_DIR / 'unreduced_baseline_results.csv', index=False)
    print("\n=== FINAL UNREDUCED BASELINE RESULTS ===", flush=True)
    print(df.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()
