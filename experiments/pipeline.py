import os
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.preprocessing import label_binarize
from datetime import datetime
import time
import argparse

from embedders.models import SimplexReducer, AutoencoderReducer, IsomapReducer, TSNEReducer, UMAPReducer
from classify.models import MLPClassifier, SVMClassifier, GBDTClassifier, KNNClassifier
from reporting.report import generate_report
from paths import PROCESSED_DIR, EMBED_DIR, RESULTS_DIR, DATASETS

def run_pipeline(dataset_name, dataset_dir, results_base_dir):
    print(f"\n=====================================")
    print(f"Processing dataset {dataset_name}")
    print(f"=====================================")
    
    X = pd.read_csv(os.path.join(dataset_dir, 'X.csv'), header=None).values.astype(np.float32)
    y = pd.read_csv(os.path.join(dataset_dir, 'y.csv'), header=None).values.flatten()
    
    output_dims = [2, 4, 8, 16, 32]
    n_points_list = [2, 3, 5, 8, 10]
    methods = ["simplex", "AE", "Isomap", "TSNE", "UMAP"]
    seeds = [42, 43, 44]
    
    # Generate embeddings
    print(f"--- Running embeddings for {dataset_name} ---")
    for method in methods:
        method_seeds = seeds
        for seed in method_seeds:
            embed_dir = os.path.join(EMBED_DIR, f"{dataset_name}_{seed}")
            os.makedirs(embed_dir, exist_ok=True)
            
            for dim in output_dims:
                if method == "simplex":
                    for n_points in n_points_list:
                        kf_3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                        for fold_idx, (train_idx, test_idx) in enumerate(kf_3.split(X, y)):
                            embed_path = os.path.join(embed_dir, f"{method}_{n_points}simplex_{dim}_fold{fold_idx}.pth")
                            time_taken = None
                            if os.path.exists(embed_path):
                                try:
                                    data = torch.load(embed_path, map_location='cpu')
                                    if isinstance(data, dict) and 'time_taken' in data:
                                        time_taken = data['time_taken']
                                except Exception:
                                    pass
                            
                            if time_taken is None:
                                print(f"Running {method} with n_points={n_points}, dim={dim}, seed={seed}, fold={fold_idx}")
                                t0 = time.time()
                                reducer = SimplexReducer(output_dim=dim, n_points=n_points, random_state=seed + fold_idx)
                                reducer.fit(X[train_idx])
                                Y_fold = reducer.transform(X)
                                time_taken = time.time() - t0
                                torch.save({
                                    'embeddings': torch.tensor(Y_fold, dtype=torch.float32),
                                    'labels': torch.tensor(y, dtype=torch.long),
                                    'time_taken': time_taken
                                }, embed_path)
                else:
                    kf_3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
                    for fold_idx, (train_idx, test_idx) in enumerate(kf_3.split(X, y)):
                        embed_path = os.path.join(embed_dir, f"{method}_{dim}_fold{fold_idx}.pth")
                        time_taken = None
                        if os.path.exists(embed_path):
                            try:
                                data = torch.load(embed_path, map_location='cpu')
                                if isinstance(data, dict) and 'time_taken' in data:
                                    time_taken = data['time_taken']
                            except Exception:
                                pass
                                
                        if time_taken is None:
                            print(f"Running {method} with dim={dim}, seed={seed}, fold={fold_idx}")
                            t0 = time.time()
                            if method == "AE":
                                reducer = AutoencoderReducer(output_dim=dim, random_state=seed + fold_idx)
                            elif method == "Isomap":
                                reducer = IsomapReducer(output_dim=dim, random_state=seed + fold_idx)
                            elif method == "TSNE":
                                reducer = TSNEReducer(output_dim=dim, random_state=seed + fold_idx)
                            elif method == "UMAP":
                                reducer = UMAPReducer(output_dim=dim, random_state=seed + fold_idx)
                            reducer.fit(X[train_idx])
                            Y_fold = reducer.transform(X)
                            time_taken = time.time() - t0
                            torch.save({
                                'embeddings': torch.tensor(Y_fold, dtype=torch.float32),
                                'labels': torch.tensor(y, dtype=torch.long),
                                'time_taken': time_taken
                            }, embed_path)

    # Classification
    print(f"\n--- Running classification for {dataset_name} ---")
    if len(np.unique(y)) <= 1:
        print(f"Skipping classification for {dataset_name} because it has only 1 class.")
        return
        
    clf_classes = {"MLP": MLPClassifier, "SVM": SVMClassifier, "GBDT": GBDTClassifier, "KNN": KNNClassifier}
    
    results = []
    
    for method in methods:
        configs = [(n, d) for n in n_points_list for d in output_dims] if method == "simplex" else [(None, d) for d in output_dims]
        method_seeds = seeds
        
        for n_point, in_dim in configs:
            for seed in method_seeds:
                embed_dir = os.path.join(EMBED_DIR, f"{dataset_name}_{seed}")
                
                avg_acc = 0
                avg_auc = 0
                folds_processed = 0
                
                for fold_outer in range(3):
                    if method == "simplex":
                        embed_path = os.path.join(embed_dir, f"{method}_{n_point}simplex_{in_dim}_fold{fold_outer}.pth")
                    else:
                        embed_path = os.path.join(embed_dir, f"{method}_{in_dim}_fold{fold_outer}.pth")
                        
                    if not os.path.exists(embed_path):
                        continue
                        
                    data = torch.load(embed_path)
                    X_all = data['embeddings'].cpu().numpy()
                    y_all = data['labels'].cpu().numpy()
                    embed_time = data.get('time_taken', np.nan)
                    
                    folds_processed += 1

                    for clf_name, clf_class in clf_classes.items():
                        unique_classes, counts = np.unique(y_all, return_counts=True)
                        if len(unique_classes) > 1 and min(counts) >= 5:
                            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                        else:
                            kf = KFold(n_splits=5, shuffle=True, random_state=42)

                        for fold, (train_idx, test_idx) in enumerate(kf.split(X_all, y_all)):
                            X_train_fold, X_test_fold = X_all[train_idx], X_all[test_idx]
                            y_train_fold, y_test_fold = y_all[train_idx], y_all[test_idx]
                            
                            clf = clf_class(random_state=fold)
                            clf.fit(X_train_fold, y_train_fold)
                            
                            preds = clf.predict(X_test_fold)
                            probs = clf.predict_proba(X_test_fold)
                            
                            acc = accuracy_score(y_test_fold, preds)
                            
                            # For AUPRC with multiple classes or single class edge cases
                            try:
                                if probs.ndim == 1:
                                    auprc = average_precision_score(y_test_fold, probs)
                                else:
                                    classes = np.unique(y_all)
                                    y_bin = label_binarize(y_test_fold, classes=classes)
                                    auprc = average_precision_score(y_bin, probs, average="macro")
                            except ValueError:
                                auprc = np.nan

                            avg_acc += acc
                            avg_auc += auprc

                            results.append({
                                "dataset": dataset_name,
                                "method": method,
                                "n_point": n_point,
                                "in_dim": in_dim,
                                "seed": seed,
                                "outer_fold": fold_outer,
                                "clf": clf_name,
                                "fold": fold,
                                "acc": acc,
                                "auprc": auprc,
                                "embed_time": embed_time
                            })
                            
                if folds_processed > 0:
                    total_runs = len(clf_classes) * 5 * folds_processed
                    print(f"{method} with n_points={n_point}, dim={in_dim}, seed={seed}, avg_acc={avg_acc/total_runs:.4f}, avg_auprc={avg_auc/total_runs:.4f}")

    # Save results
    results_dir = os.path.join(results_base_dir, dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    df_results = pd.DataFrame(results)
    results_csv_path = os.path.join(results_dir, "classification_results.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"Saved classification results to {results_csv_path}")

    # Generate reporting
    print(f"\n--- Running reporting for {dataset_name} ---")
    generate_report(results_csv_path, results_dir, readable_name=dataset_name)

def main():
    parser = argparse.ArgumentParser(description="Run embeddings, classification, and reporting.")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, help="Datasets under data/processed to run.")
    args = parser.parse_args()

    processed_dir = PROCESSED_DIR
    if not os.path.exists(processed_dir):
        print("No processed datasets found. Please run experiments/preprocess.py first.")
        return

    results_base_dir = RESULTS_DIR

    for dataset in args.datasets:
        dataset_dir = os.path.join(processed_dir, dataset)
        if os.path.isdir(dataset_dir):
            run_pipeline(dataset, dataset_dir, results_base_dir)

if __name__ == '__main__':
    main()
