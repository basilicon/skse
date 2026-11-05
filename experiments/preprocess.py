import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from paths import DATA_DIR, PROCESSED_DIR

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def extract_soft_data(file_path, class_dict):
    print(f'Extracting {file_path}...')
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    start_idx = 0
    subsets = {}
    current_subset = None
    for i, line in enumerate(lines):
        if line.startswith('!subset_description'):
            current_subset = line.split('=')[1].strip()
        if line.startswith('!subset_sample_id'):
            samples = line.split('=')[1].strip().split(',')
            if current_subset:
                subsets[current_subset] = [s.strip() for s in samples]
                
        if line.startswith('!dataset_table_begin'):
            start_idx = i + 1
            break
            
    header = lines[start_idx].strip().split('\t')
    data = []
    for line in lines[start_idx+1:]:
        if line.startswith('!dataset_table_end'):
            break
        data.append(line.strip().split('\t'))
        
    df = pd.DataFrame(data, columns=header)
    
    sample_cols = [c for c in df.columns if c.startswith('GSM')]
    X = df[sample_cols].values.T.astype(np.float32)
    sample_names = sample_cols
    
    y = np.zeros(len(sample_names))
    for i, sn in enumerate(sample_names):
        found = False
        for subset_name, samples in subsets.items():
            if sn in samples and subset_name in class_dict:
                y[i] = class_dict[subset_name]
                found = True
        if not found:
            y[i] = -999
            
    mask = y != -999
    X = X[mask]
    y = y[mask]
    
    # Handle NaN values explicitly
    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        col_mean[np.isnan(col_mean)] = 0.0
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])
        X[np.isnan(X)] = 0.0

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f'X shape: {X.shape}, y shape: {y.shape}')
    return X, y

def process_and_save(dataset_name, X, y):
    out_dir = PROCESSED_DIR / dataset_name
    ensure_dir(out_dir)
    pd.DataFrame(X).to_csv(os.path.join(out_dir, 'X.csv'), index=False, header=False)
    pd.DataFrame(y).to_csv(os.path.join(out_dir, 'y.csv'), index=False, header=False)
    print(f"Saved {dataset_name} to {out_dir}")

def main():
    # Raw files are read relative to DATA_DIR (default: <repo>/data). See README for downloads.
    os.chdir(DATA_DIR)
    
    # GDS4278
    if os.path.exists('GDS4278_full.soft/GDS4278_full.soft'):
        X, y = extract_soft_data('GDS4278_full.soft/GDS4278_full.soft', {'CEBPAdm': 1, 'CEBPAsm': 1, 'CEBPAwt': 0})
        # Note: mapping {1, -1} to {1, 0}
        process_and_save('GDS4278', X, y)

    # GDS4516
    if os.path.exists('GDS4516_full.soft/GDS4516_full.soft'):
        X, y = extract_soft_data('GDS4516_full.soft/GDS4516_full.soft', {'no metastasis': 0, 'metastatic recurrence': 1, 'metastasis': 1})
        process_and_save('GDS4516', X, y)

    # gene-expression-cancer
    if os.path.exists('gene-expression-cancer/data.csv'):
        print('Extracting gene-expression-cancer...')
        X_df = pd.read_csv('gene-expression-cancer/data.csv')
        if 'Unnamed: 0' in X_df.columns:
            X_df = X_df.drop(columns=['Unnamed: 0'])
        X = X_df.values.astype(np.float32)
        
        y_df = pd.read_csv('gene-expression-cancer/labels.csv')
        labels = y_df['Class'].values
        y = LabelEncoder().fit_transform(labels)
        
        if np.isnan(X).any():
            col_mean = np.nanmean(X, axis=0)
            col_mean[np.isnan(col_mean)] = 0.0
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
            X[np.isnan(X)] = 0.0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        process_and_save('gene-expression-cancer', X, y)

    # molecular-synth
    if os.path.exists('molecular-synth/fingerprints_multiview.npy'):
        print('Extracting molecular-synth...')
        X = np.load('molecular-synth/fingerprints_multiview.npy').astype(np.float32)
        y = np.zeros(len(X), dtype=int)
        
        if np.isnan(X).any():
            col_mean = np.nanmean(X, axis=0)
            col_mean[np.isnan(col_mean)] = 0.0
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])
            X[np.isnan(X)] = 0.0

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        process_and_save('molecular-synth', X, y)

    # arcene
    arcene_dir = os.environ.get("SKSE_ARCENE_DIR", os.path.join("arcene", "ARCENE"))
    arcene_train_data = os.path.join(arcene_dir, "arcene_train.data")
    arcene_train_labels = os.path.join(arcene_dir, "arcene_train.labels")
    arcene_valid_data = os.path.join(arcene_dir, "arcene_valid.data")
    arcene_valid_labels = os.path.join(arcene_dir, "arcene_valid.labels")

    if os.path.exists(arcene_train_data):
        print('Extracting arcene...')
        train_data = pd.read_csv(arcene_train_data, sep=r'\s+', header=None).values
        train_labels = pd.read_csv(arcene_train_labels, sep=r'\s+', header=None).values.flatten()
        
        valid_data = pd.read_csv(arcene_valid_data, sep=r'\s+', header=None).values
        valid_labels = pd.read_csv(arcene_valid_labels, sep=r'\s+', header=None).values.flatten()
        
        X = np.concatenate([train_data, valid_data], axis=0).astype(np.float32)
        y = np.concatenate([train_labels, valid_labels], axis=0)
        y = ((y + 1) / 2).astype(int) # map -1,1 to 0,1

        process_and_save('arcene', X, y)

if __name__ == '__main__':
    main()
