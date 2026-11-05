import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from paths import EMBED_DIR, FIGURES_DIR

# Set high-quality styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['legend.fontsize'] = 9

datasets = [('arcene_42', 'Arcene (Mass Spec)'), ('GDS4278_42', 'GDS4278 (Microarray)')]
methods = [
    ('simplex_2simplex_2_fold0.pth', 'SKSE (k=2)'),
    ('simplex_3simplex_2_fold0.pth', 'SKSE (k=3)'),
    ('UMAP_2_fold0.pth', 'UMAP'),
    ('TSNE_2_fold0.pth', 't-SNE'),
    ('AE_2_fold0.pth', 'Autoencoder')
]

fig, axes = plt.subplots(len(datasets), len(methods), figsize=(15, 6), sharex=False, sharey=False)

colors = ['#1f77b4', '#d62728'] # Blue: Class 0, Red: Class 1

for row_idx, (ds_name, ds_label) in enumerate(datasets):
    for col_idx, (m_file, m_label) in enumerate(methods):
        ax = axes[row_idx, col_idx]
        file_path = os.path.join(EMBED_DIR, ds_name, m_file)
        
        if not os.path.exists(file_path):
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center')
            continue
            
        data = torch.load(file_path, map_location='cpu')
        embs = data['embeddings'].numpy()
        labels = data['labels'].numpy()
        
        # Standardize for visual comparison
        embs_std = (embs - np.mean(embs, axis=0)) / (np.std(embs, axis=0) + 1e-8)
        
        c0 = labels == 0
        c1 = labels == 1
        
        ax.scatter(embs_std[c0, 0], embs_std[c0, 1], c=colors[0], label='Control / Class 0', alpha=0.75, edgecolors='none', s=28)
        ax.scatter(embs_std[c1, 0], embs_std[c1, 1], c=colors[1], label='Tumor / Class 1', alpha=0.85, edgecolors='k', linewidths=0.5, s=34)
        
        if row_idx == 0:
            ax.set_title(m_label, fontweight='bold', pad=8)
        if col_idx == 0:
            ax.set_ylabel(ds_label, fontweight='bold', labelpad=8)
            
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
            spine.set_linewidth(0.8)

# Add custom global legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Control / Negative', markerfacecolor=colors[0], markersize=8),
    Line2D([0], [0], marker='o', color='w', label='Cancer / Positive', markerfacecolor=colors[1], markeredgecolor='k', markeredgewidth=0.5, markersize=9)
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=True, facecolor='#fafafa', edgecolor='#cccccc')

plt.tight_layout(rect=[0, 0, 1, 0.96])
out_dir = FIGURES_DIR
os.makedirs(out_dir, exist_ok=True)
out_fig = os.path.join(out_dir, 'plot_qualitative_2d_projections.png')
plt.savefig(out_fig, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved 2D qualitative projections to {out_fig}")
