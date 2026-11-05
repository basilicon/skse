import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import wilcoxon
from collections import defaultdict

from paths import RESULTS_DIR, FIGURES_DIR, DATASETS

# IEEE styling settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

def holm_bonferroni(p_values):
    items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(items)
    adj_p = {}
    running_max = 0.0
    for i, (k, p) in enumerate(items):
        adj = (m - i) * p
        running_max = max(running_max, adj)
        adj_p[k] = min(1.0, running_max)
    return adj_p

def polish_all_figures():
    datasets = DATASETS
    ds_titles = {'arcene': 'Arcene', 'GDS4278': 'GDS4278', 'GDS4516': 'GDS4516'}
    out_dir = FIGURES_DIR
    os.makedirs(out_dir, exist_ok=True)

    dfs = {}
    for ds in datasets:
        dfs[ds] = pd.read_csv(RESULTS_DIR / ds / 'classification_results.csv')

    # -------------------------------------------------------------
    # FIGURE 1: Best Simplex vs Baselines (AUPRC)
    # -------------------------------------------------------------
    # Harmonize y-axes per dataset or with clean margins
    method_styles = {
        'Best simplex': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'linewidth': 2.2, 'markersize': 7},
        'Autoencoder': {'color': '#ff7f0e', 'marker': '^', 'linestyle': '--', 'linewidth': 1.8, 'markersize': 6},
        'Isomap': {'color': '#2ca02c', 'marker': 's', 'linestyle': '-.', 'linewidth': 1.8, 'markersize': 6},
        't-SNE': {'color': '#d62728', 'marker': 'D', 'linestyle': ':', 'linewidth': 1.8, 'markersize': 6},
        'UMAP': {'color': '#9467bd', 'marker': 'v', 'linestyle': '-', 'linewidth': 1.8, 'markersize': 6}
    }
    method_names = {'simplex': 'Best simplex', 'AE': 'Autoencoder', 'Isomap': 'Isomap', 'TSNE': 't-SNE', 'UMAP': 'UMAP'}

    for ds in datasets:
        df = dfs[ds]
        plt.figure(figsize=(5.5, 4.2))
        dims = sorted(df['in_dim'].unique())
        
        # Best simplex per dim
        simplex_df = df[df['method'] == 'simplex']
        best_simplex_vals = []
        for d in dims:
            sub = simplex_df[simplex_df['in_dim'] == d]
            # mean over runs for each n_point, then take max
            n_means = sub.groupby('n_point')['auprc'].mean()
            best_simplex_vals.append(n_means.max())
            
        plt.plot(dims, best_simplex_vals, label='Best simplex', **method_styles['Best simplex'], zorder=5)
        
        for m in ['AE', 'Isomap', 'TSNE', 'UMAP']:
            m_df = df[df['method'] == m]
            vals = [m_df[m_df['in_dim'] == d]['auprc'].mean() for d in dims]
            plt.plot(dims, vals, label=method_names[m], **method_styles[method_names[m]])
            
        plt.xlabel('Embedding Dimension ($d$)')
        plt.ylabel('AUPRC')
        plt.title(f"{ds_titles[ds]}", fontweight='bold')
        plt.xticks(dims)
        plt.grid(True, linestyle='--', alpha=0.4)
        if ds == 'arcene':
            plt.ylim(0.40, 0.90)
            plt.legend(loc='lower right', frameon=True, framealpha=0.9)
        elif ds == 'GDS4278':
            plt.ylim(0.12, 0.65)
        elif ds == 'GDS4516':
            plt.ylim(0.34, 0.55)
            
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'plot_best_simplex_vs_others_auprc_{ds}.png'), dpi=300)
        plt.close()

    # -------------------------------------------------------------
    # FIGURE 2: Classifier Comparison (AUPRC)
    # -------------------------------------------------------------
    clfs = ['MLP', 'SVM', 'GBDT', 'KNN']
    methods_fig2 = ['Best simplex', 'Autoencoder', 'Isomap', 't-SNE', 'UMAP']
    bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for ds in datasets:
        df = dfs[ds]
        plt.figure(figsize=(6.0, 4.2))
        x = np.arange(len(clfs))
        total_bars = len(methods_fig2)
        bar_width = 0.16
        
        # calculate vals
        vals_by_method = {}
        # Best simplex per classifier (max average across n_points and dims, or mean across dims of best simplex)
        simplex_df = df[df['method'] == 'simplex']
        simp_vals = []
        for clf in clfs:
            sub = simplex_df[simplex_df['clf'] == clf]
            simp_vals.append(sub.groupby('n_point')['auprc'].mean().max())
        vals_by_method['Best simplex'] = simp_vals
        
        for m in ['AE', 'Isomap', 'TSNE', 'UMAP']:
            m_df = df[df['method'] == m]
            m_vals = [m_df[m_df['clf'] == clf]['auprc'].mean() for clf in clfs]
            vals_by_method[method_names[m]] = m_vals
            
        for i, m_name in enumerate(methods_fig2):
            offset = (i - (total_bars - 1) / 2) * bar_width
            plt.bar(x + offset, vals_by_method[m_name], width=bar_width, label=m_name, color=bar_colors[i], edgecolor='k', linewidth=0.5)

        plt.xticks(x, clfs)
        plt.xlabel('Downstream Classifier')
        plt.ylabel('AUPRC')
        plt.title(f"{ds_titles[ds]}", fontweight='bold')
        plt.grid(axis='y', linestyle='--', alpha=0.4)
        if ds == 'arcene':
            plt.ylim(0.0, 1.0)
            plt.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=9)
        elif ds == 'GDS4278':
            plt.ylim(0.0, 0.8)
        elif ds == 'GDS4516':
            plt.ylim(0.0, 0.7)
            
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'plot_classifier_comparison_auprc_{ds}.png'), dpi=300)
        plt.close()

    # -------------------------------------------------------------
    # FIGURE 3: Simplex Order vs Embedding Size (AUPRC)
    # -------------------------------------------------------------
    simplex_orders = [2, 3, 5, 8, 10]
    simplex_styles = {
        2: {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'linewidth': 2.2, 'markersize': 7},
        3: {'color': '#2ca02c', 'marker': 's', 'linestyle': '--', 'linewidth': 1.8, 'markersize': 6},
        5: {'color': '#ff7f0e', 'marker': '^', 'linestyle': '-.', 'linewidth': 1.8, 'markersize': 6},
        8: {'color': '#9467bd', 'marker': 'D', 'linestyle': ':', 'linewidth': 1.8, 'markersize': 6},
        10: {'color': '#8c564b', 'marker': 'v', 'linestyle': '-', 'linewidth': 1.6, 'markersize': 6}
    }

    for ds in datasets:
        df = dfs[ds]
        simplex_df = df[df['method'] == 'simplex']
        plt.figure(figsize=(5.5, 4.2))
        dims = sorted(simplex_df['in_dim'].unique())
        
        for k in simplex_orders:
            k_df = simplex_df[simplex_df['n_point'] == k]
            vals = [k_df[k_df['in_dim'] == d]['auprc'].mean() for d in dims]
            plt.plot(dims, vals, label=f'{k}-simplex', **simplex_styles[k])

        plt.xlabel('Embedding Dimension ($d$)')
        plt.ylabel('AUPRC')
        plt.title(f"{ds_titles[ds]}", fontweight='bold')
        plt.xticks(dims)
        plt.grid(True, linestyle='--', alpha=0.4)
        if ds == 'arcene':
            plt.ylim(0.60, 0.88)
            plt.legend(loc='lower right', frameon=True, framealpha=0.9)
        elif ds == 'GDS4278':
            plt.ylim(0.20, 0.65)
        elif ds == 'GDS4516':
            plt.ylim(0.42, 0.54)
            
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'plot_simplex_vs_embed_auprc_{ds}.png'), dpi=300)
        plt.close()

    # -------------------------------------------------------------
    # FIGURE 5: Streamlined Statistical Heatmaps with Holm-Bonferroni
    # -------------------------------------------------------------
    # We generate a polished 2-row (dim 16, dim 32) x 3-col figure
    # with large, clear font sizes and adjusted p-values.
    dims_fig5 = [16, 32]
    cmap = mcolors.ListedColormap(['#ff9999', '#f0f0f0', '#90ee90']) # Red, Gray, Green
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    summary_table_rows = []

    for dim in [2, 4, 8, 16, 32]:
        for ds in datasets:
            df = dfs[ds].copy()
            df['model_name'] = df.apply(lambda x: f"SKSE-{int(x['n_point'])}" if pd.notnull(x['n_point']) else x['method'], axis=1)
            sub = df[df['in_dim'] == dim].copy()
            sub['env_id'] = sub['clf'] + '_' + sub['outer_fold'].astype(str) + '_' + sub['fold'].astype(str) + '_' + sub['seed'].astype(str)
            piv = sub.pivot(index='env_id', columns='model_name', values='auprc').dropna()
            skse_models = sorted([m for m in piv.columns if 'SKSE' in m], key=lambda x: int(x.split('-')[1]))
            baselines = sorted([m for m in piv.columns if 'SKSE' not in m])
            raw_p = {}
            for b in baselines:
                for s in skse_models:
                    _, p = wilcoxon(piv[s], piv[b], alternative='two-sided')
                    raw_p[(b, s)] = p
            adj_p = holm_bonferroni(raw_p)
            
            wins = sum(1 for (b, s), p in adj_p.items() if p < 0.05 and piv[s].mean() > piv[b].mean())
            losses = sum(1 for (b, s), p in adj_p.items() if p < 0.05 and piv[s].mean() < piv[b].mean())
            ties = 20 - wins - losses
            summary_table_rows.append({
                'Dimension': rf"$\mathbb{{R}}^{{{dim}}}$",
                'Dataset': ds_titles[ds],
                'Wins': wins,
                'Ties': ties,
                'Losses': losses
            })

            # Save individual heatmaps for dim 16 and dim 32 with high resolution
            if dim in dims_fig5:
                fig, ax = plt.subplots(figsize=(6.5, 4.8))
                color_mat = np.zeros((len(baselines), len(skse_models)))
                p_mat = np.zeros((len(baselines), len(skse_models)))
                
                for r_idx, b in enumerate(baselines):
                    for c_idx, s in enumerate(skse_models):
                        p_val = adj_p[(b, s)]
                        p_mat[r_idx, c_idx] = p_val
                        if p_val < 0.05:
                            if piv[s].mean() > piv[b].mean():
                                color_mat[r_idx, c_idx] = 1 # Win
                            else:
                                color_mat[r_idx, c_idx] = -1 # Loss
                        else:
                            color_mat[r_idx, c_idx] = 0 # Tie
                            
                ax.imshow(color_mat, cmap=cmap, norm=norm, aspect='auto')
                
                # Labels
                bl_labels = {'AE': 'Autoencoder', 'Isomap': 'Isomap', 'TSNE': 't-SNE', 'UMAP': 'UMAP'}
                ax.set_xticks(np.arange(len(skse_models)))
                ax.set_yticks(np.arange(len(baselines)))
                ax.set_xticklabels([f"$k={m.split('-')[1]}$" for m in skse_models], fontsize=12)
                ax.set_yticklabels([bl_labels.get(b, b) for b in baselines], fontsize=12)
                
                # Annotate text with Holm-adjusted p-values
                for r_idx in range(len(baselines)):
                    for c_idx in range(len(skse_models)):
                        p_val = p_mat[r_idx, c_idx]
                        if p_val < 0.001:
                            txt = r"$p < 0.001$"
                        else:
                            txt = f"$p = {p_val:.3f}$"
                        ax.text(c_idx, r_idx, txt, ha="center", va="center", color="black", fontsize=11, fontweight='semibold')
                        
                ax.set_title(rf"{ds_titles[ds]} ($\mathbb{{R}}^{{{dim}}}$)", fontsize=13, fontweight='bold', pad=10)
                plt.tight_layout()
                fig_name = f"significance_heatmap_dim{dim}_AUPRC_{ds}.png"
                plt.savefig(os.path.join(out_dir, fig_name), dpi=300)
                plt.close()

    summary_df = pd.DataFrame(summary_table_rows)
    summary_path = RESULTS_DIR / 'wilcoxon_holm_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved Wilcoxon Holm-Bonferroni summary to {summary_path}")
    print("\n=== WILCOXON HOLM SUMMARY ===")
    print(summary_df.to_string(index=False))

if __name__ == '__main__':
    polish_all_figures()
