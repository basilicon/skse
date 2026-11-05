import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import wilcoxon
from collections import defaultdict

def parse_results(df):
    """
    Parses dataframe output from pipeline.py.
    The df should have columns: ['dataset', 'method', 'n_point', 'in_dim', 'clf', 'fold', 'acc', 'auprc']
    """
    df = df.copy()
    # rename for gen_tables
    df = df.rename(columns={"acc": "ACC", "auprc": "AUPRC"})
    return df

def wrap_center(latex_str):
    """Wraps the tabular environment in a center environment."""
    latex_str = latex_str.replace("\\begin{tabular}", "\\begin{center}\n\\begin{tabular}")
    latex_str = latex_str.replace("\\end{tabular}", "\\end{tabular}\n\\end{center}")
    return latex_str

def format_extrema(df):
    """Bolds the maximum and italicizes the minimum value in each numeric column."""
    df_out = df.copy()
    for col in df_out.columns:
        if pd.api.types.is_numeric_dtype(df_out[col]):
            max_val = df_out[col].max()
            min_val = df_out[col].min()
            
            def format_val(x):
                if pd.isna(x):
                    return ""
                formatted_num = f"{x:.4f}"
                
                # If max and min are identical, just return the number to avoid conflicting tags
                if max_val == min_val:
                    return formatted_num
                    
                if x == max_val:
                    return f"\\textbf{{{formatted_num}}}"
                elif x == min_val:
                    return f"\\textit{{{formatted_num}}}"
                return formatted_num
                
            df_out[col] = df_out[col].apply(format_val)
    return df_out

def make_method_full(df):
    """Combine method + n_point into single column and fix naming."""
    df = df.copy()
    
    # Rename TSNE
    method_map = {"TSNE": "t-SNE"}
    df["method"] = df["method"].apply(lambda x: method_map.get(x, x))
    
    df["Technique"] = df.apply(
        lambda row: row["method"] if pd.isna(row["n_point"])
        else f"{int(row['n_point'])}-simplex",
        axis=1
    )
    return df

def generate_latex_tables(df, output_path):
    df_full = make_method_full(df)
    
    def tech_sort_key(x):
        if "-simplex" in x:
            return (0, int(x.split('-')[0]), x) # Sorts simplex by number
        return (1, 0, x) # Sorts everything else alphabetically afterward

    unique_techs = df_full["Technique"].unique()
    sorted_techs = sorted(unique_techs, key=tech_sort_key)
    
    df_full["Technique"] = pd.Categorical(df_full["Technique"], categories=sorted_techs, ordered=True)

    tables_text = ""

    for metric in ["AUPRC", "ACC"]:
        grouped = df_full.groupby(["Technique", "in_dim"], observed=True)[metric].mean().unstack("in_dim")
        overall = df_full.groupby("Technique", observed=True)[metric].mean()
        grouped["Average"] = overall
        
        grouped.index = grouped.index.astype(str)
        simplex_rows = [idx for idx in grouped.index if "-simplex" in idx]
        if simplex_rows:
            best_skse = grouped.loc[simplex_rows].max()
            grouped.loc["Best simplex"] = best_skse
            # reorder after last simplex
            rows = list(grouped.index)
            last_simplex_idx = rows.index(simplex_rows[-1])
            best_simplex_idx = rows.index("Best simplex")
            if best_simplex_idx > last_simplex_idx:
                # move best simplex after last simplex
                rows.insert(last_simplex_idx + 1, rows.pop(best_simplex_idx))
                grouped = grouped.loc[rows]
            
        formatted_grouped = format_extrema(grouped)
        formatted_grouped.columns = [str(c) for c in formatted_grouped.columns]
        
        latex_out = formatted_grouped.to_latex(
            escape=False,
            index=True,
            caption=f"Performance comparison for {metric}",
            label=f"tab:results_{metric.lower()}",
        )
        tables_text += wrap_center(latex_out) + "\n\n"

    with open(os.path.join(output_path, "tables.tex"), "w") as f:
        f.write(tables_text)

def generate_plots(df_raw, output_path, readable_name=None):
    other_methods = ["Isomap", "AE", "TSNE", "UMAP"]
    dataset_title = f" ({readable_name})" if readable_name else ""
    
    # Organize results dictionary: results[(method, n_point, in_dim, clf)] = {"acc": ..., "auprc": ...}
    results = defaultdict(lambda: {"acc": [], "auprc": []})
    for _, r in df_raw.iterrows():
        n_point = r["n_point"] if not pd.isna(r["n_point"]) else None
        key = (r["method"], n_point, r["in_dim"], r["clf"])
        results[key]["acc"].append(r["ACC"])
        results[key]["auprc"].append(r["AUPRC"])

    for k in list(results.keys()):
        results[k]["acc"] = np.nanmean(results[k]["acc"])
        results[k]["auprc"] = np.nanmean(results[k]["auprc"])

    def average_over_classifiers(method, n_point=None, metric="acc"):
        dims = sorted(set(k[2] for k in results if k[0] == method and (n_point is None or k[1] == n_point)))
        values = []
        for d in dims:
            vals = [results[k][metric] for k in results if k[0] == method and k[2] == d and (n_point is None or k[1] == n_point)]
            values.append(np.nanmean(vals) if vals else np.nan)
        return np.array(dims), np.array(values)

    def average_over_npoints_and_classifiers(method, metric="acc"):
        dims = sorted(set(k[2] for k in results if k[0] == method))
        values = []
        for d in dims:
            vals = [results[k][metric] for k in results if k[0] == method and k[2] == d]
            values.append(np.nanmean(vals) if vals else np.nan)
        return np.array(dims), np.array(values)

    def best_simplex_per_dim(metric="acc"):
        dims = sorted(set(k[2] for k in results if k[0] == "simplex"))
        best_vals = []
        for d in dims:
            npoints = sorted(set(k[1] for k in results if k[0] == "simplex" and k[2] == d and k[1] is not None))
            if not npoints:
                best_vals.append(np.nan)
                continue
            npoint_means = []
            for n in npoints:
                vals = [results[k][metric] for k in results if k[0] == "simplex" and k[2] == d and k[1] == n]
                npoint_means.append(np.nanmean(vals) if vals else np.nan)
            best_vals.append(np.nanmax(npoint_means) if any(~np.isnan(npoint_means)) else np.nan)
        return np.array(dims), np.array(best_vals)

    method_markers = {"simplex": "o", "Isomap": "s", "AE": "^", "TSNE": "D", "UMAP": "v"}
    method_labels = {"Isomap": "Isomap", "AE": "Autoencoder", "TSNE": "t-SNE", "UMAP": "UMAP"}
    simplex_npoint_markers = {2: "o", 3: "s", 5: "^", 8: "D", 10: "v"}

    for metric in ["acc", "auprc"]:
        plt.figure()
        for n_point in sorted(set(k[1] for k in results if k[0] == "simplex" and k[1] is not None)):
            dims, vals = average_over_classifiers("simplex", n_point=n_point, metric=metric)
            plt.plot(dims, vals, marker=simplex_npoint_markers.get(n_point, 'x'), label=f"{n_point}-simplex")

        plt.xlabel("Embedding dimension")
        plt.ylabel(metric.upper())
        plt.title(f"Simplex performance vs embedding size ({metric.upper()}){dataset_title}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_path, f"plot_simplex_vs_embed_{metric}.png"), dpi=200)
        plt.close()

    for metric in ["acc", "auprc"]:
        plt.figure()
        dims, simplex_best = best_simplex_per_dim(metric)
        plt.plot(dims, simplex_best, marker=method_markers["simplex"], label="Best simplex")

        for method in other_methods:
            d2, v2 = average_over_npoints_and_classifiers(method, metric=metric)
            plt.plot(d2, v2, marker=method_markers[method], label=method_labels[method])
        plt.xlabel("Embedding dimension")
        plt.ylabel(metric.upper())
        plt.title(f"Best simplex vs other methods ({metric.upper()}){dataset_title}")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_path, f"plot_best_simplex_vs_others_{metric}.png"), dpi=200)
        plt.close()

    for metric in ["acc", "auprc"]:
        plt.figure()
        clf_names = sorted(set(k[3] for k in results))
        bar_width = 1 / (2 + len(other_methods))
        x = np.arange(len(clf_names))

        simplex_vals = []
        for clf in clf_names:
            vals = [results[k][metric] for k in results if k[0] == "simplex" and k[3] == clf]
            simplex_vals.append(np.nanmax(vals) if vals else np.nan)

        plt.bar(x - bar_width, simplex_vals, width=bar_width, label="Best simplex")

        for i, method in enumerate(other_methods):
            vals = []
            for clf in clf_names:
                vals_clf = [results[k][metric] for k in results if k[0] == method and k[3] == clf]
                vals.append(np.nanmean(vals_clf) if vals_clf else np.nan)
            plt.bar(x + (i * bar_width), vals, width=bar_width, label=method_labels[method])

        plt.xticks(x, clf_names)
        plt.ylabel(metric.upper())
        plt.title(f"Classifier comparison ({metric.upper()}){dataset_title}")
        plt.legend(loc='lower left')
        plt.grid(axis='y')
        plt.savefig(os.path.join(output_path, f"plot_classifier_comparison_{metric}.png"), dpi=200)
        plt.close()

def generate_stat_tests(df_raw, output_dir, readable_name=None):
    from scipy.stats import wilcoxon
    import matplotlib.colors as mcolors
    
    df = df_raw.copy()
    # Ensure there's a run/fold column
    if 'fold' not in df.columns:
        df['fold'] = 0
        
    df['model_name'] = df.apply(
        lambda x: f"SKSE-{int(x['n_point'])}" if pd.notnull(x['n_point']) else x['method'], 
        axis=1
    )
    
    def plot_significance_heatmap(df_subset, dim, metric='auprc'):
        subset = df_subset[df_subset['in_dim'] == dim].copy()
        if len(subset) == 0: return
        if 'outer_fold' in subset.columns:
            subset['env_id'] = subset['clf'] + "_" + subset['outer_fold'].astype(str) + "_" + subset['fold'].astype(str) + "_" + subset['seed'].astype(str)
        else:
            subset['env_id'] = subset['clf'] + "_" + subset['fold'].astype(str) + "_" + subset['seed'].astype(str)
        
        pivot_df = subset.pivot(index='env_id', columns='model_name', values=metric).dropna()
        if pivot_df.empty: return
        
        all_models = pivot_df.columns
        skse_models = sorted(
            [m for m in all_models if 'SKSE' in m], 
            key=lambda x: int(float(x.split('-')[1]))
        )
        baselines = sorted([m for m in all_models if 'SKSE' not in m])
        
        if not skse_models or not baselines: return
        
        p_matrix = pd.DataFrame(index=baselines, columns=skse_models)
        color_matrix = pd.DataFrame(index=baselines, columns=skse_models)
        
        for base in baselines:
            for skse in skse_models:
                # Pairwise Wilcoxon Test
                try:
                    _, p_val = wilcoxon(pivot_df[skse], pivot_df[base], alternative='two-sided')
                except ValueError:
                    p_val = 1.0 # If all differences are 0
                p_matrix.loc[base, skse] = p_val
                
                if p_val < 0.05:
                    if pivot_df[skse].mean() > pivot_df[base].mean():
                        color_matrix.loc[base, skse] = 1   
                    else:
                        color_matrix.loc[base, skse] = -1  
                else:
                    color_matrix.loc[base, skse] = 0       

        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = mcolors.ListedColormap(['#ff9999', '#eeeeee', '#99ff99'])
        bounds = [-1.5, -0.5, 0.5, 1.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        im = ax.imshow(color_matrix.values.astype(float), cmap=cmap, norm=norm)
        
        model_labels = {"Isomap": "Isomap", "AE": "Autoencoder", "TSNE": "t-SNE", "UMAP": "UMAP"}
        
        ax.set_xticks(np.arange(len(skse_models)))
        ax.set_yticks(np.arange(len(baselines)))
        ax.set_xticklabels([str(m).replace('SKSE-', '') for m in skse_models])
        ax.set_yticklabels([model_labels.get(m, m) for m in baselines])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        for i in range(len(baselines)):
            for j in range(len(skse_models)):
                p = p_matrix.iloc[i, j]
                text = f"$p={p:.3f}$" if p >= 0.001 else "$p<0.001$"
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=10)

        prefix = f"{readable_name} - " if readable_name else ""
        ax.set_title(f"{prefix}Statistical Significance: SKSE vs Baselines (dim={dim}, {metric.upper()})\nGreen: SKSE Win | Red: SKSE Loss | Gray: Not Significant")
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f"significance_heatmap_dim{dim}_{metric}.png"), dpi=200)
        plt.close(fig)

    for dim in df['in_dim'].unique():
        plot_significance_heatmap(df, dim, metric='AUPRC')
        plot_significance_heatmap(df, dim, metric='ACC')

def format_extrema_time(df):
    """Bolds the minimum (fastest) and italicizes the maximum (slowest) value in each numeric column."""
    df_out = df.copy()
    for col in df_out.columns:
        if pd.api.types.is_numeric_dtype(df_out[col]):
            max_val = df_out[col].max()
            min_val = df_out[col].min()
            
            def format_val(x):
                if pd.isna(x):
                    return ""
                formatted_num = f"{x:.4f}"
                
                # If max and min are identical, just return the number to avoid conflicting tags
                if max_val == min_val:
                    return formatted_num
                    
                if x == min_val:
                    return f"\\textbf{{{formatted_num}}}"
                elif x == max_val:
                    return f"\\textit{{{formatted_num}}}"
                return formatted_num
                
            df_out[col] = df_out[col].apply(format_val)
    return df_out

def generate_time_tables_and_plots(df, output_path, readable_name=None):
    df_full = make_method_full(df)
    
    def tech_sort_key(x):
        if "-simplex" in x:
            return (0, int(x.split('-')[0]), x) # Sorts simplex by number
        return (1, 0, x) # Sorts everything else alphabetically afterward

    unique_techs = df_full["Technique"].unique()
    sorted_techs = sorted(unique_techs, key=tech_sort_key)
    df_full["Technique"] = pd.Categorical(df_full["Technique"], categories=sorted_techs, ordered=True)

    # Note: df has duplicate rows of embed_time for each clf and fold.
    # We want to take the first occurrence or mean for each (Technique, in_dim, seed)
    df_unique = df_full.groupby(["Technique", "in_dim", "seed"], observed=True)["embed_time"].mean().reset_index()
    time_grouped = df_unique.groupby(["Technique", "in_dim"], observed=True)["embed_time"].mean().unstack("in_dim")
    
    overall = df_unique.groupby("Technique", observed=True)["embed_time"].mean()
    time_grouped["Overall"] = overall
    
    # Generate Best simplex per column if there are simplex techniques
    simplex_rows = [idx for idx in time_grouped.index if "-simplex" in idx]
    if simplex_rows:
        # For time, we want the MINIMUM time among simplexes!
        best_skse = time_grouped.loc[simplex_rows].min()
        time_grouped.loc["Best simplex"] = best_skse
        # reorder after last simplex
        rows = list(time_grouped.index)
        last_simplex_idx = rows.index(simplex_rows[-1])
        best_simplex_idx = rows.index("Best simplex")
        if best_simplex_idx > last_simplex_idx:
            rows.insert(last_simplex_idx + 1, rows.pop(best_simplex_idx))
            time_grouped = time_grouped.loc[rows]
            
    formatted_grouped = format_extrema_time(time_grouped)
    formatted_grouped.columns = [str(c) for c in formatted_grouped.columns]
    
    latex_out = formatted_grouped.to_latex(
        escape=False,
        index=True,
        caption="Embedding execution time comparison (seconds, lower is better)",
        label="tab:results_embed_time",
    )
    latex_out = wrap_center(latex_out)
    
    with open(os.path.join(output_path, "time_table.tex"), "w") as f:
        f.write(latex_out)
        
    # Plotting
    dataset_title = f" ({readable_name})" if readable_name else ""
    method_markers = {"simplex": "o", "Isomap": "s", "AE": "^", "TSNE": "D", "UMAP": "v"}
    method_labels = {"Isomap": "Isomap", "AE": "Autoencoder", "TSNE": "t-SNE", "UMAP": "UMAP"}
    simplex_npoint_markers = {2: "o", 3: "s", 5: "^", 8: "D", 10: "v"}
    
    plt.figure()
    
    # Plot simplex envelopes (slowest and fastest among any simplex for each dimension)
    if simplex_rows:
        actual_simplex_rows = [r for r in simplex_rows if r != "Best simplex"]
        dims = [col for col in time_grouped.columns if isinstance(col, (int, float))]
        simplex_data = time_grouped.loc[actual_simplex_rows, dims]
        
        best_simplex_envelope = simplex_data.min(axis=0) # fastest at each dimension
        worst_simplex_envelope = simplex_data.max(axis=0) # slowest at each dimension
        
        plt.plot(best_simplex_envelope.index, best_simplex_envelope.values, marker="o", label="Best simplex (fastest)")
        plt.plot(worst_simplex_envelope.index, worst_simplex_envelope.values, marker="x", label="Worst simplex (slowest)")
            
    # Plot other techniques
    for method in ["Isomap", "AE", "TSNE", "UMAP"]:
        tech_name = "t-SNE" if method == "TSNE" else method
        if tech_name in time_grouped.index:
            dims = [col for col in time_grouped.columns if isinstance(col, (int, float))]
            vals = time_grouped.loc[tech_name, dims]
            plt.plot(vals.index, vals.values, marker=method_markers[method], label=method_labels[method])
            
    plt.yscale("log")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Execution Time (seconds, log scale)")
    plt.title(f"Embedding execution time comparison{dataset_title}")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "plot_embedding_time_vs_dim.png"), dpi=200)
    plt.close()

    # Create new chart for only the simplex techniques and none of the baselines
    plt.figure()
    
    for tech in [r for r in simplex_rows if r != "Best simplex"]:
        if tech in time_grouped.index:
            dims = [col for col in time_grouped.columns if isinstance(col, (int, float))]
            vals = time_grouped.loc[tech, dims]
            try:
                n_pt = int(tech.split('-')[0])
            except Exception:
                n_pt = 2
            plt.plot(vals.index, vals.values, marker=simplex_npoint_markers.get(n_pt, 'x'), label=tech)
            
    plt.yscale("log")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Execution Time (seconds, log scale)")
    plt.title(f"Simplex execution time comparison{dataset_title}")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "plot_simplex_time_vs_dim.png"), dpi=200)
    plt.close()

def generate_report(results_csv_path, output_dir, readable_name=None):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(results_csv_path)
    df_parsed = parse_results(df)
    generate_latex_tables(df_parsed, output_dir)
    generate_plots(df_parsed, output_dir, readable_name=readable_name)
    generate_stat_tests(df_parsed, output_dir, readable_name=readable_name)
    if "embed_time" in df_parsed.columns:
        generate_time_tables_and_plots(df_parsed, output_dir, readable_name=readable_name)

