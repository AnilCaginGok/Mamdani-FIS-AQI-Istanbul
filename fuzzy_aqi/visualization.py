import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.stats import norm as scipy_norm
from .fuzzy_system import TERM_KEYS, TERM_LABELS, TERM_COLORS
from .evaluation import CATEGORY_LABELS, CATEGORY_SHORT
plt.rcParams.update({'figure.dpi': 180, 'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'legend.fontsize': 10, 'lines.linewidth': 2.0})
_CMAP_CAT = mcolors.ListedColormap(TERM_COLORS)

def _label_color_for_hex(hex_color: str) -> str:
    rgb = mcolors.to_rgb(hex_color)
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return 'white' if luminance < 0.55 else 'black'

class Visualizer:

    def __init__(self, output_dir: str='results'):
        os.makedirs(output_dir, exist_ok=True)
        self.out = output_dir

    def plot_input_mfs(self, fis, save: bool=True):
        mf_data = fis.get_mf_values()
        (fig, axes) = plt.subplots(2, 1, figsize=(10, 11))
        fig.suptitle('Input Membership Functions — Mamdani FIS\n(dashed grey lines = EPA AQI category breakpoints)', fontweight='bold')
        _epa_pm25_breaks = [12.0, 35.4, 55.4, 150.4]
        _epa_no2_breaks = [101.4, 191.3]
        inputs = [('pm25', 'PM₂.₅ Concentration (μg/m³)', [0, 250], _epa_pm25_breaks), ('no2', 'NO₂ Concentration (μg/m³)', [0, 200], _epa_no2_breaks)]
        for (ax, (key, xlabel, xlim, epa_breaks)) in zip(axes, inputs):
            universe = mf_data[key]['universe']
            for (term, color) in zip(TERM_KEYS, TERM_COLORS):
                label = TERM_LABELS[TERM_KEYS.index(term)]
                ax.plot(universe, mf_data[key]['mfs'][term], color=color, label=label, linewidth=2.0)
                ax.fill_between(universe, mf_data[key]['mfs'][term], alpha=0.12, color=color)
            for bp in epa_breaks:
                ax.axvline(bp, color='dimgray', linestyle='--', linewidth=1.2, alpha=0.65)
                ax.text(bp, 1.04, f'{bp:.0f}', ha='center', va='bottom', fontsize=7.5, color='dimgray', fontstyle='italic')
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Membership Degree μ(x)')
            ax.set_xlim(xlim)
            ax.set_ylim([-0.05, 1.15])
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.set_title(key.upper() + ' Input MFs  (with EPA breakpoints)')
        handles, labels = axes[0].get_legend_handles_labels()
        if not handles:
            for (term, color) in zip(TERM_KEYS, TERM_COLORS):
                handles.append(plt.Line2D([0], [0], color=color, linewidth=2.0))
                labels.append(TERM_LABELS[TERM_KEYS.index(term)])
        fig.legend(handles, labels, loc='lower center', ncol=5, framealpha=0.92, bbox_to_anchor=(0.5, -0.01))
        plt.tight_layout(rect=[0, 0.04, 1, 0.97])
        if save:
            path = os.path.join(self.out, 'fig_mf_inputs.png')
            fig.savefig(path, bbox_inches='tight')
            print(f'  Saved: {path}')
        return fig

    def plot_output_mf(self, fis, save: bool=True):
        mf_data = fis.get_mf_values()
        universe = mf_data['aqi']['universe']
        (fig, ax) = plt.subplots(figsize=(11, 5))
        ax.set_title('Output Membership Functions — AQI Category Index', fontweight='bold')
        for (term, color) in zip(TERM_KEYS, TERM_COLORS):
            label = TERM_LABELS[TERM_KEYS.index(term)]
            ax.plot(universe, mf_data['aqi']['mfs'][term], color=color, label=label, linewidth=2.0)
            ax.fill_between(universe, mf_data['aqi']['mfs'][term], alpha=0.12, color=color)
        ax.text(0, 1.05, 'μ(0)=1', ha='left', fontsize=8, color='dimgray')
        ax.text(5, 1.05, 'μ(5)=1', ha='right', fontsize=8, color='dimgray')
        centers = [0.5, 1.5, 2.5, 3.5, 4.5]
        for (c, color) in zip(centers, TERM_COLORS):
            ax.axvline(c, color=color, linestyle=':', linewidth=1.2, alpha=0.7)
        ax.set_xlabel('AQI Category Index')
        ax.set_ylabel('Membership Degree μ(y)')
        ax.set_xlim([0, 5])
        ax.set_ylim([-0.05, 1.1])
        ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
        ax.set_xticklabels(['Good', 'Moderate', 'USG', 'Unhealthy', 'VU'], fontsize=9)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=5, framealpha=0.92, fontsize=8.5)
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        if save:
            path = os.path.join(self.out, 'fig_mf_output.png')
            fig.savefig(path, bbox_inches='tight')
            print(f'  Saved: {path}')
        return fig

    def plot_rule_table(self, fis, save: bool=True):
        mat = fis.get_rule_matrix().astype(float)
        short = ['Good', 'Moderate', 'USG', 'Unhealthy', 'VU']
        (fig, ax) = plt.subplots(figsize=(9, 7))
        ax.set_title('Fuzzy Rule Base — 5×5 Rule Matrix\n(Output = max severity of PM₂.₅ and NO₂ inputs)', fontweight='bold')
        im = ax.imshow(mat, cmap=_CMAP_CAT, vmin=-0.5, vmax=4.5, aspect='auto')
        for i in range(5):
            for j in range(5):
                cell_color = TERM_COLORS[int(mat[i, j])]
                ax.text(j, i, short[int(mat[i, j])], ha='center', va='center', fontsize=11, color=_label_color_for_hex(cell_color), fontweight='bold')
        ax.set_xticks(range(5))
        ax.set_xticklabels(short)
        ax.set_yticks(range(5))
        ax.set_yticklabels(short)
        ax.set_xlabel('NO₂ Linguistic Level')
        ax.set_ylabel('PM₂.₅ Linguistic Level')
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4])
        cbar.ax.set_yticklabels(['Good', 'Moderate', 'USG', 'Unhealthy', 'VU'])
        cbar.set_label('AQI Output Category')
        rule_n = 1
        for i in range(5):
            for j in range(5):
                ax.text(j, i - 0.38, f'R{rule_n}', ha='center', va='center', fontsize=7.5, color='dimgray')
                rule_n += 1
        plt.tight_layout()
        if save:
            path = os.path.join(self.out, 'fig_rule_table.png')
            fig.savefig(path, bbox_inches='tight')
            print(f'  Saved: {path}')
        return fig

    def plot_surface_3d(self, surface_data: dict, method: str='fuzzy', save: bool=True):
        PM = surface_data['PM25']
        NO = surface_data['NO2']
        key = 'AQI_fuzzy' if method == 'fuzzy' else 'AQI_crisp'
        Z = surface_data[key]
        title = 'Mamdani FIS Response Surface' if method == 'fuzzy' else 'Crisp EPA Response Surface'
        fig = plt.figure(figsize=(12, 8.5))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title, fontweight='bold', pad=14)
        surf = ax.plot_surface(PM, NO, Z, cmap='RdYlGn_r', vmin=0, vmax=5, alpha=0.9, linewidth=0, antialiased=True)
        ax.set_xlabel('PM₂.₅ (μg/m³)', labelpad=8)
        ax.set_ylabel('NO₂ (μg/m³)', labelpad=8)
        ax.set_zlabel('AQI Category Index', labelpad=8)
        ax.set_zlim(0, 5)
        ax.set_zticks([0.5, 1.5, 2.5, 3.5, 4.5])
        ax.set_zticklabels(['Good', 'Mod', 'USG', 'Unhlt', 'VU'], fontsize=7)
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, ticks=[0.5, 1.5, 2.5, 3.5, 4.5])
        cbar.ax.set_yticklabels(['Good', 'Moderate', 'USG', 'Unhealthy', 'V.Unhealthy'], fontsize=8)
        fname = f'fig_surface_{method}.png'
        if save:
            path = os.path.join(self.out, fname)
            fig.savefig(path, bbox_inches='tight')
            print(f'  Saved: {path}')
        return fig

    def plot_surface_comparison(self, surface_data: dict, save: bool=True):
        PM = surface_data['PM25']
        NO = surface_data['NO2']
        fig = plt.figure(figsize=(10, 14))
        fig.suptitle('AQI Response Surface: Mamdani FIS vs. Crisp EPA Method', fontweight='bold', fontsize=13)
        for (idx, (key, title)) in enumerate([('AQI_fuzzy', 'Mamdani FIS (Fuzzy)'), ('AQI_crisp', 'Crisp EPA Reference')]):
            ax = fig.add_subplot(2, 1, idx + 1, projection='3d')
            ax.set_title(title, fontsize=11)
            surf = ax.plot_surface(PM, NO, surface_data[key], cmap='RdYlGn_r', vmin=0, vmax=5, alpha=0.9, linewidth=0, antialiased=True)
            ax.set_xlabel('PM₂.₅ (μg/m³)', labelpad=6, fontsize=9)
            ax.set_ylabel('NO₂ (μg/m³)', labelpad=6, fontsize=9)
            ax.set_zlabel('AQI Index', labelpad=6, fontsize=9)
            ax.set_zlim(0, 5)
            ax.set_zticks([0.5, 1.5, 2.5, 3.5, 4.5])
            ax.set_zticklabels(['G', 'M', 'U', 'H', 'VU'], fontsize=7)
        fig.subplots_adjust(left=0.08, right=0.95, hspace=0.28)
        if save:
            path = os.path.join(self.out, 'fig_surface_compare.png')
            fig.savefig(path, bbox_inches='tight')
            print(f'  Saved: {path}')
        return fig

    def plot_surface_diff(self, surface_data: dict, save: bool=True):
        PM = surface_data['PM25']
        NO = surface_data['NO2']
        diff = np.abs(surface_data['AQI_fuzzy'] - surface_data['AQI_crisp'])
        (fig, ax) = plt.subplots(figsize=(8, 6))
        ax.set_title('Absolute Difference |Fuzzy – Crisp| AQI Index\n(darker = larger discrepancy)', fontweight='bold')
        c = ax.contourf(PM, NO, diff, levels=20, cmap='hot_r')
        ax.contour(PM, NO, diff, levels=5, colors='white', linewidths=0.5, alpha=0.5)
        cbar = fig.colorbar(c, ax=ax)
        cbar.set_label('|ΔAQI Index|')
        ax.set_xlabel('PM₂.₅ (μg/m³)')
        ax.set_ylabel('NO₂ (μg/m³)')
        max_idx = np.unravel_index(np.nanargmax(diff), diff.shape)
        ax.plot(PM[max_idx], NO[max_idx], 'b*', markersize=12, label=f'Max diff = {diff[max_idx]:.3f}')
        ax.legend(fontsize=9)
        plt.tight_layout()
        if save:
            path = os.path.join(self.out, 'fig_surface_diff.png')
            fig.savefig(path, bbox_inches='tight')
            print(f'  Saved: {path}')
        return fig

    def plot_scatter(self, metrics: dict, save: bool=True):
        fv = metrics['fis_vals']
        cv = metrics['crisp_vals']
        cidx = metrics['crisp_idx']
        (fig, ax) = plt.subplots(figsize=(7, 6))
        ax.set_title(f"Fuzzy vs. Crisp AQI Index\nAccuracy = {metrics['accuracy'] * 100:.1f}%  |  RMSE = {metrics['rmse']:.3f}  |  MAE = {metrics['mae']:.3f}", fontweight='bold')
        for (k, (cat, color)) in enumerate(zip(CATEGORY_LABELS, TERM_COLORS)):
            mask = cidx == k
            if mask.sum() > 0:
                short = CATEGORY_SHORT[k].replace('\n', ' ')
                ax.scatter(cv[mask], fv[mask], color=color, label=short, alpha=0.55, s=18, zorder=3)
        lims = [0, 5]
        ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect agreement', zorder=5)
        ax.set_xlabel('Crisp EPA AQI Index (reference)')
        ax.set_ylabel('Mamdani FIS AQI Index (predicted)')
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 5])
        ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
        ax.set_xticklabels(['Good', 'Moderate', 'USG', 'Unhealthy', 'VU'], fontsize=8)
        ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
        ax.set_yticklabels(['Good', 'Moderate', 'USG', 'Unhealthy', 'VU'], fontsize=8)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        if save:
            path = os.path.join(self.out, 'fig_scatter.png')
            fig.savefig(path, bbox_inches='tight')
            print(f'  Saved: {path}')
        return fig

    def plot_confusion_matrix(self, metrics: dict, save: bool=True):
        conf_mat = metrics['conf_matrix']
        labels = ['Good', 'Moderate', 'USG', 'Unhealthy', 'V.Unhealthy']
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_mat = conf_mat.astype(float) / conf_mat.sum(axis=1, keepdims=True)
            norm_mat = np.nan_to_num(norm_mat)
        (fig, axes) = plt.subplots(2, 1, figsize=(8, 10))
        fig.suptitle('Confusion Matrix: Mamdani FIS vs. Crisp EPA\n(rows = crisp reference, cols = fuzzy predicted)', fontweight='bold')
        for (ax, data, title, fmt) in zip(axes, [conf_mat, norm_mat], ['Absolute counts', 'Row-normalised (recall)'], ['d', '.2f']):
            sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax, linewidths=0.5, linecolor='lightgray')
            ax.set_xlabel('Predicted (Fuzzy FIS)')
            ax.set_ylabel('Reference (Crisp EPA)')
            ax.set_title(title)
            ax.tick_params(axis='x', labelrotation=30)
            ax.tick_params(axis='y', labelrotation=0)
        fig.subplots_adjust(left=0.14, right=0.98, hspace=0.45, top=0.9)
        if save:
            path = os.path.join(self.out, 'fig_confusion.png')
            fig.savefig(path, bbox_inches='tight')
            print(f'  Saved: {path}')
        return fig

    def plot_category_distribution(self, metrics: dict, save: bool=True):
        fis_idx = metrics['fis_idx']
        crisp_idx = metrics['crisp_idx']
        cats = ['Good', 'Moderate', 'USG', 'Unhealthy', 'V.Unhealthy']
        n = len(fis_idx)
        fis_cnt = np.array([(fis_idx == k).sum() for k in range(5)])
        crisp_cnt = np.array([(crisp_idx == k).sum() for k in range(5)])
        x = np.arange(5)
        w = 0.35
        (fig, ax) = plt.subplots(figsize=(9, 5))
        ax.set_title('Category Distribution: Mamdani FIS vs. Crisp EPA', fontweight='bold')
        b1 = ax.bar(x - w / 2, fis_cnt / n * 100, w, label='Mamdani FIS', color='#3498DB', alpha=0.85)
        b2 = ax.bar(x + w / 2, crisp_cnt / n * 100, w, label='Crisp EPA', color='#E74C3C', alpha=0.85)
        ax.bar_label(b1, fmt='%.1f%%', fontsize=8, padding=2)
        ax.bar_label(b2, fmt='%.1f%%', fontsize=8, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        ax.set_ylabel('Percentage of Test Samples (%)')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.set_ylim(0, max(fis_cnt.max(), crisp_cnt.max()) / n * 120)
        plt.tight_layout()
        if save:
            path = os.path.join(self.out, 'fig_category_dist.png')
            fig.savefig(path, bbox_inches='tight')
            print(f'  Saved: {path}')
        return fig

    def plot_category_map(self, surface_data: dict, save: bool=True):
        PM = surface_data['PM25']
        NO = surface_data['NO2']
        fuzzy_aqi = np.nan_to_num(surface_data['AQI_fuzzy'], nan=4.0)
        fuzzy_cat = np.clip(np.floor(fuzzy_aqi).astype(int), 0, 4)
        crisp_cat = np.clip(np.floor(surface_data['AQI_crisp']).astype(int), 0, 4)
        _epa_pm25_breaks = [12.0, 35.4, 55.4, 150.4]
        _epa_no2_breaks = [101.4, 191.3]
        cmap_cat = mcolors.ListedColormap(TERM_COLORS)
        levels = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
        (fig, axes) = plt.subplots(2, 1, figsize=(10, 12))
        fig.suptitle('Category Decision Maps: Mamdani FIS  vs.  Crisp EPA\n(white lines = fuzzy/crisp category boundaries  |  grey dashes = EPA breakpoints)', fontweight='bold', fontsize=12)
        titles = ['Mamdani FIS  (Fuzzy)', 'Crisp EPA Reference']
        for (ax, cat_data, title) in zip(axes, [fuzzy_cat, crisp_cat], titles):
            cf = ax.contourf(PM, NO, cat_data, levels=levels, colors=TERM_COLORS, alpha=0.82)
            ax.contour(PM, NO, cat_data, levels=[0.5, 1.5, 2.5, 3.5], colors='white', linewidths=1.8, alpha=0.9)
            for bp in _epa_pm25_breaks:
                ax.axvline(bp, color='dimgray', linestyle='--', linewidth=1.0, alpha=0.6)
            for bp in _epa_no2_breaks:
                ax.axhline(bp, color='dimgray', linestyle='--', linewidth=1.0, alpha=0.6)
            ax.set_xlabel('PM₂.₅ Concentration (μg/m³)', fontsize=10)
            ax.set_ylabel('NO₂ Concentration (μg/m³)', fontsize=10)
            ax.set_xlim([0, 250])
            ax.set_ylim([0, 200])
            ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
            ax.grid(False)
        legend_patches = [mpatches.Patch(color=TERM_COLORS[k], label=TERM_LABELS[k]) for k in range(5)]
        fig.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=9, framealpha=0.92, bbox_to_anchor=(0.5, -0.04))
        fig.subplots_adjust(left=0.1, right=0.98, hspace=0.35, bottom=0.12)
        if save:
            path = os.path.join(self.out, 'fig_category_map.png')
            fig.savefig(path, bbox_inches='tight')
            print(f'  Saved: {path}')
        return fig

    def plot_residual_distribution(self, metrics: dict, save: bool=True):
        diff = metrics['fis_vals'] - metrics['crisp_vals']
        mu = float(np.mean(diff))
        sigma = float(np.std(diff))
        (fig, ax) = plt.subplots(figsize=(10, 5))
        ax.set_title('Residual Distribution  (Fuzzy − Crisp AQI Index)\nwith Fitted Normal Curve', fontweight='bold')
        (counts, bins, patches) = ax.hist(diff, bins=35, density=True, color='steelblue', alpha=0.7, edgecolor='white', label='Observed residuals')
        x_fit = np.linspace(bins[0] - 0.2, bins[-1] + 0.2, 400)
        ax.plot(x_fit, scipy_norm.pdf(x_fit, mu, sigma), color='crimson', linewidth=2.5, label=f'Normal fit  μ = {mu:+.3f},  σ = {sigma:.3f}')
        ax.axvline(0, color='dimgray', linestyle=':', linewidth=1.2, alpha=0.7, label='Zero reference')
        ax.axvline(mu, color='navy', linestyle='-', linewidth=2.2, label=f'Mean bias = {mu:+.3f}')
        ax.axvline(mu + sigma, color='darkorange', linestyle='--', linewidth=1.5, label=f'+1σ = {mu + sigma:+.3f}')
        ax.axvline(mu - sigma, color='darkorange', linestyle='--', linewidth=1.5, label=f'−1σ = {mu - sigma:+.3f}')
        ax.axvline(mu + 2 * sigma, color='tomato', linestyle=':', linewidth=1.2, label=f'+2σ = {mu + 2 * sigma:+.3f}')
        ax.axvline(mu - 2 * sigma, color='tomato', linestyle=':', linewidth=1.2, label=f'−2σ = {mu - 2 * sigma:+.3f}')
        ax.axvspan(mu - sigma, mu + sigma, alpha=0.08, color='darkorange')
        ax.axvspan(mu - 2 * sigma, mu + 2 * sigma, alpha=0.04, color='tomato')
        ax.set_xlabel('Residual  (Fuzzy − Crisp)  AQI Category Index')
        ax.set_ylabel('Probability Density')
        ax.legend(fontsize=8, framealpha=0.92, ncol=2)
        ax.grid(True, linestyle='--', alpha=0.35)
        plt.tight_layout()
        if save:
            path = os.path.join(self.out, 'fig_residual_dist.png')
            fig.savefig(path, bbox_inches='tight')
            print(f'  Saved: {path}')
        return fig

    def plot_bland_altman(self, metrics: dict, save: bool=True):
        fv = metrics['fis_vals']
        cv = metrics['crisp_vals']
        cidx = metrics['crisp_idx']
        mean_val = (fv + cv) / 2.0
        diff_val = fv - cv
        mean_diff = float(np.mean(diff_val))
        std_diff = float(np.std(diff_val))
        loa_upper = mean_diff + 1.96 * std_diff
        loa_lower = mean_diff - 1.96 * std_diff
        (fig, ax) = plt.subplots(figsize=(10, 6))
        ax.set_title('Bland–Altman Agreement Plot\nFuzzy FIS vs. Crisp EPA  —  AQI Category Index [0–5]', fontweight='bold')
        for (k, (cat, color)) in enumerate(zip(CATEGORY_LABELS, TERM_COLORS)):
            mask = cidx == k
            if mask.sum() > 0:
                short = CATEGORY_SHORT[k].replace('\n', ' ')
                ax.scatter(mean_val[mask], diff_val[mask], color=color, label=short, alpha=0.5, s=20, zorder=3)
        ax.axhline(mean_diff, color='navy', linestyle='-', linewidth=2.2, label=f'Mean bias = {mean_diff:+.3f}', zorder=5)
        ax.axhline(loa_upper, color='crimson', linestyle='--', linewidth=1.6, label=f'+1.96 SD = {loa_upper:+.3f}', zorder=5)
        ax.axhline(loa_lower, color='crimson', linestyle='--', linewidth=1.6, label=f'−1.96 SD = {loa_lower:+.3f}', zorder=5)
        ax.axhline(0.0, color='dimgray', linestyle=':', linewidth=1.0, alpha=0.7, zorder=4)
        ax.annotate(f'{mean_diff:+.3f}', xy=(5.0, mean_diff), xytext=(4.6, mean_diff + 0.12), fontsize=9, color='navy', fontweight='bold')
        ax.annotate(f'{loa_upper:+.3f}', xy=(5.0, loa_upper), xytext=(4.6, loa_upper + 0.12), fontsize=8, color='crimson')
        ax.annotate(f'{loa_lower:+.3f}', xy=(5.0, loa_lower), xytext=(4.6, loa_lower - 0.22), fontsize=8, color='crimson')
        ax.set_xlabel('Mean of Fuzzy and Crisp AQI Index  [(F + C) / 2]')
        ax.set_ylabel('Difference  (Fuzzy − Crisp)  AQI Index')
        ax.legend(loc='upper left', fontsize=8, framealpha=0.92)
        ax.grid(True, linestyle='--', alpha=0.35)
        plt.tight_layout()
        if save:
            path = os.path.join(self.out, 'fig_bland_altman.png')
            fig.savefig(path, bbox_inches='tight')
            print(f'  Saved: {path}')
        return fig

    def plot_all(self, fis, metrics: dict, surface_data: dict):
        print('\n─── Generating figures ───────────────────────────────────────')
        self.plot_input_mfs(fis)
        self.plot_output_mf(fis)
        self.plot_rule_table(fis)
        self.plot_surface_3d(surface_data, method='fuzzy')
        self.plot_surface_3d(surface_data, method='crisp')
        self.plot_surface_comparison(surface_data)
        self.plot_surface_diff(surface_data)
        self.plot_scatter(metrics)
        self.plot_confusion_matrix(metrics)
        self.plot_category_distribution(metrics)
        self.plot_bland_altman(metrics)
        self.plot_residual_distribution(metrics)
        self.plot_category_map(surface_data)
        print('─── All figures saved ────────────────────────────────────────\n')
