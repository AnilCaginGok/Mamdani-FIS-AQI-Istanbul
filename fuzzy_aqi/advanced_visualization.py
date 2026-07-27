import os
from math import pi

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .fuzzy_system import TERM_KEYS, TERM_LABELS, TERM_COLORS
from .evaluation import CATEGORY_LABELS, CATEGORY_SHORT
from .seasonal_analysis import SEASON_COLORS, SEASON_ORDER

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 11, "legend.fontsize": 9, "lines.linewidth": 2.0,
})

_METHOD_PALETTE = [
    "#2ECC71", "#3498DB", "#9B59B6", "#E74C3C", "#F39C12",
    "#1ABC9C", "#E67E22", "#34495E", "#E91E63",
]


class AdvancedVisualizer:

    def __init__(self, output_dir: str = "results") -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.out = output_dir

    def _save(self, fig: plt.Figure, filename: str) -> None:
        path = os.path.join(self.out, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ------------------------------------------------------------------
    def plot_method_comparison_bars(self, fis_metrics: dict, ml_results: dict, save: bool = True) -> plt.Figure:
        methods = ["Mamdani FIS"] + list(ml_results.keys())
        data = {
            "Accuracy":   [fis_metrics["accuracy"]]               + [ml_results[m]["accuracy"]  for m in ml_results],
            "Cohen's κ":  [fis_metrics["kappa"]]                  + [ml_results[m]["kappa"]     for m in ml_results],
            "Macro-F1":   [fis_metrics.get("macro_f1", 0)]        + [ml_results[m]["macro_f1"]  for m in ml_results],
            "MCC":        [fis_metrics["mcc"]]                    + [ml_results[m]["mcc"]       for m in ml_results],
        }
        n_methods = len(methods)
        x = np.arange(n_methods)
        width = 0.2
        fig, ax = plt.subplots(figsize=(max(14, n_methods * 1.3), 6))
        fig.suptitle(
            "Method Comparison: Mamdani FIS vs Machine-Learning Classifiers\n"
            "(higher = better for all metrics)",
            fontweight="bold",
        )
        colors_m = ["#2ECC71", "#3498DB", "#E74C3C", "#9B59B6"]
        for k, (mname, color) in enumerate(zip(data.keys(), colors_m)):
            vals = data[mname]
            bars = ax.bar(x + (k - 1.5) * width, vals, width, label=mname, color=color, alpha=0.85, zorder=3)
            for bar, val in zip(bars, vals):
                if val > 0.05:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{val:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
        ax.axvline(0.5, color="dimgray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Metric Value")
        ax.set_ylim(0, 1.15)
        ax.legend(loc="upper right", framealpha=0.9)
        ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        plt.tight_layout()
        if save:
            self._save(fig, "fig_method_comparison_bars.png")
        return fig

    # ------------------------------------------------------------------
    def plot_radar_comparison(self, fis_metrics: dict, ml_results: dict, top_n: int = 5, save: bool = True) -> plt.Figure:
        metric_keys   = ["accuracy", "kappa", "macro_f1", "mcc"]
        metric_labels = ["Accuracy", "Cohen's κ", "Macro-F1", "MCC"]
        N      = len(metric_keys)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        sorted_ml    = sorted(ml_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)[:top_n]
        plot_methods = [("Mamdani FIS", fis_metrics)] + list(sorted_ml)

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.set_title(
            f"Performance Radar: Mamdani FIS vs Top-{top_n} ML Methods\n(outer edge = 1.0)",
            fontweight="bold", pad=20,
        )
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
        ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.6)

        palette = ["#E74C3C"] + _METHOD_PALETTE[:top_n]
        for (name, m), color in zip(plot_methods, palette):
            vals  = [max(0.0, float(m.get(k, 0))) for k in metric_keys]
            vals += vals[:1]
            lw    = 3.0 if "FIS" in name else 1.8
            ax.plot(angles, vals, linewidth=lw, linestyle="solid", label=name, color=color)
            ax.fill(angles, vals, color=color, alpha=0.07)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8, framealpha=0.9)
        plt.tight_layout()
        if save:
            self._save(fig, "fig_radar_comparison.png")
        return fig

    # ------------------------------------------------------------------
    def plot_mf_type_comparison(self, tri_fis, gauss_fis, pollutant: str = "pm10", save: bool = True) -> plt.Figure:
        tri_mfs   = tri_fis.get_mf_values()
        gauss_mfs = gauss_fis.get_mf_values()
        p_key = "pm10" if pollutant == "pm10" else "pm25"
        xlabels = {
            "pm10": "PM₁₀ Concentration (µg/m³)",
            "pm25": "PM₂.₅ Concentration (µg/m³)",
            "no2":  "NO₂ Concentration (µg/m³)",
        }
        fig, axes = plt.subplots(4, 1, figsize=(10, 16))
        fig.suptitle(
            "Membership Function Comparison:\nTriangular/Trapezoidal (solid) vs Gaussian (dashed)",
            fontweight="bold", fontsize=13,
        )
        legend_handles = [mpatches.Patch(color=c, label=TERM_LABELS[i]) for i, c in enumerate(TERM_COLORS)]
        for row, (key, xlabel) in enumerate([(p_key, xlabels[p_key]), ("no2", xlabels["no2"])]):
            tri   = tri_mfs[key]
            gauss = gauss_mfs[key]
            ax_tri  = axes[row * 2]
            ax_both = axes[row * 2 + 1]
            for term, color in zip(TERM_KEYS, TERM_COLORS):
                label = TERM_LABELS[TERM_KEYS.index(term)]
                ax_tri.plot(tri["universe"], tri["mfs"][term], color=color, label=label)
                ax_tri.fill_between(tri["universe"], tri["mfs"][term], alpha=0.1, color=color)
                ax_both.plot(tri["universe"],   tri["mfs"][term],   color=color, linestyle="-",  linewidth=2.0)
                ax_both.plot(gauss["universe"], gauss["mfs"][term], color=color, linestyle="--", linewidth=1.8, alpha=0.85)
            for ax in [ax_tri, ax_both]:
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Membership µ(x)")
                ax.set_ylim(-0.05, 1.15)
                ax.grid(True, linestyle="--", alpha=0.35)
            ax_tri.set_title(f"{key.upper()} — Triangular/Trapezoidal MFs", fontweight="bold")
            ax_both.set_title(f"{key.upper()} — Overlay Comparison", fontweight="bold")
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=5,
            fontsize=8,
            framealpha=0.92,
            bbox_to_anchor=(0.5, -0.01),
            title="Category (solid = Tri/Trap, dashed = Gaussian)",
            title_fontsize=8,
        )
        plt.tight_layout(rect=[0, 0.04, 1, 0.97])
        if save:
            self._save(fig, "fig_mf_type_comparison.png")
        return fig

    # ------------------------------------------------------------------
    def plot_defuzz_comparison(self, defuzz_results: dict, save: bool = True) -> plt.Figure:
        labels = [v["label"] for v in defuzz_results.values()]
        accs   = [v["accuracy"] * 100 for v in defuzz_results.values()]
        rmses  = [v["rmse"]  for v in defuzz_results.values()]
        x = np.arange(len(labels))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))
        fig.suptitle("Defuzzification Method Comparison (same FIS rule base, triangular MFs)", fontweight="bold")
        colors_d = ["#2ECC71", "#3498DB", "#E74C3C", "#F39C12", "#9B59B6"]
        bars1 = ax1.bar(x, accs,  color=colors_d, alpha=0.85, zorder=3)
        bars2 = ax2.bar(x, rmses, color=colors_d, alpha=0.85, zorder=3)
        ax1.bar_label(bars1, fmt="%.2f%%", padding=3, fontsize=9)
        ax2.bar_label(bars2, fmt="%.4f",   padding=3, fontsize=9)
        for ax, data, ylabel, title in [
            (ax1, accs,  "Accuracy (%)",            "Accuracy vs Defuzzification Method"),
            (ax2, rmses, "RMSE (category index)",   "RMSE vs Defuzzification Method"),
        ]:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, max(data) * 1.15)
            ax.set_title(title)
            ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        plt.tight_layout()
        if save:
            self._save(fig, "fig_defuzz_comparison.png")
        return fig

    # ------------------------------------------------------------------
    def plot_seasonal_analysis(self, seasonal_df: pd.DataFrame, save: bool = True) -> plt.Figure:
        if seasonal_df.empty:
            return None
        seasons = seasonal_df["season"].tolist()
        cols    = ["accuracy", "kappa", "macro_f1", "rmse"]
        titles  = ["Accuracy", "Cohen's κ", "Macro-F1", "RMSE"]
        ylabels = ["Accuracy (%)", "Cohen's κ", "Macro-F1", "RMSE"]
        fig, axes = plt.subplots(4, 1, figsize=(8, 14))
        fig.suptitle("Seasonal Performance of Mamdani FIS — Istanbul IBB Data", fontweight="bold")
        colors = [SEASON_COLORS.get(s, "#95A5A6") for s in seasons]
        for ax, col, title, ylabel in zip(axes, cols, titles, ylabels):
            vals = seasonal_df[col].values.copy()
            if col == "accuracy":
                vals = vals * 100
            bars = ax.bar(seasons, vals, color=colors, alpha=0.85, zorder=3)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(vals) * 0.01,
                        f"{val:.2f}" + ("%" if col == "accuracy" else ""),
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
            n_vals = seasonal_df["n"].tolist()
            ax.set_xticks(range(len(seasons)))
            ax.set_xticklabels([f"{s}\n(n={n:,})" for s, n in zip(seasons, n_vals)], fontsize=8)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontweight="bold")
            ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
            ax.set_ylim(0, max(vals) * 1.18)
        plt.tight_layout()
        if save:
            self._save(fig, "fig_seasonal_analysis.png")
        return fig

    # ------------------------------------------------------------------
    def plot_station_accuracy(self, station_df: pd.DataFrame, top_n: int = 15, save: bool = True) -> plt.Figure:
        if station_df.empty:
            return None
        df = station_df.head(top_n).copy()
        df["accuracy_pct"] = df["accuracy"] * 100
        fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.45)))
        ax.set_title(
            "Per-Station Mamdani FIS Accuracy — Istanbul IBB\n"
            "(colour = dominant AQI category)",
            fontweight="bold",
        )
        cat_color_map = {cat: col for cat, col in zip(CATEGORY_LABELS, TERM_COLORS)}
        colors = [cat_color_map.get(str(c), "#95A5A6") for c in df["dominant_crisp_cat"]]
        bars = ax.barh(range(len(df)), df["accuracy_pct"], color=colors, alpha=0.85, zorder=3)
        for bar, val in zip(bars, df["accuracy_pct"]):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df["station_id"].astype(str), fontsize=8)
        ax.set_xlabel("Accuracy (%)")
        ax.set_xlim(0, 105)
        ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=0)
        legend_patches = [mpatches.Patch(color=col, label=lab) for col, lab in zip(TERM_COLORS, CATEGORY_LABELS)]
        ax.legend(handles=legend_patches, title="Dominant category", loc="lower right", fontsize=8, framealpha=0.9)
        plt.tight_layout()
        if save:
            self._save(fig, "fig_station_accuracy.png")
        return fig

    # ------------------------------------------------------------------
    def plot_extended_benchmark(self, bench_df: pd.DataFrame, save: bool = True) -> plt.Figure:
        fuzzy_correct = int(bench_df["Fuzzy Correct"].sum())
        crisp_correct = int(bench_df["Crisp Correct"].sum())
        total = len(bench_df)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))
        fig.suptitle(
            f"Extended Benchmark ({total} hand-crafted test cases)\nMamdani FIS vs Crisp EPA Reference",
            fontweight="bold",
        )
        methods = ["Mamdani FIS", "Crisp EPA"]
        accs    = [fuzzy_correct / total * 100, crisp_correct / total * 100]
        bars = ax1.bar(methods, accs, color=["#2ECC71", "#E74C3C"], alpha=0.85, zorder=3)
        ax1.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=11, fontweight="bold")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_ylim(0, 115)
        ax1.set_title("Overall Accuracy on Extended Benchmark")
        ax1.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        if "Group" in bench_df.columns:
            groups = bench_df.groupby("Group").agg(
                fuzzy_acc=("Fuzzy Correct", "mean"),
                crisp_acc=("Crisp Correct", "mean"),
                n=("Fuzzy Correct", "count"),
            ).reset_index()
            x, w = np.arange(len(groups)), 0.35
            b1 = ax2.bar(x - w / 2, groups["fuzzy_acc"] * 100, w, label="Mamdani FIS", color="#2ECC71", alpha=0.85)
            b2 = ax2.bar(x + w / 2, groups["crisp_acc"] * 100, w, label="Crisp EPA",   color="#E74C3C", alpha=0.85)
            ax2.bar_label(b1, fmt="%.0f%%", fontsize=8, padding=2)
            ax2.bar_label(b2, fmt="%.0f%%", fontsize=8, padding=2)
            ax2.set_xticks(x)
            ax2.set_xticklabels(groups["Group"], fontsize=9)
            ax2.set_ylabel("Group Accuracy (%)")
            ax2.set_ylim(0, 120)
            ax2.set_title("Accuracy by Case Group")
            ax2.legend()
            ax2.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
        else:
            ax2.axis("off")
        plt.tight_layout()
        if save:
            self._save(fig, "fig_extended_benchmark.png")
        return fig

    # ------------------------------------------------------------------
    def plot_yearly_trend(self, yearly_df: pd.DataFrame, save: bool = True) -> plt.Figure:
        if yearly_df.empty:
            return None
        years  = yearly_df["year"].astype(str).tolist()
        accs   = yearly_df["accuracy"].values * 100
        kappas = yearly_df["kappa"].values
        x = np.arange(len(years))
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()
        fig.suptitle("Year-over-Year Mamdani FIS Performance — Istanbul IBB", fontweight="bold")
        bars = ax1.bar(x, accs, color="#3498DB", alpha=0.6, label="Accuracy (%)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(years, fontsize=10)
        ax1.set_ylabel("Accuracy (%)", color="#3498DB")
        ax1.set_ylim(0, max(accs) * 1.2)
        ax1.tick_params(axis="y", labelcolor="#3498DB")
        ax2.plot(x, kappas, "o-", color="#E74C3C", linewidth=2.5, markersize=7, label="Cohen's κ", zorder=5)
        ax2.set_ylabel("Cohen's κ", color="#E74C3C")
        ax2.tick_params(axis="y", labelcolor="#E74C3C")
        ax2.set_ylim(0, 1.0)
        for bar, val in zip(bars, accs):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)
        ax1.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
        ax1.set_xlabel("Year")
        plt.tight_layout()
        if save:
            self._save(fig, "fig_yearly_trend.png")
        return fig

    # ------------------------------------------------------------------
    def plot_per_class_heatmap(self, fis_metrics: dict, ml_results: dict, metric: str = "recall", save: bool = True) -> plt.Figure:
        from sklearn.metrics import confusion_matrix as _cm

        methods   = ["Mamdani FIS"] + list(ml_results.keys())
        cat_names = ["Good", "Mod.", "USG", "Unhlt.", "V.Unhlt."]
        matrix    = np.zeros((len(methods), 5), dtype=float)

        def _per_class(ref, pred, m):
            cm = _cm(ref, pred, labels=list(range(5)))
            if m == "recall":
                row_sums = cm.sum(axis=1)
                return np.where(row_sums > 0, np.diag(cm) / row_sums, 0.0)
            col_sums = cm.sum(axis=0)
            return np.where(col_sums > 0, np.diag(cm) / col_sums, 0.0)

        for k, name in enumerate(methods):
            if name == "Mamdani FIS":
                if "conf_matrix" in fis_metrics:
                    cm = fis_metrics["conf_matrix"]
                    if metric == "recall":
                        row_sums = cm.sum(axis=1)
                        matrix[k] = np.where(row_sums > 0, np.diag(cm) / row_sums, 0.0)
                    else:
                        col_sums = cm.sum(axis=0)
                        matrix[k] = np.where(col_sums > 0, np.diag(cm) / col_sums, 0.0)
            else:
                ref  = ml_results[name].get("_ref")
                pred = ml_results[name]["pred"]
                if ref is None:
                    continue
                matrix[k] = _per_class(ref, pred, metric)

        title_metric = "Recall" if metric == "recall" else "Precision"
        fig, ax = plt.subplots(figsize=(10, max(5, len(methods) * 0.55)))
        ax.set_title(
            f"Per-Class {title_metric} Heat-Map — All Methods vs Crisp EPA\n"
            "(rows = methods, cols = AQI categories)",
            fontweight="bold",
        )
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="YlGn", vmin=0, vmax=1,
                    xticklabels=cat_names, yticklabels=methods, ax=ax,
                    linewidths=0.5, linecolor="lightgray",
                    cbar_kws={"label": title_metric})
        ax.set_xlabel("AQI Category")
        ax.set_ylabel("Method")
        ax.tick_params(axis="x", labelrotation=15, labelsize=9)
        ax.tick_params(axis="y", labelrotation=0,  labelsize=8)
        plt.tight_layout()
        if save:
            fn = f"fig_per_class_{metric}_heatmap.png"
            self._save(fig, fn)
        return fig

    # ------------------------------------------------------------------
    def plot_mf_type_scatter(self, tri_metrics: dict, gauss_metrics: dict, save: bool = True) -> plt.Figure:
        fig, axes = plt.subplots(2, 1, figsize=(8, 10))
        fig.suptitle(
            "AQI Index Scatter: Triangular MF FIS (left) vs Gaussian MF FIS (right)\nvs Crisp EPA reference",
            fontweight="bold",
        )
        for ax, m, title in zip(axes, [tri_metrics, gauss_metrics], ["Triangular/Trapezoidal MFs", "Gaussian MFs"]):
            cidx = m["crisp_idx"]
            for k, (cat, color) in enumerate(zip(CATEGORY_LABELS, TERM_COLORS)):
                mask = cidx == k
                if mask.sum() > 0:
                    ax.scatter(m["crisp_vals"][mask], m["fis_vals"][mask], color=color,
                               label=CATEGORY_SHORT[k].replace("\n", " "), alpha=0.45, s=15)
            lims = [0, 5]
            ax.plot(lims, lims, "k--", linewidth=1.5, label="Perfect agreement")
            ax.set_xlabel("Crisp EPA AQI Index (reference)")
            ax.set_ylabel("FIS AQI Index (predicted)")
            acc, kappa = m["accuracy"] * 100, m["kappa"]
            ax.set_title(f"{title}\nAcc={acc:.1f}%  κ={kappa:.4f}", fontweight="bold")
            ax.set_xlim(0, 5)
            ax.set_ylim(0, 5)
            ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
            ax.set_xticklabels(["Good", "Mod.", "USG", "Unhlt.", "VU"], fontsize=8)
            ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
            ax.set_yticklabels(["Good", "Mod.", "USG", "Unhlt.", "VU"], fontsize=8)
            ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
            ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        if save:
            self._save(fig, "fig_mf_type_scatter.png")
        return fig
