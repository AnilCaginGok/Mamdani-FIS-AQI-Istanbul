from __future__ import annotations
import os
from typing import Dict, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .dataset import AirQualityDataset
from .evaluation import Evaluator, compute_pair_metrics, CATEGORY_SHORT
from .sources.base import COL_DATETIME, COL_NO2

class RealDataEvaluator:

    def __init__(self, fis, crisp, dataset: AirQualityDataset, output_dir: str='results'):
        self.fis = fis
        self.crisp = crisp
        self.dataset = dataset
        self.output_dir = output_dir
        self.metrics: Dict[str, dict] = {}
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_splits(self, verbose: bool=True) -> Dict[str, dict]:
        for (split_name, frame) in self.dataset.splits.items():
            if len(frame) == 0:
                print(f'  [{split_name}] skipped (empty)')
                continue
            if verbose:
                print(f"\n  Evaluating split '{split_name}' ({len(frame):,} samples)…")
            pcol = self.dataset.particulate_col
            show_prog = verbose and len(frame) > 5000
            m = compute_pair_metrics(self.fis, self.crisp, frame[pcol].values, frame['no2'].values, verbose=verbose, show_progress=show_prog)
            self.metrics[split_name] = m
        return self.metrics

    def save_results_csv(self, split: str='test') -> str:
        frame = self.dataset.splits[split]
        pcol = self.dataset.particulate_col
        p_vals = frame[pcol].values
        no2 = frame['no2'].values
        (fis_vals, fis_cats) = self.fis.evaluate_batch(p_vals, no2)
        (crisp_vals, crisp_cats) = self.crisp.evaluate_batch(p_vals, no2)
        out = frame.copy()
        out['fuzzy_value'] = fis_vals
        out['fuzzy_category'] = fis_cats
        out['crisp_value'] = crisp_vals
        out['crisp_category'] = crisp_cats
        out['match'] = [f == c for (f, c) in zip(fis_cats, crisp_cats)]
        path = os.path.join(self.output_dir, f'istanbul_{split}_predictions.csv')
        out.to_csv(path, index=False)
        return path

    def plot_real_data_summary(self, split: str='test') -> None:
        if split not in self.metrics:
            return
        m = self.metrics[split]
        (fig, ax) = plt.subplots(figsize=(7, 6))
        ax.scatter(m['crisp_vals'], m['fis_vals'], alpha=0.35, s=12, c='#2980B9')
        lims = [0, 5.2]
        ax.plot(lims, lims, 'k--', lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Crisp EPA index')
        ax.set_ylabel('Fuzzy AQI index')
        ax.set_title(f"Istanbul real data — scatter ({split}, n={m['n_valid']:,})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f'fig_real_scatter_{split}.png'))
        plt.close(fig)
        (fig, ax) = plt.subplots(figsize=(8, 6))
        sns.heatmap(m['conf_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=CATEGORY_SHORT, yticklabels=CATEGORY_SHORT, ax=ax)
        ax.set_xlabel('Fuzzy predicted')
        ax.set_ylabel('Crisp EPA (reference)')
        ax.set_title(f'Istanbul real data — confusion ({split})')
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f'fig_real_confusion_{split}.png'))
        plt.close(fig)

    def print_summary_table(self) -> None:
        print('\n' + '=' * 72)
        print('REAL DATA — SPLIT SUMMARY (Fuzzy vs Crisp EPA)')
        print('=' * 72)
        print(f"  {'Split':<8} {'N':>8} {'Accuracy':>10} {'Kappa':>8} {'MacroF1':>8} {'RMSE':>8}")
        print('  ' + '-' * 68)
        for (name, m) in self.metrics.items():
            mf1 = m.get('macro_f1', 0.0)
            print(f"  {name:<8} {m['n_valid']:>8,} {m['accuracy'] * 100:>9.2f}% {m['kappa']:>8.4f} {mf1:>8.4f} {m['rmse']:>8.4f}")
        print()

    def run_zone_analysis(self, split: str='test') -> dict:
        if split not in self.metrics:
            return {}
        ev = Evaluator(self.fis, self.crisp)
        return ev.run_zone_analysis(self.metrics[split])
