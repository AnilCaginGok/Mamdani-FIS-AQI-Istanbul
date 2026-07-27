import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from .console import safe_print_df
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, mean_absolute_error, cohen_kappa_score, matthews_corrcoef, f1_score
CATEGORY_LABELS = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy']
CATEGORY_SHORT = ['Good', 'Moderate', 'USG', 'Unhealthy', 'Very\nUnhealthy']

def _label_to_idx(label: str) -> int:
    try:
        return CATEGORY_LABELS.index(label)
    except ValueError:
        for (i, l) in enumerate(CATEGORY_LABELS):
            if label.lower() in l.lower() or l.lower() in label.lower():
                return i
        return -1

def generate_random_samples(n: int=500, seed: int=42) -> tuple:
    rng = np.random.default_rng(seed)
    pm25 = rng.uniform(0, 250, n)
    no2 = rng.uniform(0, 200, n)
    return (pm25, no2)
BENCHMARK_CASES = [(5.0, 10.0, 'Pure Good (both pollutants low)', 0), (18.0, 60.0, 'Pure Moderate (both moderate)', 1), (45.0, 100.0, 'Pure USG (both USG)', 2), (90.0, 145.0, 'Pure Unhealthy (both unhealthy)', 3), (160.0, 185.0, 'Pure Very Unhealthy (both VU)', 4), (5.0, 145.0, 'Mixed: Good PM2.5, Unhealthy NO₂', 3), (18.0, 100.0, 'Mixed: Moderate PM2.5, USG NO₂', 2), (45.0, 10.0, 'Mixed: USG PM2.5, Good NO₂', 2), (90.0, 60.0, 'Mixed: Unhealthy PM2.5, Moderate NO₂', 3), (12.0, 40.0, 'Boundary: Good/Moderate crossover', 1), (35.0, 90.0, 'Boundary: Moderate/USG crossover', 2), (65.0, 130.0, 'Boundary: USG/Unhealthy crossover', 3), (150.0, 180.0, 'Boundary: Unhealthy/VU crossover', 4)]

EXTENDED_BENCHMARK_CASES = [
    # --- Extreme values ---
    (0.0,   0.0,   'Extreme: perfect clean air (zero emissions)', 0),
    (250.0, 200.0, 'Extreme: maximum pollution (both at ceiling)', 4),
    (1.0,   5.0,   'Extreme: near-zero pollution, rural background', 0),
    (200.0, 5.0,   'Extreme: VU PM2.5, clean NO₂ → dominated by PM2.5', 4),
    (2.0,   190.0, 'Extreme: Good PM2.5, VU NO₂ → dominated by NO₂', 4),
    # --- Exact EPA breakpoints ---
    (12.0,  5.0,   'EPA boundary: PM2.5 at Good/Moderate edge', 1),
    (35.4,  5.0,   'EPA boundary: PM2.5 at Moderate/USG edge', 1),
    (55.4,  5.0,   'EPA boundary: PM2.5 at USG/Unhealthy edge', 2),
    (150.4, 5.0,   'EPA boundary: PM2.5 at Unhealthy/VU edge', 3),
    (5.0,   101.4, 'EPA boundary: NO₂ at Good/Moderate edge', 1),
    # --- Asymmetric mix: large severity gap ---
    (5.0,   185.0, 'Asymmetric: Good PM2.5 + near-VU NO₂ → VU', 4),
    (180.0, 10.0,  'Asymmetric: VU PM2.5 + Good NO₂ → VU', 4),
    (8.0,   130.0, 'Asymmetric: Good PM2.5 + Unhealthy NO₂ → Unhealthy', 3),
    (120.0, 20.0,  'Asymmetric: Unhealthy PM2.5 + Good NO₂ → Unhealthy', 3),
    # --- Realistic urban scenarios ---
    (22.0,  70.0,  'Urban: typical moderate city day', 1),
    (40.0,  85.0,  'Urban: heavy traffic corridor (USG)', 2),
    (85.0,  155.0, 'Urban: industrial district (Unhealthy)', 3),
]

ALL_BENCHMARK_CASES = BENCHMARK_CASES + [
    (pm25, no2, desc, cat) for (pm25, no2, desc, cat) in EXTENDED_BENCHMARK_CASES
]

BENCHMARK_GROUPS = (
    ['Pure'] * 5 + ['Mixed'] * 4 + ['Boundary'] * 4 +
    ['Extreme'] * 5 + ['EPA-Boundary'] * 5 + ['Asymmetric'] * 4 + ['Urban'] * 3
)

class Evaluator:

    def __init__(self, fis, crisp):
        self.fis = fis
        self.crisp = crisp
        self.results_df = None

    def run_extended_benchmark(self, verbose: bool = True) -> pd.DataFrame:
        rows = []
        for idx, (pm25, no2, desc, ref_cat_idx) in enumerate(ALL_BENCHMARK_CASES):
            (fis_val, fis_cat) = self.fis.evaluate(pm25, no2)
            (crisp_val, crisp_cat) = self.crisp.evaluate(pm25, no2)
            fis_idx = _label_to_idx(fis_cat)
            crisp_idx_val = _label_to_idx(crisp_cat)
            group = BENCHMARK_GROUPS[idx] if idx < len(BENCHMARK_GROUPS) else 'Other'
            rows.append({
                'PM2.5 (μg/m³)': pm25, 'NO₂ (μg/m³)': no2,
                'Description': desc, 'Group': group,
                'Expected': CATEGORY_LABELS[ref_cat_idx],
                'Fuzzy Value': round(fis_val, 3), 'Fuzzy Category': fis_cat,
                'Crisp Value': round(crisp_val, 3), 'Crisp Category': crisp_cat,
                'Fuzzy Correct': fis_idx == ref_cat_idx,
                'Crisp Correct': crisp_idx_val == ref_cat_idx,
            })
        df = pd.DataFrame(rows)
        if verbose:
            print('\n' + '=' * 90)
            print('EXTENDED BENCHMARK — 30 TEST CASES — MAMDANI FIS vs CRISP EPA')
            print('=' * 90)
            cols = ['Group', 'PM2.5 (μg/m³)', 'NO₂ (μg/m³)', 'Expected',
                    'Fuzzy Category', 'Crisp Category', 'Fuzzy Correct']
            safe_print_df(df, cols)
            print(f'\nFuzzy overall accuracy : {df["Fuzzy Correct"].mean() * 100:.1f}%  '
                  f'({df["Fuzzy Correct"].sum()}/{len(df)})')
            print(f'Crisp overall accuracy : {df["Crisp Correct"].mean() * 100:.1f}%  '
                  f'({df["Crisp Correct"].sum()}/{len(df)})')
            print('\nPer-group summary:')
            for grp in df['Group'].unique():
                sub = df[df['Group'] == grp]
                print(f'  {grp:<15}: Fuzzy {sub["Fuzzy Correct"].mean()*100:.0f}%  '
                      f'Crisp {sub["Crisp Correct"].mean()*100:.0f}%  (n={len(sub)})')
        return df

    def run_benchmark(self, verbose: bool=True) -> pd.DataFrame:
        rows = []
        for (pm25, no2, desc, ref_cat_idx) in BENCHMARK_CASES:
            (fis_val, fis_cat) = self.fis.evaluate(pm25, no2)
            (crisp_val, crisp_cat) = self.crisp.evaluate(pm25, no2)
            fis_idx = _label_to_idx(fis_cat)
            crisp_idx = _label_to_idx(crisp_cat)
            rows.append({'PM2.5 (μg/m³)': pm25, 'NO₂ (μg/m³)': no2, 'Description': desc, 'Expected': CATEGORY_LABELS[ref_cat_idx], 'Fuzzy Value': round(fis_val, 3), 'Fuzzy Category': fis_cat, 'Crisp Value': round(crisp_val, 3), 'Crisp Category': crisp_cat, 'Fuzzy Correct': fis_idx == ref_cat_idx, 'Crisp Correct': crisp_idx == ref_cat_idx})
        df = pd.DataFrame(rows)
        if verbose:
            self._print_benchmark(df)
        return df

    def run_statistical_evaluation(self, n_samples: int=500, seed: int=42, verbose: bool=True) -> dict:
        (pm25, no2) = generate_random_samples(n_samples, seed)
        print(f'\nEvaluating Mamdani FIS on {n_samples} random samples (seed={seed})…')
        (fis_vals, fis_cats) = self.fis.evaluate_batch(pm25, no2)
        (crisp_vals, crisp_cats) = self.crisp.evaluate_batch(pm25, no2)
        fis_idx = np.array([_label_to_idx(c) for c in fis_cats])
        crisp_idx = np.array([_label_to_idx(c) for c in crisp_cats])
        valid = ~(np.isnan(fis_vals) | np.isnan(crisp_vals))
        fis_vals_v = fis_vals[valid]
        crisp_vals_v = crisp_vals[valid]
        fis_idx_v = fis_idx[valid]
        crisp_idx_v = crisp_idx[valid]
        accuracy = np.mean(fis_idx_v == crisp_idx_v)
        rmse = np.sqrt(mean_squared_error(crisp_vals_v, fis_vals_v))
        mae = mean_absolute_error(crisp_vals_v, fis_vals_v)
        kappa = cohen_kappa_score(crisp_idx_v, fis_idx_v)
        macro_f1 = float(f1_score(crisp_idx_v, fis_idx_v, average='macro', labels=list(range(5)), zero_division=0))
        mcc = matthews_corrcoef(crisp_idx_v, fis_idx_v)
        (r_val, p_val) = pearsonr(crisp_vals_v, fis_vals_v)
        bias = float(np.mean(fis_vals_v - crisp_vals_v))
        std_err = float(np.std(fis_vals_v - crisp_vals_v))
        n_ci = int(valid.sum())
        p_hat = float(accuracy)
        z_ci = 1.96
        denom = 1.0 + z_ci ** 2 / n_ci
        center = (p_hat + z_ci ** 2 / (2 * n_ci)) / denom
        margin = z_ci * np.sqrt(p_hat * (1 - p_hat) / n_ci + z_ci ** 2 / (4 * n_ci ** 2)) / denom
        ci_low = float(np.clip(center - margin, 0.0, 1.0))
        ci_high = float(np.clip(center + margin, 0.0, 1.0))
        conf_mat = confusion_matrix(crisp_idx_v, fis_idx_v, labels=list(range(5)))
        report = classification_report(crisp_idx_v, fis_idx_v, labels=list(range(5)), target_names=CATEGORY_SHORT, output_dict=True, zero_division=0)
        self.results_df = pd.DataFrame({'PM2.5': pm25, 'NO2': no2, 'Fuzzy_Value': fis_vals, 'Fuzzy_Cat_Idx': fis_idx, 'Crisp_Value': crisp_vals, 'Crisp_Cat_Idx': crisp_idx, 'Match': fis_idx == crisp_idx})
        metrics = {'n_valid': int(valid.sum()), 'accuracy': float(accuracy), 'ci_low': ci_low, 'ci_high': ci_high, 'rmse': float(rmse), 'mae': float(mae), 'kappa': float(kappa), 'macro_f1': macro_f1, 'mcc': float(mcc), 'pearson_r': float(r_val), 'pearson_p': float(p_val), 'r2': float(r_val ** 2), 'bias': bias, 'std_error': std_err, 'conf_matrix': conf_mat, 'cls_report': report, 'fis_vals': fis_vals_v, 'crisp_vals': crisp_vals_v, 'fis_idx': fis_idx_v, 'crisp_idx': crisp_idx_v, 'pm25': pm25[valid], 'no2': no2[valid]}
        if verbose:
            self._print_statistics(metrics)
        return metrics

    def run_zone_analysis(self, metrics: dict) -> dict:
        fis_vals = metrics['fis_vals']
        crisp_vals = metrics['crisp_vals']
        fis_idx = metrics['fis_idx']
        crisp_idx = metrics['crisp_idx']
        zones = {'Low   (Good + Moderate)': crisp_idx <= 1, 'Medium (USG)           ': crisp_idx == 2, 'High   (Unhealthy + VU)': crisp_idx >= 3}
        results = {}
        for (name, mask) in zones.items():
            n = int(mask.sum())
            if n > 0:
                acc = float(np.mean(fis_idx[mask] == crisp_idx[mask]))
                bias = float(np.mean(fis_vals[mask] - crisp_vals[mask]))
                rmse = float(np.sqrt(np.mean((fis_vals[mask] - crisp_vals[mask]) ** 2)))
                results[name] = {'n': n, 'accuracy': acc, 'bias': bias, 'rmse': rmse}
        return results

    def compute_surface(self, n_pts: int=40) -> dict:
        pm25_vals = np.linspace(0, 250, n_pts)
        no2_vals = np.linspace(0, 200, n_pts)
        (PM, NO) = np.meshgrid(pm25_vals, no2_vals)
        AQI_fuzzy = np.zeros_like(PM)
        AQI_crisp = np.zeros_like(PM)
        print(f'\nComputing response surface ({n_pts}×{n_pts} = {n_pts ** 2} points)…')
        total = n_pts * n_pts
        count = 0
        for i in range(n_pts):
            for j in range(n_pts):
                try:
                    (AQI_fuzzy[i, j], _) = self.fis.evaluate(PM[i, j], NO[i, j])
                except Exception:
                    AQI_fuzzy[i, j] = np.nan
                (AQI_crisp[i, j], _) = self.crisp.evaluate(PM[i, j], NO[i, j])
                count += 1
            pct = count / total * 100
            if (i + 1) % 10 == 0 or i == n_pts - 1:
                print(f'  {pct:5.1f}% done')
        return {'PM25': PM, 'NO2': NO, 'AQI_fuzzy': AQI_fuzzy, 'AQI_crisp': AQI_crisp}

    @staticmethod
    def _print_benchmark(df: pd.DataFrame):
        print('\n' + '=' * 80)
        print('BENCHMARK TEST CASES — MAMDANI FIS vs. CRISP EPA')
        print('=' * 80)
        cols = ['PM2.5 (μg/m³)', 'NO₂ (μg/m³)', 'Expected', 'Fuzzy Category', 'Crisp Category', 'Fuzzy Correct']
        safe_print_df(df, cols)
        fuzzy_acc = df['Fuzzy Correct'].mean()
        crisp_acc = df['Crisp Correct'].mean()
        print(f'\nFuzzy Accuracy on benchmark : {fuzzy_acc * 100:.1f}%')
        print(f'Crisp Accuracy on benchmark : {crisp_acc * 100:.1f}%')

    @staticmethod
    def _print_statistics(m: dict):
        print('\n' + '=' * 80)
        print('STATISTICAL EVALUATION RESULTS')
        print('=' * 80)
        print(f"  Valid samples              : {m['n_valid']}")
        print()
        print('  ── Category Agreement ─────────────────────────────────────')
        print(f"  Category Accuracy          : {m['accuracy'] * 100:.2f}%  (95% CI: [{m['ci_low'] * 100:.1f}%, {m['ci_high'] * 100:.1f}%])")
        print(f"  Cohen's Kappa (κ)          : {m['kappa']:.4f}  {('[Substantial]' if m['kappa'] >= 0.6 else '[Moderate]' if m['kappa'] >= 0.4 else '[Fair]')}")
        print(f"  Matthews CC (MCC)          : {m['mcc']:.4f}  (range −1 to +1, higher = better)")
        print()
        print('  ── Continuous Index Agreement (scale 0–5) ─────────────────')
        print(f"  Pearson r                  : {m['pearson_r']:.4f}  (p = {m['pearson_p']:.2e})")
        print(f"  R² Score                   : {m['r2']:.4f}  ({m['r2'] * 100:.1f}% of variance explained)")
        print(f"  RMSE                       : {m['rmse']:.4f}")
        print(f"  MAE                        : {m['mae']:.4f}")
        direction = 'fuzzy OVER-predicts' if m['bias'] > 0 else 'fuzzy UNDER-predicts'
        print(f"  Mean Bias (Fuzzy − Crisp)  : {m['bias']:+.4f}  ({direction})")
        print(f"  Std of Errors (±1σ)        : {m['std_error']:.4f}")
        print(f"  Limits of Agreement (±1.96σ): [{m['bias'] - 1.96 * m['std_error']:+.3f},  {m['bias'] + 1.96 * m['std_error']:+.3f}]")
        print()
        print('  ── Per-Category Agreement ─────────────────────────────────')
        conf = m['conf_matrix']
        total_per_cat = conf.sum(axis=1)
        cat_names = ['Good', 'Moderate', 'USG', 'Unhealthy', 'V.Unhealthy']
        print(f"  {'Category':<14} {'N':>5} {'Correct':>8} {'Recall':>8} {'Precision':>10}")
        print('  ' + '-' * 47)
        for k in range(5):
            total = total_per_cat[k]
            correct = conf[k, k]
            recall = correct / total if total > 0 else 0.0
            fuzzy_total = conf[:, k].sum()
            precision = correct / fuzzy_total if fuzzy_total > 0 else 0.0
            print(f'  {cat_names[k]:<14} {total:>5.0f} {correct:>8.0f} {recall:>8.3f} {precision:>10.3f}')
        print()
        print('  ── Classification Report ──────────────────────────────────')
        print('  (Fuzzy predictions vs. Crisp EPA reference)')
        report_df = pd.DataFrame(m['cls_report']).T
        print(report_df.round(3).to_string())

def compute_pair_metrics(fis, crisp, pm25_arr, no2_arr, verbose: bool=False, show_progress: bool=False) -> dict:
    pm25_arr = np.asarray(pm25_arr, dtype=float)
    no2_arr = np.asarray(no2_arr, dtype=float)
    (fis_vals, fis_cats) = fis.evaluate_batch(pm25_arr, no2_arr, show_progress=show_progress)
    (crisp_vals, crisp_cats) = crisp.evaluate_batch(pm25_arr, no2_arr)
    fis_idx = np.array([_label_to_idx(c) for c in fis_cats])
    crisp_idx = np.array([_label_to_idx(c) for c in crisp_cats])
    valid = ~(np.isnan(fis_vals) | np.isnan(crisp_vals))
    fis_vals_v = fis_vals[valid]
    crisp_vals_v = crisp_vals[valid]
    fis_idx_v = fis_idx[valid]
    crisp_idx_v = crisp_idx[valid]
    accuracy = float(np.mean(fis_idx_v == crisp_idx_v))
    rmse = float(np.sqrt(mean_squared_error(crisp_vals_v, fis_vals_v)))
    mae = float(mean_absolute_error(crisp_vals_v, fis_vals_v))
    kappa = cohen_kappa_score(crisp_idx_v, fis_idx_v)
    macro_f1 = float(f1_score(crisp_idx_v, fis_idx_v, average='macro', labels=list(range(5)), zero_division=0))
    mcc = matthews_corrcoef(crisp_idx_v, fis_idx_v)
    (r_val, p_val) = pearsonr(crisp_vals_v, fis_vals_v)
    bias = float(np.mean(fis_vals_v - crisp_vals_v))
    std_err = float(np.std(fis_vals_v - crisp_vals_v))
    n_ci = int(valid.sum())
    p_hat = float(accuracy)
    z_ci = 1.96
    denom = 1.0 + z_ci ** 2 / n_ci
    center = (p_hat + z_ci ** 2 / (2 * n_ci)) / denom
    margin = z_ci * np.sqrt(p_hat * (1 - p_hat) / n_ci + z_ci ** 2 / (4 * n_ci ** 2)) / denom
    ci_low = float(np.clip(center - margin, 0.0, 1.0))
    ci_high = float(np.clip(center + margin, 0.0, 1.0))
    conf_mat = confusion_matrix(crisp_idx_v, fis_idx_v, labels=list(range(5)))
    report = classification_report(crisp_idx_v, fis_idx_v, labels=list(range(5)), target_names=CATEGORY_SHORT, output_dict=True, zero_division=0)
    metrics = {'n_valid': n_ci, 'accuracy': accuracy, 'ci_low': ci_low, 'ci_high': ci_high, 'rmse': rmse, 'mae': mae, 'kappa': float(kappa), 'macro_f1': macro_f1, 'mcc': float(mcc), 'pearson_r': float(r_val), 'pearson_p': float(p_val), 'r2': float(r_val ** 2), 'bias': bias, 'std_error': std_err, 'conf_matrix': conf_mat, 'cls_report': report, 'fis_vals': fis_vals_v, 'crisp_vals': crisp_vals_v, 'fis_idx': fis_idx_v, 'crisp_idx': crisp_idx_v, 'pm25': pm25_arr[valid], 'no2': no2_arr[valid]}
    if verbose:
        Evaluator._print_statistics(metrics)
    return metrics
