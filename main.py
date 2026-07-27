import os
import sys
import time
import textwrap
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from fuzzy_aqi import MamdaniFIS, CrispEPA, Evaluator, Visualizer

def separator(title: str='', width: int=72):
    if title:
        pad = max(0, (width - len(title) - 2) // 2)
        print('=' * pad + f'  {title}  ' + '=' * pad)
    else:
        print('=' * width)

def print_header():
    header = textwrap.dedent('    +======================================================================+\n    |  BLU 513E -- Fuzzy Logic Course Project                             |\n    |  Air Quality Index Estimation Using Mamdani FIS                     |\n    |  Author : Anil Cagin Gok  |  ID : 708251021                        |\n    |  Istanbul Technical University  |  May 2026                        |\n    +======================================================================+\n    ')
    print(header)

def demo_single_evaluations(fis: MamdaniFIS, crisp: CrispEPA):
    separator('DEMO: Single-Point Evaluations')
    demo_inputs = [(5.0, 10.0, 'Low pollution day (Good)'), (18.0, 60.0, 'Moderate urban pollution'), (45.0, 100.0, 'Moderate traffic + USG NO₂'), (5.0, 145.0, 'Good PM2.5, but Unhealthy NO₂ → conservative rule'), (90.0, 60.0, 'Unhealthy PM2.5, Moderate NO₂ → dominant PM2.5'), (160.0, 185.0, 'Very Unhealthy — severe smog event')]
    header = f"{'PM2.5':>8}  {'NO₂':>6}  {'Fuzzy Val':>10}  {'Fuzzy Cat':<35}  {'Crisp Cat':<35}"
    print(header)
    print('-' * len(header))
    for (pm25, no2, desc) in demo_inputs:
        (fis_val, fis_cat) = fis.evaluate(pm25, no2)
        (crisp_val, crisp_cat) = crisp.evaluate(pm25, no2)
        print(f'  {pm25:6.1f}  {no2:6.1f}  {fis_val:10.4f}  {fis_cat:<35}  {crisp_cat:<35}  ← {desc}')
    print()

def print_rule_base(fis: MamdaniFIS):
    separator('RULE BASE — 5×5 Matrix (output = max severity)')
    fis.print_rule_table()
    print()
    print('Encoding: G=Good  M=Moderate  U=USG  H=Unhealthy  VU=Very Unhealthy')
    print('Rule principle: IF PM2.5 is X AND NO₂ is Y THEN AQI is max(X, Y)')
    print()

def main():
    t0 = time.time()
    print_header()
    separator('BUILDING SYSTEMS')
    print('  Initialising Mamdani FIS…')
    fis = MamdaniFIS(resolution=1001)
    print('  Initialising Crisp EPA reference system…')
    crisp = CrispEPA()
    print('  Systems ready.\n')
    print_rule_base(fis)
    demo_single_evaluations(fis, crisp)
    evaluator = Evaluator(fis, crisp)
    separator('BENCHMARK TEST CASES')
    bench_df = evaluator.run_benchmark(verbose=True)
    bench_df.to_csv('results/benchmark_results.csv', index=False)
    print('\n  Benchmark CSV saved: results/benchmark_results.csv')
    separator('STATISTICAL EVALUATION (500 random samples)')
    metrics = evaluator.run_statistical_evaluation(n_samples=500, seed=42, verbose=True)
    evaluator.results_df.to_csv('results/evaluation_results.csv', index=False)
    print('\n  Evaluation CSV saved: results/evaluation_results.csv')
    separator('ZONE-BASED AGREEMENT ANALYSIS')
    zone_results = evaluator.run_zone_analysis(metrics)
    print(f"\n  {'Pollution Zone':<30} {'N':>5}  {'Accuracy':>9}  {'Bias (F−C)':>11}  {'RMSE':>6}")
    print('  ' + '-' * 68)
    for (zone, res) in zone_results.items():
        print(f"  {zone:<30} {res['n']:>5}  {res['accuracy'] * 100:>8.2f}%  {res['bias']:>+11.4f}  {res['rmse']:>6.4f}")
    print()
    separator('RESPONSE SURFACE COMPUTATION')
    surface_data = evaluator.compute_surface(n_pts=40)
    separator('FIGURE GENERATION')
    viz = Visualizer(output_dir='results')
    viz.plot_all(fis, metrics, surface_data)
    elapsed = time.time() - t0
    separator('SUMMARY — KEY NUMERICAL RESULTS')
    print()
    print('  ┌─ CATEGORY AGREEMENT ──────────────────────────────────────┐')
    print(f"  │  Category Accuracy    : {metrics['accuracy'] * 100:6.2f}%                             │")
    print(f"  │  95% CI (Wilson)      : [{metrics['ci_low'] * 100:.1f}%,  {metrics['ci_high'] * 100:.1f}%]                      │")
    kappa_label = 'Substantial' if metrics['kappa'] >= 0.6 else 'Moderate' if metrics['kappa'] >= 0.4 else 'Fair'
    print(f"  │  Cohen's Kappa (κ)   : {metrics['kappa']:7.4f}  [{kappa_label}]               │")
    print(f"  │  Matthews CC (MCC)   : {metrics['mcc']:7.4f}  (−1 worst … +1 perfect)   │")
    print('  └───────────────────────────────────────────────────────────┘')
    print()
    print('  ┌─ CONTINUOUS INDEX AGREEMENT (scale 0–5) ──────────────────┐')
    print(f"  │  Pearson r            : {metrics['pearson_r']:7.4f}  (p = {metrics['pearson_p']:.2e})         │")
    print(f"  │  R² Score             : {metrics['r2']:7.4f}  ({metrics['r2'] * 100:.1f}% variance explained)  │")
    print(f"  │  RMSE                 : {metrics['rmse']:7.4f}                             │")
    print(f"  │  MAE                  : {metrics['mae']:7.4f}                             │")
    direction = 'over-pred.' if metrics['bias'] > 0 else 'under-pred.'
    print(f"  │  Mean Bias (F−C)      : {metrics['bias']:+7.4f}  ({direction})                 │")
    print(f"  │  Std of Errors (σ)    : {metrics['std_error']:7.4f}                             │")
    loa_lo = metrics['bias'] - 1.96 * metrics['std_error']
    loa_hi = metrics['bias'] + 1.96 * metrics['std_error']
    print(f'  │  Limits of Agreement  : [{loa_lo:+.3f},  {loa_hi:+.3f}]                │')
    print('  └───────────────────────────────────────────────────────────┘')
    print()
    print('  ┌─ PER-CATEGORY RECALL (rows = crisp reference) ────────────┐')
    conf = metrics['conf_matrix']
    cat_names = ['Good', 'Moderate', 'USG', 'Unhealthy', 'V.Unhealthy']
    print(f"  │  {'Category':<12} {'N':>5}  {'Correct':>7}  {'Recall':>7}  {'Precision':>9}  │")
    print('  │  ' + '-' * 52 + '  │')
    for k in range(5):
        n_cat = conf.sum(axis=1)[k]
        corr = conf[k, k]
        recall = corr / n_cat if n_cat > 0 else 0.0
        prec = corr / conf[:, k].sum() if conf[:, k].sum() > 0 else 0.0
        print(f'  │  {cat_names[k]:<12} {n_cat:>5.0f}  {corr:>7.0f}  {recall:>7.3f}  {prec:>9.3f}  │')
    print('  └───────────────────────────────────────────────────────────┘')
    print()
    print(f'  Total runtime      : {elapsed:.1f} s')
    print()
    print('  All results saved in: ./results/')
    print('  Figures  : 13 PNG files  (fig_*.png)')
    print('  Tables   : benchmark_results.csv  |  evaluation_results.csv')
    print()
    separator()
    separator('INFERENCE TRACE — sample: PM2.5=45 μg/m³, NO₂=100 μg/m³')
    (pm25_test, no2_test) = (45.0, 100.0)
    (val, cat) = fis.evaluate(pm25_test, no2_test)
    print(f'  Input  : PM2.5 = {pm25_test} μg/m³,  NO₂ = {no2_test} μg/m³')
    print(f"  Output : AQI index = {val:.4f}  →  Category = '{cat}'")
    print()
    import skfuzzy as fuzz
    import numpy as np
    from fuzzy_aqi.fuzzy_system import PM25_MF_PARAMS, NO2_MF_PARAMS, TERM_KEYS

    def get_membership(value, params_dict, universe):
        grades = {}
        for (term, (mf_type, params)) in params_dict.items():
            if mf_type == 'trimf':
                mf = fuzz.trimf(universe, params)
            else:
                mf = fuzz.trapmf(universe, params)
            grades[term] = float(fuzz.interp_membership(universe, mf, value))
        return grades
    pm25_u = np.linspace(0, 250, 1001)
    no2_u = np.linspace(0, 200, 1001)
    pm25_grades = get_membership(pm25_test, PM25_MF_PARAMS, pm25_u)
    no2_grades = get_membership(no2_test, NO2_MF_PARAMS, no2_u)
    print('  Fuzzification:')
    print(f'    PM2.5 = {pm25_test} μg/m³ →')
    for (term, grade) in pm25_grades.items():
        bar = '█' * int(grade * 20)
        print(f'      μ_{term:<15} = {grade:.4f}  {bar}')
    print(f'    NO₂ = {no2_test} μg/m³ →')
    for (term, grade) in no2_grades.items():
        bar = '█' * int(grade * 20)
        print(f'      μ_{term:<15} = {grade:.4f}  {bar}')
    print()
    print('  Active rules (firing strength > 0):')
    for (i, p) in enumerate(TERM_KEYS):
        for (j, n) in enumerate(TERM_KEYS):
            strength = min(pm25_grades[p], no2_grades[n])
            if strength > 0.001:
                out = TERM_KEYS[max(i, j)]
                print(f'    IF PM2.5={p:<15} AND NO2={n:<15} → AQI={out:<15}  strength={strength:.4f}')
    print(f"\n  Defuzzification (centroid) → AQI = {val:.4f}  →  '{cat}'")
    separator()
if __name__ == '__main__':
    main()
