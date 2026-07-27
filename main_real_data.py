import os
import sys
import time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from fuzzy_aqi import config
from fuzzy_aqi.baselines import KNNBaseline, baseline_accuracy
from fuzzy_aqi.calibration import run_full_calibration, save_calibration
from fuzzy_aqi.dataset import AirQualityDataset
from fuzzy_aqi.evaluation import compute_pair_metrics, _label_to_idx
from fuzzy_aqi.real_evaluation import RealDataEvaluator
from fuzzy_aqi.sources.csv_source import CSVSource

def _save_run_summary(path, dataset, cal_cfg, metrics, elapsed_sec):
    lines = ['BLU 513E — Real Data Evaluation Summary', '=' * 50, f'Runtime (s)     : {elapsed_sec:.1f}', f"Particulate mode: {cal_cfg.get('particulate_mode', 'n/a').upper()}", f"PM10->PM2.5     : {cal_cfg.get('pm10_to_pm25_factor', 'n/a')}", f"Bias correction : {cal_cfg.get('bias_correction', 0):+.4f}", f"Thresholds      : {cal_cfg.get('category_thresholds', [])}", '', dataset.summary(), '', 'Split metrics (Fuzzy vs Crisp EPA):']
    for split in ('train', 'val', 'test'):
        if split in metrics:
            m = metrics[split]
            lines.append(f"  {split:5s}  acc={m['accuracy'] * 100:6.2f}%  kappa={m['kappa']:.4f}  macro-F1={m.get('macro_f1', 0):.4f}  rmse={m['rmse']:.4f}")
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

def separator(title: str='', width: int=72):
    if title:
        pad = max(0, (width - len(title) - 2) // 2)
        print('=' * pad + f'  {title}  ' + '=' * pad)
    else:
        print('=' * width)

def main():
    t0 = time.time()
    config.ensure_data_dir()
    separator('BLU 513E — Real Data Evaluation (Istanbul)')
    print()
    print('  Data source : IBB Open Data API (cached CSV)')
    print('  Pipeline    : full calibration + fast lookup + baselines')
    print(f'  Expected CSV: {config.ISTANBUL_HOURLY_CSV}')
    print()
    if not os.path.isfile(config.ISTANBUL_HOURLY_CSV):
        print('  ERROR: Hourly data file not found.')
        print('  Run: python scripts/fetch_istanbul_ibb.py')
        sys.exit(1)
    separator('LOAD DATASET')
    source = CSVSource(config.ISTANBUL_HOURLY_CSV)
    dataset = AirQualityDataset(source, use_daily=True, particulate_mode='pm10').build()
    print(dataset.summary())
    print()
    train = dataset.splits['train']
    val = dataset.splits.get('val')
    separator('FULL CALIBRATION (train / val only)')
    print(f"  Particulate   : {config.ISTANBUL_PARTICULATE_MODE.upper()} (IBB native PM10; set config.ISTANBUL_PARTICULATE_MODE='pm25' for estimated PM2.5)")
    print()
    cal = run_full_calibration(train, val, build_lookup=True, lookup_size=101, verbose=True)
    fis = cal['fis']
    crisp = cal['crisp']
    cal_cfg = cal['config']
    dataset.particulate_mode = cal_cfg['particulate_mode']
    dataset.particulate_col = cal_cfg['particulate_col']
    pcol = cal_cfg['particulate_col']
    print()
    separator('BASELINE COMPARISON (test split)')
    test = dataset.splits['test']
    (_, crisp_cats) = crisp.evaluate_batch(test[pcol].values, test['no2'].values)
    crisp_idx = np.array([_label_to_idx(c) for c in crisp_cats])
    m_fis = compute_pair_metrics(fis, crisp, test[pcol].values, test['no2'].values, verbose=False)
    print(f"  Mamdani FIS (calibrated) : {m_fis['accuracy'] * 100:.2f}%  kappa={m_fis['kappa']:.4f}  macro-F1={m_fis.get('macro_f1', 0):.4f}")
    (_, train_crisp_cats) = crisp.evaluate_batch(train[pcol].values, train['no2'].values)
    knn = KNNBaseline(n_neighbors=11)
    knn.fit(train[pcol].values, train['no2'].values, train_crisp_cats)
    knn_idx = np.array([_label_to_idx(c) for c in knn.evaluate_batch(test[pcol].values, test['no2'].values)[1]])
    knn_acc = baseline_accuracy(knn_idx, crisp_idx) * 100
    print(f'  k-NN vs Crisp EPA (ref.) : {knn_acc:.2f}%  (interpolates EPA; not fair vs FIS)')
    print()
    separator('EVALUATE SPLITS (train / val / test)')
    evaluator = RealDataEvaluator(fis, crisp, dataset, output_dir=config.RESULTS_DIR)
    evaluator.evaluate_splits(verbose=True)
    evaluator.print_summary_table()
    separator('ZONE ANALYSIS (test split)')
    zones = evaluator.run_zone_analysis('test')
    for (zone, res) in zones.items():
        print(f"  {zone:<32} n={res['n']:>6,}  acc={res['accuracy'] * 100:6.2f}%  bias={res['bias']:+.4f}  rmse={res['rmse']:.4f}")
    print()
    separator('SAVE OUTPUTS')
    pred_path = evaluator.save_results_csv('test')
    evaluator.plot_real_data_summary('test')
    print(f'  Predictions CSV : {pred_path}')
    cal_json = os.path.join(config.RESULTS_DIR, 'calibration_config.json')
    save_calibration(cal_json, cal)
    print(f'  Calibration JSON: {cal_json}')
    elapsed = time.time() - t0
    summary_path = os.path.join(config.RESULTS_DIR, 'evaluation_summary.txt')
    _save_run_summary(summary_path, dataset, cal_cfg, evaluator.metrics, elapsed)
    print(f'  Run summary      : {summary_path}')
    if 'test' in evaluator.metrics:
        m = evaluator.metrics['test']
        print()
        print(f"  Mode            : {cal_cfg['particulate_mode'].upper()} + NO2")
        print(f"  PM10->PM2.5     : {cal_cfg['pm10_to_pm25_factor']}")
        print(f"  Bias correction : {cal_cfg['bias_correction']:+.4f}")
        print(f"  Thresholds      : {cal_cfg['category_thresholds']}")
        print(f"  TEST accuracy   : {m['accuracy'] * 100:.2f}%")
        print(f"  TEST kappa      : {m['kappa']:.4f}")
        print(f"  TEST macro-F1   : {m.get('macro_f1', 0):.4f}")
        print(f"  TEST RMSE       : {m['rmse']:.4f}")
    print()
    separator()
    print(f'  Total runtime: {elapsed:.1f} s')
    print()
if __name__ == '__main__':
    main()
