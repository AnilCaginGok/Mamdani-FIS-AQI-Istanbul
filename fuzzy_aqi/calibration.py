from __future__ import annotations
import json
import os
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score
from . import config
from .crisp_system import CrispEPA
from .evaluation import compute_pair_metrics, _label_to_idx
from .fuzzy_system import MamdaniFIS
from .sources.base import COL_DATETIME

def _subsample(p, no2, max_n: int=5000, seed: int=42):
    p = np.asarray(p, dtype=float)
    no2 = np.asarray(no2, dtype=float)
    if len(p) <= max_n:
        return (p, no2)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(p), max_n, replace=False)
    return (p[idx], no2[idx])

def _pred_from_thresholds(fis_vals: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    t = thresholds
    pred = np.zeros(len(fis_vals), dtype=int)
    pred[fis_vals >= t[0]] = 1
    pred[fis_vals >= t[1]] = 2
    pred[fis_vals >= t[2]] = 3
    pred[fis_vals >= t[3]] = 4
    return pred

def _min_threshold_gap(particulate_mode: str='pm25') -> float:
    if particulate_mode == 'pm10':
        return config.MIN_THRESHOLD_GAP_PM10
    return config.MIN_THRESHOLD_GAP

def _thresholds_valid(thresholds: np.ndarray, min_gap: float=None, particulate_mode: str='pm25') -> bool:
    gap = min_gap if min_gap is not None else _min_threshold_gap(particulate_mode)
    return bool(np.all(np.diff(thresholds) >= gap))

def _class_weights_balanced(crisp_idx: np.ndarray) -> np.ndarray:
    counts = np.bincount(crisp_idx, minlength=5).astype(float)
    weights = 1.0 / (counts + 1.0)
    return weights / weights.sum()

def _composite_score(fis_vals: np.ndarray, crisp_idx: np.ndarray, thresholds: np.ndarray, particulate_mode: str='pm25') -> float:
    pred = _pred_from_thresholds(fis_vals, thresholds)
    acc = float(np.mean(pred == crisp_idx))
    kappa = float(cohen_kappa_score(crisp_idx, pred, labels=list(range(5))))
    macro_f1 = float(f1_score(crisp_idx, pred, average='macro', labels=list(range(5)), zero_division=0))
    w = _class_weights_balanced(crisp_idx)
    bal_recall = 0.0
    for c in range(5):
        mask = crisp_idx == c
        if mask.sum() > 0:
            bal_recall += w[c] * float(np.mean(pred[mask] == c))
    if particulate_mode == 'pm10':
        return 0.5 * acc + 0.3 * kappa + 0.12 * macro_f1 + 0.08 * bal_recall
    return 0.3 * acc + 0.45 * kappa + 0.15 * macro_f1 + 0.1 * bal_recall

def _ensure_finite_thresholds(thresholds: np.ndarray, particulate_mode: str='pm25') -> np.ndarray:
    gap = _min_threshold_gap(particulate_mode)
    t = np.array(thresholds, dtype=float).flatten()
    if t.size != 4 or not np.all(np.isfinite(t)):
        t = np.array([0.95, 1.85, 2.75, 3.65], dtype=float)
    t = np.clip(t, 0.4, 4.8)
    for k in range(1, 4):
        if t[k] <= t[k - 1] + gap:
            t[k] = t[k - 1] + gap
    return t

def _init_thresholds_from_data(fis_vals: np.ndarray, crisp_idx: np.ndarray, particulate_mode: str='pm25') -> np.ndarray:
    defaults = np.array([0.95, 1.85, 2.75, 3.65], dtype=float)
    for k in range(4):
        border = (crisp_idx == k) | (crisp_idx == k + 1)
        if border.sum() >= 30:
            med = float(np.nanmedian(fis_vals[border]))
            if np.isfinite(med):
                defaults[k] = med
    return _ensure_finite_thresholds(defaults, particulate_mode=particulate_mode)

def compare_particulate_modes(pm25_arr, pm10_arr, no2_arr, verbose: bool=True) -> dict:
    results = {}
    for (mode, p_arr) in [('pm25', pm25_arr), ('pm10', pm10_arr)]:
        fis = _calibration_fis(mode)
        crisp = CrispEPA(particulate_mode=mode)
        m = compute_pair_metrics(fis, crisp, p_arr, no2_arr, verbose=False)
        results[mode] = m
        if verbose:
            print(f"  Mode {mode:4s}: accuracy={m['accuracy'] * 100:6.2f}%  kappa={m['kappa']:.4f}  rmse={m['rmse']:.4f}  n={m['n_valid']:,}")
    return results

def fit_bias_correction(fis, crisp, particulate_arr, no2_arr, verbose: bool=False) -> float:
    if fis._lookup is None and fis._lookup_p is None:
        if verbose:
            print(f'    Pre-building calibration lookup ({config.CALIBRATION_LOOKUP_GRID}×{config.CALIBRATION_LOOKUP_GRID})…', flush=True)
        fis.build_lookup(grid_size=config.CALIBRATION_LOOKUP_GRID, verbose=verbose)
    (fis_vals, _) = fis.evaluate_batch(particulate_arr, no2_arr, show_progress=verbose)
    (crisp_vals, _) = crisp.evaluate_batch(particulate_arr, no2_arr)
    valid = ~(np.isnan(fis_vals) | np.isnan(crisp_vals))
    if valid.sum() == 0:
        return 0.0
    return float(np.mean(fis_vals[valid] - crisp_vals[valid]))

def apply_bias_calibration(fis, crisp, particulate_arr, no2_arr, verbose: bool=True) -> float:
    bias = fit_bias_correction(fis, crisp, particulate_arr, no2_arr, verbose=verbose)
    fis.set_bias_correction(bias)
    if verbose:
        direction = 'over' if bias > 0 else 'under'
        print(f'  Bias correction: {bias:+.4f} (fuzzy {direction}-predicts on train)', flush=True)
    return bias

def _accuracy_with_thresholds(fis_vals, crisp_idx, thresholds) -> float:
    pred = _pred_from_thresholds(fis_vals, thresholds)
    return float(np.mean(pred == crisp_idx))

def _calibration_fis(particulate_mode: str='pm25', resolution: int=None) -> MamdaniFIS:
    res = resolution if resolution is not None else config.CALIBRATION_FIS_RESOLUTION
    return MamdaniFIS(particulate_mode=particulate_mode, resolution=res)

def _should_skip_pm10_to_pm25_factor_search() -> bool:
    if not getattr(config, 'SKIP_PM10_TO_PM25_FACTOR_WHEN_NATIVE', True):
        return False
    forced = getattr(config, 'ISTANBUL_PARTICULATE_MODE', None)
    return forced == 'pm10'

def _score_factor(f: float, pm10_arr, no2_arr, w: float) -> tuple:
    pm25 = pm10_arr * f
    fis = _calibration_fis('pm25', resolution=config.FACTOR_SEARCH_RESOLUTION)
    crisp = CrispEPA(particulate_mode='pm25')
    grid = config.FACTOR_SEARCH_LOOKUP_GRID
    fis.build_lookup(grid_size=grid, verbose=False)
    apply_bias_calibration(fis, crisp, pm25, no2_arr, verbose=False)
    fis.build_lookup(grid_size=grid, verbose=False)
    m = compute_pair_metrics(fis, crisp, pm25, no2_arr, verbose=False)
    score = (1.0 - w) * m['accuracy'] + w * max(0.0, m['kappa'])
    return (float(f), score)

def calibrate_pm10_to_pm25_factor(pm10_arr, no2_arr, factors=None, verbose: bool=True) -> tuple:
    max_n = getattr(config, 'FACTOR_SEARCH_SAMPLE_N', 2000)
    (pm10_arr, no2_arr) = _subsample(pm10_arr, no2_arr, max_n=max_n)
    best_factor = config.PM10_TO_PM25_FACTOR
    best_score = -1.0
    w = config.FACTOR_SCORE_KAPPA_WEIGHT
    if factors is None:
        coarse = np.round(np.arange(0.76, 0.95, 0.04), 2)
    else:
        coarse = np.asarray(factors, dtype=float)
    n_coarse = len(coarse)
    for (i, f) in enumerate(coarse):
        if verbose:
            print(f'    coarse {i + 1}/{n_coarse}: factor={float(f):.2f}…', flush=True)
        (f, score) = _score_factor(float(f), pm10_arr, no2_arr, w)
        if score > best_score:
            best_score = score
            best_factor = f
    fine = np.round(np.arange(best_factor - 0.04, best_factor + 0.041, 0.02), 2)
    fine = fine[(fine >= 0.7) & (fine <= 0.98)]
    n_fine = len(fine)
    for (i, f) in enumerate(fine):
        if verbose:
            print(f'    fine {i + 1}/{n_fine}: factor={float(f):.2f}…', flush=True)
        (f, score) = _score_factor(float(f), pm10_arr, no2_arr, w)
        if score > best_score:
            best_score = score
            best_factor = f
    if verbose:
        print(f'  PM10->PM2.5 factor: {best_factor}  (composite score {best_score:.4f})', flush=True)
    return (best_factor, best_score)

def calibrate_category_thresholds(fis: MamdaniFIS, crisp: CrispEPA, particulate_arr, no2_arr, verbose: bool=True) -> np.ndarray:
    (p_sub, n_sub) = _subsample(particulate_arr, no2_arr, max_n=12000)
    if fis._lookup is None and fis._lookup_p is None:
        fis.build_lookup(grid_size=config.CALIBRATION_LOOKUP_GRID, verbose=False)
    (fis_vals, _) = fis.evaluate_batch(p_sub, n_sub)
    (_, crisp_cats) = crisp.evaluate_batch(p_sub, n_sub)
    crisp_idx = np.array([_label_to_idx(c) for c in crisp_cats])
    pmode = fis.particulate_mode
    tgap = _min_threshold_gap(pmode)
    thresholds = _init_thresholds_from_data(fis_vals, crisp_idx, pmode)
    best_score = _composite_score(fis_vals, crisp_idx, thresholds, pmode)
    best_acc = _accuracy_with_thresholds(fis_vals, crisp_idx, thresholds)
    for _ in range(7):
        for k in range(4):
            lo = thresholds[k - 1] + tgap if k > 0 else 0.4
            hi = thresholds[k + 1] - tgap if k < 3 else 4.75
            if hi <= lo:
                continue
            candidates = np.linspace(lo, hi, 45)
            local_best = thresholds[k]
            local_score = best_score
            local_acc = best_acc
            for t in candidates:
                if not np.isfinite(t):
                    continue
                trial = _ensure_finite_thresholds(np.concatenate([thresholds[:k], [t], thresholds[k + 1:]]), particulate_mode=pmode)
                if not _thresholds_valid(trial, particulate_mode=pmode):
                    continue
                score = _composite_score(fis_vals, crisp_idx, trial, pmode)
                acc = _accuracy_with_thresholds(fis_vals, crisp_idx, trial)
                if score > local_score or (score == local_score and acc > local_acc):
                    local_score = score
                    local_acc = acc
                    local_best = trial[k]
            thresholds = _ensure_finite_thresholds(np.concatenate([thresholds[:k], [local_best], thresholds[k + 1:]]), particulate_mode=pmode)
            best_score = _composite_score(fis_vals, crisp_idx, thresholds, pmode)
            best_acc = _accuracy_with_thresholds(fis_vals, crisp_idx, thresholds)
    thresholds = _ensure_finite_thresholds(thresholds, particulate_mode=pmode)
    fis.set_category_thresholds(thresholds)
    if verbose:
        acc = _accuracy_with_thresholds(fis_vals, crisp_idx, thresholds)
        pred = _pred_from_thresholds(fis_vals, thresholds)
        kappa = cohen_kappa_score(crisp_idx, pred, labels=list(range(5)))
        macro = f1_score(crisp_idx, pred, average='macro', labels=list(range(5)), zero_division=0)
        print(f'  Category thresholds: {thresholds.round(3).tolist()}', flush=True)
        print(f'    train acc={acc * 100:.2f}%  kappa={kappa:.4f}  macro-F1={macro:.4f}  composite={best_score:.4f}', flush=True)
    return thresholds

def _val_composite_score(fis_vals: np.ndarray, crisp_idx: np.ndarray, thresholds: np.ndarray) -> float:
    pred = _pred_from_thresholds(fis_vals, thresholds)
    acc = float(np.mean(pred == crisp_idx))
    kappa = float(cohen_kappa_score(crisp_idx, pred, labels=list(range(5))))
    macro_f1 = float(f1_score(crisp_idx, pred, average='macro', labels=list(range(5)), zero_division=0))
    return 0.35 * acc + 0.5 * kappa + 0.15 * macro_f1

def refine_thresholds_on_validation(fis: MamdaniFIS, crisp: CrispEPA, val_p, val_n, train_thresholds: np.ndarray, verbose: bool=True) -> np.ndarray:
    (p_sub, n_sub) = _subsample(val_p, val_n, max_n=6000)
    if fis._lookup is None and fis._lookup_p is None:
        fis.build_lookup(grid_size=config.CALIBRATION_LOOKUP_GRID, verbose=False)
    (fis_vals, _) = fis.evaluate_batch(p_sub, n_sub)
    (_, crisp_cats) = crisp.evaluate_batch(p_sub, n_sub)
    crisp_idx = np.array([_label_to_idx(c) for c in crisp_cats])
    thresholds = _ensure_finite_thresholds(np.array(train_thresholds, dtype=float))
    best_score = _val_composite_score(fis_vals, crisp_idx, thresholds)
    for _ in range(3):
        for k in range(4):
            center = thresholds[k]
            lo = max(thresholds[k - 1] + config.MIN_THRESHOLD_GAP if k > 0 else 0.4, center - 0.25)
            hi = min(thresholds[k + 1] - config.MIN_THRESHOLD_GAP if k < 3 else 4.75, center + 0.25)
            if hi <= lo:
                continue
            for t in np.linspace(lo, hi, 21):
                trial = _ensure_finite_thresholds(np.concatenate([thresholds[:k], [t], thresholds[k + 1:]]))
                if not _thresholds_valid(trial):
                    continue
                score = _val_composite_score(fis_vals, crisp_idx, trial)
                if score > best_score:
                    best_score = score
                    thresholds = trial
    thresholds = _ensure_finite_thresholds(thresholds)
    fis.set_category_thresholds(thresholds)
    if verbose:
        pred = _pred_from_thresholds(fis_vals, thresholds)
        kappa = cohen_kappa_score(crisp_idx, pred, labels=list(range(5)))
        acc = float(np.mean(pred == crisp_idx))
        print(f'  Val-refined thresholds: {thresholds.round(3).tolist()}  (val acc={acc * 100:.2f}%  kappa={kappa:.4f})', flush=True)
    return thresholds

def _mode_composite_score(metrics: dict) -> float:
    w = config.MODE_SCORE_KAPPA_WEIGHT
    mf1 = metrics.get('macro_f1', 0.0)
    return (1.0 - w) * metrics['accuracy'] + w * max(0.0, metrics['kappa']) + 0.05 * mf1

def _temporal_holdout_split(train, holdout_ratio: float=None):
    ratio = holdout_ratio if holdout_ratio is not None else config.MODE_HOLDOUT_RATIO
    df = train.sort_values(COL_DATETIME).reset_index(drop=True)
    cut = max(1, int(len(df) * (1.0 - ratio)))
    return (df.iloc[:cut], df.iloc[cut:])

def _fit_mode_on_split(mode: str, fit_df, eval_df, pcol: str) -> dict:
    (tr_p, tr_n) = _subsample(fit_df[pcol].values, fit_df['no2'].values, max_n=5000)
    (ev_p, ev_n) = (eval_df[pcol].values, eval_df['no2'].values)
    fis = _calibration_fis(mode)
    crisp = CrispEPA(particulate_mode=mode)
    apply_bias_calibration(fis, crisp, tr_p, tr_n, verbose=False)
    fis.build_lookup(grid_size=config.CALIBRATION_LOOKUP_GRID, verbose=False)
    calibrate_category_thresholds(fis, crisp, tr_p, tr_n, verbose=False)
    return compute_pair_metrics(fis, crisp, ev_p, ev_n, verbose=False)

def select_best_mode(train, val) -> str:
    (tr_fit, tr_hold) = _temporal_holdout_split(train)
    best_mode = 'pm10'
    best_score = -1.0
    w_val = config.MODE_VAL_WEIGHT
    w_hold = config.MODE_HOLDOUT_WEIGHT
    for mode in ('pm25', 'pm10'):
        pcol = 'pm10' if mode == 'pm10' else 'pm25'
        m_val = _fit_mode_on_split(mode, train, val, pcol)
        m_hold = _fit_mode_on_split(mode, tr_fit, tr_hold, pcol)
        score = w_val * _mode_composite_score(m_val) + w_hold * _mode_composite_score(m_hold)
        if config.PREFER_NATIVE_PARTICULATE and mode == 'pm10':
            score += config.NATIVE_PM10_SCORE_BONUS
        print(f"  {mode:4s}  val: acc={m_val['accuracy'] * 100:.2f}% k={m_val['kappa']:.3f}  holdout: acc={m_hold['accuracy'] * 100:.2f}% k={m_hold['kappa']:.3f}  score={score:.4f}", flush=True)
        if score > best_score:
            best_score = score
            best_mode = mode
    print(f'  -> Selected: {best_mode.upper()}  (blended score={best_score:.4f})', flush=True)
    return best_mode

def run_full_calibration(train, val=None, build_lookup: bool=True, lookup_size: int=101, verbose: bool=True) -> Dict:
    skip_factor_search = _should_skip_pm10_to_pm25_factor_search()
    if verbose:
        if skip_factor_search:
            print('  [1/6] PM10->PM2.5 factor search… skipped (native PM10 in config)', flush=True)
        else:
            print('  [1/6] PM10->PM2.5 factor search (two-stage, kappa-weighted)…', flush=True)
    if 'pm10' in train.columns:
        if skip_factor_search:
            best_factor = config.PM10_TO_PM25_FACTOR
            if verbose:
                print(f'  Using config default PM10->PM2.5 factor: {best_factor}', flush=True)
        else:
            (best_factor, _) = calibrate_pm10_to_pm25_factor(train['pm10'].values, train['no2'].values, verbose=verbose)
            config.PM10_TO_PM25_FACTOR = best_factor
        for split_df in (train, val) if val is not None else (train,):
            if split_df is not None and 'pm10' in split_df.columns:
                split_df['pm25'] = split_df['pm10'] * best_factor
    if verbose:
        print('  [2/6] Mode selection (val + temporal holdout)…', flush=True)
    forced = getattr(config, 'ISTANBUL_PARTICULATE_MODE', None)
    if forced in ('pm10', 'pm25'):
        selected_mode = forced
        if verbose:
            print(f'  -> Fixed (config): {forced.upper()}  (IBB native sensor)', flush=True)
    elif val is not None and len(val) > 100:
        selected_mode = select_best_mode(train, val)
    else:
        selected_mode = 'pm25'
        if verbose:
            print('  -> Default: PM25', flush=True)
    pcol = 'pm10' if selected_mode == 'pm10' else 'pm25'
    fis = MamdaniFIS(particulate_mode=selected_mode)
    crisp = CrispEPA(particulate_mode=selected_mode)
    if verbose:
        print('  [3/6] Bias calibration…', flush=True)
    bias_max_n = getattr(config, 'BIAS_CALIBRATION_SAMPLE_N', 3000)
    (bias_p, bias_n) = _subsample(train[pcol].values, train['no2'].values, max_n=bias_max_n)
    apply_bias_calibration(fis, crisp, bias_p, bias_n, verbose=verbose)
    if verbose:
        print('  [4/6] Balanced category thresholds (train)…', flush=True)
    (thr_p, thr_n) = _subsample(train[pcol].values, train['no2'].values, max_n=10000)
    thresholds = calibrate_category_thresholds(fis, crisp, thr_p, thr_n, verbose=verbose)
    do_val_refine = config.ENABLE_VAL_THRESHOLD_REFINE and val is not None and (len(val) > 200) and (not (config.VAL_REFINE_ONLY_PM25 and selected_mode != 'pm25'))
    if do_val_refine:
        if verbose:
            print('  [5/6] Validation threshold refinement…', flush=True)
        thresholds = refine_thresholds_on_validation(fis, crisp, val[pcol].values, val['no2'].values, thresholds, verbose=verbose)
    elif verbose and val is not None:
        print('  [5/6] Validation threshold refinement… skipped (native PM10)', flush=True)
    if verbose:
        print('  [6/6] Fast lookup grid…', flush=True)
    if build_lookup:
        final_grid = lookup_size or config.FINAL_LOOKUP_GRID
        fis.build_lookup(grid_size=final_grid, verbose=verbose)
        lookup_size = final_grid
    result = {'particulate_mode': selected_mode, 'particulate_col': pcol, 'pm10_to_pm25_factor': config.PM10_TO_PM25_FACTOR, 'bias_correction': fis.bias_correction, 'category_thresholds': thresholds.tolist(), 'lookup_size': lookup_size if build_lookup else 0}
    return {'fis': fis, 'crisp': crisp, 'config': result}

def save_calibration(path: str, cal: Dict) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cal['config'], f, indent=2)

def load_calibration(path: str) -> Optional[Dict]:
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
