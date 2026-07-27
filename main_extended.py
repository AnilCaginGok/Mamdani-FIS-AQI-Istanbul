import os
import sys
import time
import textwrap

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from fuzzy_aqi.console import configure_stdout

configure_stdout()

from fuzzy_aqi import (
    MamdaniFIS,
    CrispEPA,
    Evaluator,
    AirQualityDataset,
    run_full_calibration,
    save_calibration,
    MLComparison,
    GaussianMamdaniFIS,
    DefuzzificationComparison,
    ThreeInputFIS,
    SeasonalAnalyzer,
    StationAnalyzer,
    AdvancedVisualizer,
    config,
)
from fuzzy_aqi.evaluation import compute_pair_metrics, _label_to_idx
from fuzzy_aqi.sources.csv_source import CSVSource


def separator(title: str = "", width: int = 76) -> None:
    if title:
        pad = max(0, (width - len(title) - 2) // 2)
        print("=" * pad + f"  {title}  " + "=" * pad)
    else:
        print("=" * width)


def print_header() -> None:
    header = textwrap.dedent(
        """
    +=========================================================================+
    |  BLU 513E  --  Air Quality Index Estimation Using Mamdani FIS          |
    |  Extended Analysis: ML Comparison, Seasonal & Station Study             |
    |  Author : Anil Cagin Gok   |   ID : 708251021                          |
    |  Istanbul Technical University   |   May 2026                          |
    +=========================================================================+
    """
    )
    print(header)


def main() -> None:
    t0 = time.time()
    config.ensure_data_dir()
    print_header()

    # ------------------------------------------------------------------
    # Data loading and calibration
    # ------------------------------------------------------------------
    separator("DATA LOADING & CALIBRATION")
    print()
    if not os.path.isfile(config.ISTANBUL_HOURLY_CSV):
        print("  ERROR: Hourly data file not found.")
        print("  Run:   python scripts/fetch_istanbul_ibb.py")
        sys.exit(1)

    source = CSVSource(config.ISTANBUL_HOURLY_CSV)
    dataset = AirQualityDataset(source, use_daily=True, particulate_mode="pm10").build()
    print(dataset.summary())
    print()

    train = dataset.splits["train"]
    val   = dataset.splits.get("val")
    test  = dataset.splits["test"]

    print("  Running calibration pipeline...")
    cal = run_full_calibration(train, val, build_lookup=True, lookup_size=101, verbose=True)
    fis: MamdaniFIS   = cal["fis"]
    crisp: CrispEPA   = cal["crisp"]
    cal_cfg: dict     = cal["config"]
    pcol: str         = cal_cfg["particulate_col"]

    m_test = compute_pair_metrics(fis, crisp, test[pcol].values, test["no2"].values, verbose=False)
    print(
        f"\n  Calibrated FIS test  |  mode={cal_cfg['particulate_mode'].upper()}"
        f"  acc={m_test['accuracy']*100:.2f}%"
        f"  kappa={m_test['kappa']:.4f}"
        f"  F1={m_test.get('macro_f1',0):.4f}"
    )
    print()

    # ------------------------------------------------------------------
    # Extended benchmark (30 test cases)
    # ------------------------------------------------------------------
    separator("EXTENDED BENCHMARK — 30 TEST CASES")
    ev = Evaluator(fis, crisp)
    ext_bench_df = ev.run_extended_benchmark(verbose=True)
    ext_bench_df.to_csv(os.path.join(config.RESULTS_DIR, "extended_benchmark.csv"), index=False)
    print()

    # ------------------------------------------------------------------
    # Gaussian MF FIS comparison
    # ------------------------------------------------------------------
    separator("GAUSSIAN vs TRIANGULAR MEMBERSHIP FUNCTIONS")
    print()
    print("  Building Gaussian MF FIS (PM10 mode, same 25-rule base)...")
    gauss_fis = GaussianMamdaniFIS(resolution=1001, particulate_mode="pm10")
    gauss_fis.set_bias_correction(fis.bias_correction)
    gauss_fis.set_category_thresholds(fis.category_thresholds)
    gauss_fis.build_lookup(grid_size=71, verbose=True)

    # Evaluation on 2000 random samples
    rng = np.random.default_rng(99)
    rnd_pm10 = rng.uniform(0, 500, 2000)
    rnd_no2  = rng.uniform(0, 200, 2000)
    m_tri   = compute_pair_metrics(fis,       crisp, rnd_pm10, rnd_no2, verbose=False)
    m_gauss = compute_pair_metrics(gauss_fis, crisp, rnd_pm10, rnd_no2, verbose=False)

    print(
        f"  Triangular MF FIS  :  acc={m_tri['accuracy']*100:.2f}%  "
        f"kappa={m_tri['kappa']:.4f}  F1={m_tri['macro_f1']:.4f}  RMSE={m_tri['rmse']:.4f}"
    )
    print(
        f"  Gaussian   MF FIS  :  acc={m_gauss['accuracy']*100:.2f}%  "
        f"kappa={m_gauss['kappa']:.4f}  F1={m_gauss['macro_f1']:.4f}  RMSE={m_gauss['rmse']:.4f}"
    )

    m_gauss_real = compute_pair_metrics(
        gauss_fis, crisp, test[pcol].values, test["no2"].values, verbose=False
    )
    print(
        f"  Gaussian FIS (Istanbul test) :  acc={m_gauss_real['accuracy']*100:.2f}%  "
        f"kappa={m_gauss_real['kappa']:.4f}  F1={m_gauss_real.get('macro_f1',0):.4f}"
    )
    print()

    # ------------------------------------------------------------------
    # Defuzzification method comparison
    # ------------------------------------------------------------------
    separator("DEFUZZIFICATION METHOD COMPARISON")
    print()
    rng2     = np.random.default_rng(42)
    d_pm10   = rng2.uniform(0, 500, 500)
    d_no2    = rng2.uniform(0, 200, 500)
    _, d_cats = crisp.evaluate_batch(d_pm10, d_no2)
    d_idx    = np.array([_label_to_idx(c) for c in d_cats])

    print("  Building 5 FIS variants (centroid / bisector / MoM / SoM / LoM)...", flush=True)
    defuzz_cmp     = DefuzzificationComparison(
        particulate_mode="pm10",
        resolution=config.CALIBRATION_FIS_RESOLUTION,
    )
    defuzz_results = defuzz_cmp.compare_all(d_pm10, d_no2, d_idx, verbose=True)
    print()

    # ------------------------------------------------------------------
    # Three-input FIS  (PM10 + NO2 + SO2)
    # ------------------------------------------------------------------
    separator("THREE-INPUT FIS  (PM10 + NO2 + SO2,  125 rules)")
    print()
    print("  Building 3-input FIS lookup grid (21x21x21)...")
    three_fis = ThreeInputFIS(particulate_mode="pm10", resolution=501)
    three_fis.build_lookup(grid_size=21, verbose=True)

    # Synthetic SO2 correlated with PM10 (both from combustion sources)
    rng3     = np.random.default_rng(7)
    syn_pm10 = rng3.uniform(0, 500, 1000)
    syn_no2  = rng3.uniform(0, 200, 1000)
    syn_so2  = np.clip(syn_pm10 * 0.8 + rng3.normal(0, 40, 1000), 0, 1000)

    (three_vals, three_cats) = three_fis.evaluate_batch(syn_pm10, syn_no2, syn_so2)
    (two_vals,   two_cats)   = fis.evaluate_batch(syn_pm10, syn_no2)
    (_, crisp_syn_cats)      = crisp.evaluate_batch(syn_pm10, syn_no2)

    from sklearn.metrics import accuracy_score as _acc_score
    crisp_syn_idx = np.array([_label_to_idx(c) for c in crisp_syn_cats])
    three_idx     = np.array([_label_to_idx(c) for c in three_cats])
    two_idx       = np.array([_label_to_idx(c) for c in two_cats])
    valid         = np.isfinite(three_vals) & np.isfinite(two_vals)

    acc_two   = float(_acc_score(crisp_syn_idx[valid], two_idx[valid]))
    acc_three = float(_acc_score(crisp_syn_idx[valid], three_idx[valid]))
    print(f"  2-input  (PM10+NO2)      :  acc={acc_two*100:.2f}%  vs Crisp EPA")
    print(f"  3-input  (PM10+NO2+SO2)  :  acc={acc_three*100:.2f}%  vs Crisp EPA")
    print(
        f"  Agreement (2-input vs 3-input) : "
        f"{float(_acc_score(two_idx[valid], three_idx[valid]))*100:.2f}%"
    )

    p0, n0, s0 = 120.0, 80.0, 150.0
    v3, c3 = three_fis.evaluate(p0, n0, s0)
    v2, c2 = fis.evaluate(p0, n0)
    print(f"\n  Sample (PM10={p0}, NO2={n0}, SO2={s0}):")
    print(f"    2-input  ->  AQI={v2:.4f}  ->  '{c2}'")
    print(f"    3-input  ->  AQI={v3:.4f}  ->  '{c3}'")
    print()

    # ------------------------------------------------------------------
    # ML method comparison
    # ------------------------------------------------------------------
    separator("ML METHOD COMPARISON  (9 classifiers vs Mamdani FIS)")
    print()
    print(f"  Train: {len(train):,} daily samples  |  Test: {len(test):,} daily samples")
    print()

    (_, train_crisp_cats) = crisp.evaluate_batch(train[pcol].values, train["no2"].values)
    print("  Fitting classifiers on training labels...")
    ml = MLComparison()
    ml.fit(train[pcol].values, train["no2"].values, train_crisp_cats, verbose=True)

    (_, test_crisp_cats) = crisp.evaluate_batch(test[pcol].values, test["no2"].values)
    test_crisp_idx = np.array([_label_to_idx(c) for c in test_crisp_cats])

    print("\n  Evaluating on test set...")
    ml_results = ml.evaluate(test[pcol].values, test["no2"].values, test_crisp_idx, verbose=True)
    for name in ml_results:
        ml_results[name]["_ref"] = test_crisp_idx

    print()
    ml.print_summary(fis_metrics=m_test)
    summary_df = ml.get_summary_df()
    summary_df.to_csv(
        os.path.join(config.RESULTS_DIR, "ml_comparison_results.csv"), index=False
    )
    print()

    # ------------------------------------------------------------------
    # Seasonal analysis
    # ------------------------------------------------------------------
    separator("SEASONAL ANALYSIS  (Winter / Spring / Summer / Autumn)")
    print()
    full_daily    = dataset.daily
    sea_analyzer  = SeasonalAnalyzer(fis, crisp)
    seasonal_df   = sea_analyzer.run(full_daily, particulate_col=pcol, verbose=True)
    seasonal_df.to_csv(os.path.join(config.RESULTS_DIR, "seasonal_analysis.csv"), index=False)
    print()

    # ------------------------------------------------------------------
    # Station-level analysis
    # ------------------------------------------------------------------
    separator("STATION-LEVEL ANALYSIS  (22 IBB Monitoring Stations)")
    print()
    station_df = StationAnalyzer(fis, crisp).run(
        full_daily, particulate_col=pcol, min_samples=30, verbose=True
    )
    station_df.to_csv(os.path.join(config.RESULTS_DIR, "station_analysis.csv"), index=False)
    if len(station_df) > 0:
        best  = station_df.iloc[0]
        worst = station_df.iloc[-1]
        print(
            f"\n  Best  station : {best['station_id']}  "
            f"acc={best['accuracy']*100:.2f}%  kappa={best['kappa']:.4f}"
        )
        print(
            f"  Worst station : {worst['station_id']}  "
            f"acc={worst['accuracy']*100:.2f}%  kappa={worst['kappa']:.4f}"
        )
    print()

    # ------------------------------------------------------------------
    # Year-over-year trend
    # ------------------------------------------------------------------
    separator("YEAR-OVER-YEAR ACCURACY TREND  (2020-2026)")
    print()
    yearly_df = sea_analyzer.run_yearly(full_daily, particulate_col=pcol, verbose=True)
    yearly_df.to_csv(os.path.join(config.RESULTS_DIR, "yearly_analysis.csv"), index=False)
    print()

    # ------------------------------------------------------------------
    # Figure generation
    # ------------------------------------------------------------------
    separator("FIGURE GENERATION")
    print()
    viz = AdvancedVisualizer(output_dir=config.RESULTS_DIR)

    viz.plot_method_comparison_bars(m_test, ml_results)
    viz.plot_radar_comparison(m_test, ml_results, top_n=5)
    viz.plot_mf_type_comparison(fis, gauss_fis, pollutant="pm10")
    viz.plot_defuzz_comparison(defuzz_results)
    viz.plot_seasonal_analysis(seasonal_df)
    viz.plot_station_accuracy(station_df, top_n=15)
    viz.plot_extended_benchmark(ext_bench_df)
    viz.plot_yearly_trend(yearly_df)

    m_test_full = compute_pair_metrics(
        fis, crisp, test[pcol].values, test["no2"].values, verbose=False
    )
    viz.plot_per_class_heatmap(m_test_full, ml_results, metric="recall")
    viz.plot_mf_type_scatter(m_test, m_gauss_real)
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    separator("RESULTS SUMMARY")
    print()
    print("  ┌─ MAMDANI FIS (calibrated, Istanbul test set) ─────────────────┐")
    kl = "Substantial" if m_test["kappa"] >= 0.6 else "Moderate" if m_test["kappa"] >= 0.4 else "Fair"
    print(f"  │  Accuracy  : {m_test['accuracy']*100:7.2f}%                               │")
    print(f"  │  Cohen's κ : {m_test['kappa']:7.4f}  [{kl}]                   │")
    print(f"  │  Macro-F1  : {m_test.get('macro_f1',0):7.4f}                               │")
    print(f"  │  RMSE      : {m_test['rmse']:7.4f}                               │")
    print("  └──────────────────────────────────────────────────────────────┘")
    print()
    print("  ┌─ GAUSSIAN MF FIS (Istanbul test set) ─────────────────────────┐")
    print(f"  │  Accuracy  : {m_gauss_real['accuracy']*100:7.2f}%                               │")
    print(f"  │  Cohen's κ : {m_gauss_real['kappa']:7.4f}                               │")
    print("  └──────────────────────────────────────────────────────────────┘")
    print()
    print("  ┌─ DEFUZZIFICATION COMPARISON (500 random samples) ─────────────┐")
    for method, res in defuzz_results.items():
        lbl = res["label"][:30]
        print(f"  │  {lbl:<30} acc={res['accuracy']*100:6.2f}%  RMSE={res['rmse']:.4f}  │")
    print("  └──────────────────────────────────────────────────────────────┘")
    print()
    print("  ┌─ THREE-INPUT FIS (PM10 + NO2 + SO2) ──────────────────────────┐")
    print(f"  │  2-input acc : {acc_two*100:6.2f}%  (PM10+NO2 vs Crisp EPA)        │")
    print(f"  │  3-input acc : {acc_three*100:6.2f}%  (PM10+NO2+SO2 vs Crisp)      │")
    print("  └──────────────────────────────────────────────────────────────┘")
    print()
    print("  ┌─ ML METHOD COMPARISON (Istanbul test set) ─────────────────────┐")
    print(f"  │  {'Method':<22}  {'Acc':>7}  {'Kappa':>7}  {'F1':>7}  {'RMSE':>6}  │")
    print("  │  " + "-" * 56 + "  │")
    print(
        f"  │  {'Mamdani FIS (calib.)':<22}  "
        f"{m_test['accuracy']*100:>6.2f}%  {m_test['kappa']:>7.4f}  "
        f"{m_test.get('macro_f1',0):>7.4f}  {m_test['rmse']:>6.4f}  *** │"
    )
    for _, row in summary_df.iterrows():
        kv = row["Cohen's kappa"]
        print(
            f"  │  {row['Method']:<22}  {row['Accuracy (%)']:>6.2f}%  "
            f"{kv:>7.4f}  {row['Macro-F1']:>7.4f}  "
            f"{row['RMSE']:>6.4f}  │"
        )
    print("  └──────────────────────────────────────────────────────────────┘")
    print()
    print("  ┌─ EXTENDED BENCHMARK (30 test cases) ──────────────────────────┐")
    fc = ext_bench_df["Fuzzy Correct"].mean() * 100
    cc = ext_bench_df["Crisp Correct"].mean() * 100
    print(
        f"  │  Fuzzy : {fc:5.1f}%  ({ext_bench_df['Fuzzy Correct'].sum()}/{len(ext_bench_df)})   "
        f"Crisp : {cc:5.1f}%  ({ext_bench_df['Crisp Correct'].sum()}/{len(ext_bench_df)})  │"
    )
    for grp in ext_bench_df["Group"].unique():
        sub  = ext_bench_df[ext_bench_df["Group"] == grp]
        fa   = sub["Fuzzy Correct"].mean() * 100
        ca   = sub["Crisp Correct"].mean() * 100
        print(f"  │    {grp:<14}: Fuzzy {fa:5.1f}%  Crisp {ca:5.1f}%  (n={len(sub)})           │")
    print("  └──────────────────────────────────────────────────────────────┘")
    print()
    if not seasonal_df.empty:
        print("  ┌─ SEASONAL PERFORMANCE ────────────────────────────────────────┐")
        for _, row in seasonal_df.iterrows():
            print(
                f"  │  {row['season']:<8} : acc={row['accuracy']*100:6.2f}%  "
                f"kappa={row['kappa']:.4f}  F1={row['macro_f1']:.4f}  n={row['n']:>6,}  │"
            )
        print("  └──────────────────────────────────────────────────────────────┘")
        print()
    print(f"  Total runtime : {elapsed:.1f} s")
    print()
    separator()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}", file=sys.stderr, flush=True)
        raise
