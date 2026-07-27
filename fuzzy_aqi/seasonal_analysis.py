from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
)

from .evaluation import _label_to_idx, CATEGORY_LABELS
from .sources.base import COL_DATETIME, COL_STATION_ID


def _get_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Spring"
    elif month in (6, 7, 8):
        return "Summer"
    return "Autumn"


SEASON_ORDER  = ["Winter", "Spring", "Summer", "Autumn"]
SEASON_COLORS = {
    "Winter": "#3498DB",
    "Spring": "#2ECC71",
    "Summer": "#E74C3C",
    "Autumn": "#E67E22",
}


def _compute_metrics(ref: np.ndarray, pred: np.ndarray) -> dict:
    if len(ref) == 0:
        return {}
    labels = list(range(5))
    return {
        "n":        len(ref),
        "accuracy": float(accuracy_score(ref, pred)),
        "kappa":    float(cohen_kappa_score(ref, pred, labels=labels)),
        "macro_f1": float(f1_score(ref, pred, average="macro", labels=labels, zero_division=0)),
        "mcc":      float(matthews_corrcoef(ref, pred)),
        "rmse":     float(np.sqrt(mean_squared_error(ref.astype(float), pred.astype(float)))),
        "mae":      float(mean_absolute_error(ref.astype(float), pred.astype(float))),
    }


class SeasonalAnalyzer:

    def __init__(self, fis, crisp) -> None:
        self.fis   = fis
        self.crisp = crisp

    def run(self, df: pd.DataFrame, particulate_col: str = "pm10", verbose: bool = True) -> pd.DataFrame:
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[COL_DATETIME]):
            df[COL_DATETIME] = pd.to_datetime(df[COL_DATETIME])
        df["_season"] = df[COL_DATETIME].dt.month.map(_get_season)

        p_arr = df[particulate_col].values
        n_arr = df["no2"].values
        (fis_vals, fis_cats)    = self.fis.evaluate_batch(p_arr, n_arr)
        (_, crisp_cats)         = self.crisp.evaluate_batch(p_arr, n_arr)
        fis_idx   = np.array([_label_to_idx(c) for c in fis_cats])
        crisp_idx = np.array([_label_to_idx(c) for c in crisp_cats])

        rows = []
        for season in SEASON_ORDER:
            mask = (df["_season"] == season).values & np.isfinite(fis_vals)
            if mask.sum() == 0:
                continue
            m = _compute_metrics(crisp_idx[mask], fis_idx[mask])
            if not m:
                continue
            rows.append({"season": season, **m})
            if verbose:
                print(
                    f"  {season:<8} n={m['n']:>6,}  "
                    f"acc={m['accuracy']*100:6.2f}%  "
                    f"kappa={m['kappa']:.4f}  "
                    f"F1={m['macro_f1']:.4f}  "
                    f"RMSE={m['rmse']:.4f}"
                )
        return pd.DataFrame(rows)

    def run_yearly(self, df: pd.DataFrame, particulate_col: str = "pm10", verbose: bool = True) -> pd.DataFrame:
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df[COL_DATETIME]):
            df[COL_DATETIME] = pd.to_datetime(df[COL_DATETIME])
        df["_year"] = df[COL_DATETIME].dt.year

        p_arr = df[particulate_col].values
        n_arr = df["no2"].values
        (fis_vals, fis_cats) = self.fis.evaluate_batch(p_arr, n_arr)
        (_, crisp_cats)      = self.crisp.evaluate_batch(p_arr, n_arr)
        fis_idx   = np.array([_label_to_idx(c) for c in fis_cats])
        crisp_idx = np.array([_label_to_idx(c) for c in crisp_cats])

        rows = []
        for yr in sorted(df["_year"].unique()):
            mask = (df["_year"] == yr).values & np.isfinite(fis_vals)
            if mask.sum() == 0:
                continue
            m = _compute_metrics(crisp_idx[mask], fis_idx[mask])
            if not m:
                continue
            rows.append({"year": int(yr), **m})
            if verbose:
                print(
                    f"  {yr}  n={m['n']:>5,}  "
                    f"acc={m['accuracy']*100:6.2f}%  kappa={m['kappa']:.4f}"
                )
        return pd.DataFrame(rows)


class StationAnalyzer:

    def __init__(self, fis, crisp) -> None:
        self.fis   = fis
        self.crisp = crisp

    def run(
        self,
        df: pd.DataFrame,
        particulate_col: str = "pm10",
        min_samples: int = 30,
        verbose: bool = True,
    ) -> pd.DataFrame:
        if COL_STATION_ID not in df.columns:
            raise ValueError(f"DataFrame must contain column '{COL_STATION_ID}'.")

        p_arr = df[particulate_col].values
        n_arr = df["no2"].values
        (fis_vals, fis_cats) = self.fis.evaluate_batch(p_arr, n_arr)
        (_, crisp_cats)      = self.crisp.evaluate_batch(p_arr, n_arr)
        fis_idx   = np.array([_label_to_idx(c) for c in fis_cats])
        crisp_idx = np.array([_label_to_idx(c) for c in crisp_cats])

        rows = []
        for station in sorted(df[COL_STATION_ID].unique()):
            mask = (df[COL_STATION_ID] == station).values & np.isfinite(fis_vals)
            if mask.sum() < min_samples:
                continue
            m = _compute_metrics(crisp_idx[mask], fis_idx[mask])
            if not m:
                continue
            dom_cat = CATEGORY_LABELS[int(np.bincount(crisp_idx[mask], minlength=5).argmax())]
            rows.append({
                "station_id":         station,
                "mean_pm":            round(float(np.nanmean(p_arr[mask])), 2),
                "mean_no2":           round(float(np.nanmean(n_arr[mask])), 2),
                "dominant_crisp_cat": dom_cat,
                **m,
            })

        result = (
            pd.DataFrame(rows)
            .sort_values("accuracy", ascending=False)
            .reset_index(drop=True)
        )
        if verbose:
            print(f"\n  {'Station':<40} {'N':>5}  {'Acc':>7}  {'Kappa':>6}  {'F1':>6}")
            print("  " + "-" * 70)
            for _, row in result.iterrows():
                sid = str(row["station_id"])[:38]
                print(
                    f"  {sid:<40} {row['n']:>5,}  "
                    f"{row['accuracy']*100:>6.2f}%  "
                    f"{row['kappa']:>6.4f}  "
                    f"{row['macro_f1']:>6.4f}"
                )
        return result
