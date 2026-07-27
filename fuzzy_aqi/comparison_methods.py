import copy

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    matthews_corrcoef,
    mean_squared_error,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .evaluation import _label_to_idx, CATEGORY_LABELS


CLASSIFIERS_CONFIG: dict = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42
    ),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)),
    ]),
    "MLP Neural Net": Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=600,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )),
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=1000, random_state=42,
            multi_class="multinomial", solver="lbfgs", C=1.0,
        )),
    ]),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100, random_state=42, algorithm="SAMME"
    ),
    "k-NN (k=11)": KNeighborsClassifier(n_neighbors=11, weights="distance"),
}


class MLComparison:

    def __init__(self) -> None:
        self.models: dict = {}
        self.results: dict = {}

    def fit(self, particulate_arr, no2_arr, crisp_labels, verbose: bool = True) -> None:
        X = np.column_stack([
            np.asarray(particulate_arr, dtype=float),
            np.asarray(no2_arr, dtype=float),
        ])
        y = np.array([_label_to_idx(c) for c in crisp_labels])
        if verbose:
            print(f"  Training {len(CLASSIFIERS_CONFIG)} classifiers on {len(X):,} samples...")
        for name, clf in CLASSIFIERS_CONFIG.items():
            model = copy.deepcopy(clf)
            model.fit(X, y)
            self.models[name] = model
            if verbose:
                print(f"    OK {name}")

    def evaluate(self, particulate_arr, no2_arr, reference_idx, verbose: bool = True) -> dict:
        X = np.column_stack([
            np.asarray(particulate_arr, dtype=float),
            np.asarray(no2_arr, dtype=float),
        ])
        ref = np.asarray(reference_idx, dtype=int)
        results: dict = {}
        for name, model in self.models.items():
            pred = model.predict(X)
            acc   = float(accuracy_score(ref, pred))
            kappa = float(cohen_kappa_score(ref, pred, labels=list(range(5))))
            f1    = float(f1_score(ref, pred, average="macro", labels=list(range(5)), zero_division=0))
            mcc   = float(matthews_corrcoef(ref, pred))
            rmse  = float(np.sqrt(mean_squared_error(ref.astype(float), pred.astype(float))))
            mae   = float(mean_absolute_error(ref.astype(float), pred.astype(float)))
            conf_mat = confusion_matrix(ref, pred, labels=list(range(5)))
            results[name] = {
                "accuracy": acc, "kappa": kappa, "macro_f1": f1,
                "mcc": mcc, "rmse": rmse, "mae": mae,
                "pred": pred, "conf_matrix": conf_mat,
            }
            if verbose:
                print(
                    f"  {name:<22}: acc={acc*100:.2f}%  "
                    f"kappa={kappa:.4f}  F1={f1:.4f}  RMSE={rmse:.4f}"
                )
        self.results = results
        return results

    def get_summary_df(self) -> pd.DataFrame:
        rows = []
        for name, res in self.results.items():
            kappa_label = (
                "Substantial" if res["kappa"] >= 0.6
                else "Moderate"  if res["kappa"] >= 0.4
                else "Fair"      if res["kappa"] >= 0.2
                else "Slight"
            )
            rows.append({
                "Method":         name,
                "Accuracy (%)":   round(res["accuracy"] * 100, 2),
                "Cohen's kappa":  round(res["kappa"], 4),
                "Kappa Label":    kappa_label,
                "Macro-F1":       round(res["macro_f1"], 4),
                "MCC":            round(res["mcc"], 4),
                "RMSE":           round(res["rmse"], 4),
                "MAE":            round(res["mae"], 4),
            })
        return (
            pd.DataFrame(rows)
            .sort_values("Accuracy (%)", ascending=False)
            .reset_index(drop=True)
        )

    def print_summary(self, fis_metrics: dict = None) -> None:
        print("\n" + "=" * 80)
        print("ML vs MAMDANI FIS — COMPARISON TABLE (test set)")
        print("=" * 80)
        header = (
            f"  {'Method':<22}  {'Accuracy':>9}  {'Kappa':>7}  "
            f"{'Macro-F1':>9}  {'MCC':>7}  {'RMSE':>6}  {'MAE':>6}"
        )
        print(header)
        print("  " + "-" * 76)
        if fis_metrics is not None:
            acc  = fis_metrics["accuracy"] * 100
            k    = fis_metrics["kappa"]
            f1   = fis_metrics.get("macro_f1", 0.0)
            mcc  = fis_metrics["mcc"]
            rmse = fis_metrics["rmse"]
            mae  = fis_metrics["mae"]
            print(
                f"  {'Mamdani FIS (calib.)':<22}  {acc:>8.2f}%  "
                f"{k:>7.4f}  {f1:>9.4f}  {mcc:>7.4f}  {rmse:>6.4f}  {mae:>6.4f}  *** (fuzzy)"
            )
        for _, row in self.get_summary_df().iterrows():
            kappa_val = row["Cohen's kappa"]
            print(
                f"  {row['Method']:<22}  {row['Accuracy (%)']:>8.2f}%  "
                f"{kappa_val:>7.4f}  {row['Macro-F1']:>9.4f}  "
                f"{row['MCC']:>7.4f}  {row['RMSE']:>6.4f}  {row['MAE']:>6.4f}"
            )
        print()
