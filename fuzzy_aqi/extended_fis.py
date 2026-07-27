import numpy as np
import skfuzzy as fuzz
from scipy.interpolate import RegularGridInterpolator
from skfuzzy import control as ctrl

from .fuzzy_system import (
    AQI_MF_PARAMS,
    PM10_MF_PARAMS,
    PM25_MF_PARAMS,
    NO2_MF_PARAMS,
    TERM_KEYS,
    TERM_LABELS,
    _add_mfs,
    _aqi_value_to_category_index,
)

# Gaussian MF centres and sigmas derived from the centres of the existing
# triangular/trapezoidal MFs.  Sigma ≈ 1/4 of the original MF full-width.
PM25_GAUSSIAN_PARAMS: dict = {
    "Good":           (5.0,   5.0),
    "Moderate":       (24.0,  8.0),
    "USG":            (45.0,  9.0),
    "Unhealthy":      (103.0, 30.0),
    "Very_Unhealthy": (200.0, 40.0),
}

PM10_GAUSSIAN_PARAMS: dict = {
    "Good":           (27.0,  27.0),
    "Moderate":       (100.0, 38.0),
    "USG":            (205.0, 38.0),
    "Unhealthy":      (305.0, 38.0),
    "Very_Unhealthy": (430.0, 50.0),
}

NO2_GAUSSIAN_PARAMS: dict = {
    "Good":           (12.0,  12.0),
    "Moderate":       (65.0,  18.0),
    "USG":            (110.0, 18.0),
    "Unhealthy":      (148.0, 18.0),
    "Very_Unhealthy": (192.0,  8.0),
}

# SO2 MF parameters (µg/m³) — EPA breakpoints:
#   Good: 0–91.4,  Moderate: 91.5–196.4,  USG: 196.5–484.4,
#   Unhealthy: 484.5–796.2,  VU: 796.3+
SO2_MF_PARAMS: dict = {
    "Good":           ("trapmf", [0, 0, 55, 91]),
    "Moderate":       ("trimf",  [80, 145, 200]),
    "USG":            ("trimf",  [185, 340, 490]),
    "Unhealthy":      ("trimf",  [475, 640, 800]),
    "Very_Unhealthy": ("trapmf", [785, 850, 1000, 1000]),
}

DEFUZZ_METHODS: dict = {
    "centroid":  "Centroid (CoG)",
    "bisector":  "Bisector",
    "mom":       "Mean of Maximum (MoM)",
    "som":       "Smallest of Maximum (SoM)",
    "lom":       "Largest of Maximum (LoM)",
}


def _add_gaussian_mfs(variable, gaussian_params: dict, universe: np.ndarray) -> None:
    for term, (mean, sigma) in gaussian_params.items():
        variable[term] = fuzz.gaussmf(universe, mean, sigma)


# ===========================================================================
class GaussianMamdaniFIS:
    """Mamdani FIS using Gaussian input MFs; rule base identical to MamdaniFIS."""

    def __init__(self, resolution: int = 1001, particulate_mode: str = "pm25") -> None:
        if particulate_mode not in ("pm25", "pm10"):
            raise ValueError("particulate_mode must be 'pm25' or 'pm10'")
        self.resolution = resolution
        self.particulate_mode = particulate_mode
        self._particulate_key = "PM25" if particulate_mode == "pm25" else "PM10"
        self._particulate_max = 250.0 if particulate_mode == "pm25" else 500.0
        self.bias_correction: float = 0.0
        self.category_thresholds = None
        self._lookup = None
        self._lookup_p = None
        self._lookup_n = None
        self._build()

    def _build(self) -> None:
        if self.particulate_mode == "pm25":
            p_u    = np.linspace(0, 250, self.resolution)
            p_gauss = PM25_GAUSSIAN_PARAMS
            p_label = "PM25"
        else:
            p_u    = np.linspace(0, 500, self.resolution)
            p_gauss = PM10_GAUSSIAN_PARAMS
            p_label = "PM10"
        no2_u = np.linspace(0, 200, self.resolution)
        aqi_u = np.linspace(0, 5, self.resolution)
        self._particulate = ctrl.Antecedent(p_u, p_label)
        self._no2         = ctrl.Antecedent(no2_u, "NO2")
        self._aqi         = ctrl.Consequent(aqi_u, "AQI", defuzzify_method="centroid")
        _add_gaussian_mfs(self._particulate, p_gauss, p_u)
        _add_gaussian_mfs(self._no2, NO2_GAUSSIAN_PARAMS, no2_u)
        _add_mfs(self._aqi, AQI_MF_PARAMS, aqi_u)
        rules = []
        for i, p_term in enumerate(TERM_KEYS):
            for j, n_term in enumerate(TERM_KEYS):
                out_term = TERM_KEYS[max(i, j)]
                rules.append(ctrl.Rule(
                    antecedent=self._particulate[p_term] & self._no2[n_term],
                    consequent=self._aqi[out_term],
                ))
        self._cs  = ctrl.ControlSystem(rules)
        self._sim = ctrl.ControlSystemSimulation(self._cs)

    def set_bias_correction(self, bias: float) -> None:
        self.bias_correction = float(bias)
        self._lookup = None
        self._lookup_p = None

    def set_category_thresholds(self, thresholds) -> None:
        self.category_thresholds = np.asarray(thresholds, dtype=float)

    def _category_index(self, aqi_val: float) -> int:
        return min(4, _aqi_value_to_category_index(aqi_val, self.category_thresholds))

    def _simulate_raw(self, particulate: float, no2: float) -> float:
        self._sim.input[self._particulate_key] = float(np.clip(particulate, 0.0, self._particulate_max))
        self._sim.input["NO2"] = float(np.clip(no2, 0.0, 200.0))
        self._sim.compute()
        return float(np.clip(float(self._sim.output["AQI"]) - self.bias_correction, 0.0, 5.0))

    def evaluate(self, particulate: float, no2: float):
        if self._lookup is not None:
            aqi_val = float(np.clip(self._lookup([[no2, particulate]])[0], 0.0, 5.0))
        else:
            aqi_val = self._simulate_raw(particulate, no2)
        return (aqi_val, TERM_LABELS[self._category_index(aqi_val)])

    def evaluate_batch(self, particulate_arr, no2_arr, show_progress: bool = False):
        particulate_arr = np.asarray(particulate_arr, dtype=float)
        no2_arr         = np.asarray(no2_arr, dtype=float)
        n = len(particulate_arr)
        if self._lookup is not None:
            pts = np.column_stack([no2_arr, particulate_arr])
            aqi_values = np.clip(self._lookup(pts).ravel(), 0.0, 5.0)
            categories = [TERM_LABELS[self._category_index(v)] for v in aqi_values]
            return (aqi_values, categories)
        aqi_values = np.empty(n)
        categories = []
        step = max(1, n // 10)
        for k, (p, nv) in enumerate(zip(particulate_arr, no2_arr)):
            try:
                val = self._simulate_raw(p, nv)
            except Exception:
                val = np.nan
            aqi_values[k] = val
            categories.append(TERM_LABELS[self._category_index(val)])
            if show_progress and (k + 1) % step == 0:
                print(f"    ... {k+1:,}/{n:,} ({100*(k+1)/n:.0f}%)", flush=True)
        return (aqi_values, categories)

    def build_lookup(self, grid_size: int = 101, verbose: bool = False) -> None:
        if verbose:
            print(f"    Gaussian FIS lookup {grid_size}x{grid_size}...", flush=True)
        p_grid = np.linspace(0.0, self._particulate_max, grid_size)
        n_grid = np.linspace(0.0, 200.0, grid_size)
        values = np.zeros((grid_size, grid_size), dtype=float)
        for i, nv in enumerate(n_grid):
            for j, pv in enumerate(p_grid):
                values[i, j] = self._simulate_raw(pv, nv)
        self._lookup = RegularGridInterpolator(
            (n_grid, p_grid), values, bounds_error=False, fill_value=None
        )
        self._lookup_p = p_grid
        self._lookup_n = n_grid
        if verbose:
            print("    Gaussian FIS lookup ready.", flush=True)

    def get_mf_values(self) -> dict:
        key = "pm25" if self.particulate_mode == "pm25" else "pm10"
        return {
            key: {"universe": self._particulate.universe, "mfs": {t: self._particulate[t].mf for t in TERM_KEYS}},
            "no2": {"universe": self._no2.universe, "mfs": {t: self._no2[t].mf for t in TERM_KEYS}},
            "aqi": {"universe": self._aqi.universe, "mfs": {t: self._aqi[t].mf for t in TERM_KEYS}},
        }


# ===========================================================================
class DefuzzificationComparison:
    """Compare five defuzzification strategies on the standard triangular FIS."""

    def __init__(self, particulate_mode: str = "pm25", resolution: int = 501) -> None:
        self.particulate_mode = particulate_mode
        self.resolution       = resolution
        self._particulate_max = 250.0 if particulate_mode == "pm25" else 500.0
        self._sims: dict      = {}
        self._build_all()

    def _build_all(self) -> None:
        for method in DEFUZZ_METHODS:
            if self.particulate_mode == "pm25":
                p_u, p_params, p_label = np.linspace(0, 250, self.resolution), PM25_MF_PARAMS, "PM25"
            else:
                p_u, p_params, p_label = np.linspace(0, 500, self.resolution), PM10_MF_PARAMS, "PM10"
            no2_u = np.linspace(0, 200, self.resolution)
            aqi_u = np.linspace(0, 5,   self.resolution)
            part = ctrl.Antecedent(p_u,   p_label)
            no2  = ctrl.Antecedent(no2_u, "NO2")
            aqi  = ctrl.Consequent(aqi_u, "AQI", defuzzify_method=method)
            _add_mfs(part, p_params,   p_u)
            _add_mfs(no2,  NO2_MF_PARAMS, no2_u)
            _add_mfs(aqi,  AQI_MF_PARAMS, aqi_u)
            rules = []
            for i, p_term in enumerate(TERM_KEYS):
                for j, n_term in enumerate(TERM_KEYS):
                    out_term = TERM_KEYS[max(i, j)]
                    rules.append(ctrl.Rule(
                        antecedent=part[p_term] & no2[n_term],
                        consequent=aqi[out_term],
                    ))
            cs = ctrl.ControlSystem(rules)
            self._sims[method] = {
                "sim":      ctrl.ControlSystemSimulation(cs),
                "label":    DEFUZZ_METHODS[method],
                "part_key": p_label,
                "lookup":   None,
                "lookup_p": None,
                "lookup_n": None,
                "lookup_values": None,
            }

    def _ensure_method_lookup(self, method: str, grid_size: int = 41, verbose: bool = False) -> None:
        entry = self._sims[method]
        if entry.get("lookup") is not None or entry.get("lookup_p") is not None:
            return
        sim, p_key = entry["sim"], entry["part_key"]
        if verbose:
            print(f"    Lookup {DEFUZZ_METHODS[method]:<28} ({grid_size}x{grid_size})…", flush=True)
        p_grid = np.linspace(0.0, self._particulate_max, grid_size)
        n_grid = np.linspace(0.0, 200.0, grid_size)
        values = np.zeros((grid_size, grid_size), dtype=float)
        for i, nv in enumerate(n_grid):
            for j, pv in enumerate(p_grid):
                try:
                    sim.input[p_key] = float(pv)
                    sim.input["NO2"] = float(nv)
                    sim.compute()
                    values[i, j] = float(np.clip(float(sim.output["AQI"]), 0.0, 5.0))
                except Exception:
                    values[i, j] = np.nan
        try:
            entry["lookup"] = RegularGridInterpolator(
                (n_grid, p_grid), values, bounds_error=False, fill_value=None
            )
        except Exception:
            entry["lookup"] = None
            entry["lookup_values"] = values
        entry["lookup_p"] = p_grid
        entry["lookup_n"] = n_grid

    def evaluate_method(
        self,
        method: str,
        particulate_arr,
        no2_arr,
        category_thresholds=None,
        verbose: bool = False,
    ):
        self._ensure_method_lookup(method, grid_size=41, verbose=verbose)
        entry = self._sims[method]
        sim, p_key = entry["sim"], entry["part_key"]
        particulate_arr = np.asarray(particulate_arr, dtype=float)
        no2_arr = np.asarray(no2_arr, dtype=float)
        n = len(particulate_arr)
        aqi_values = np.empty(n, dtype=float)
        if entry.get("lookup") is not None:
            pts = np.column_stack([no2_arr, particulate_arr])
            aqi_values = np.asarray(entry["lookup"](pts), dtype=float).ravel()
            aqi_values = np.clip(aqi_values, 0.0, 5.0)
            bad = ~np.isfinite(aqi_values)
            if bad.any():
                for k in np.where(bad)[0]:
                    try:
                        sim.input[p_key] = float(np.clip(particulate_arr[k], 0.0, self._particulate_max))
                        sim.input["NO2"] = float(np.clip(no2_arr[k], 0.0, 200.0))
                        sim.compute()
                        aqi_values[k] = float(np.clip(float(sim.output["AQI"]), 0.0, 5.0))
                    except Exception:
                        aqi_values[k] = np.nan
        elif entry.get("lookup_p") is not None and entry.get("lookup_values") is not None:
            j_idx = np.searchsorted(entry["lookup_p"], particulate_arr)
            j_idx = np.clip(j_idx, 0, len(entry["lookup_p"]) - 1)
            i_idx = np.searchsorted(entry["lookup_n"], no2_arr)
            i_idx = np.clip(i_idx, 0, len(entry["lookup_n"]) - 1)
            aqi_values = entry["lookup_values"][i_idx, j_idx].astype(float)
            aqi_values = np.clip(aqi_values, 0.0, 5.0)
        else:
            categories = []
            step = max(1, n // 10)
            for k, (p, nv) in enumerate(zip(particulate_arr, no2_arr)):
                try:
                    sim.input[p_key] = float(np.clip(p, 0.0, self._particulate_max))
                    sim.input["NO2"] = float(np.clip(nv, 0.0, 200.0))
                    sim.compute()
                    val = float(np.clip(float(sim.output["AQI"]), 0.0, 5.0))
                except Exception:
                    val = np.nan
                aqi_values[k] = val
                if verbose and (k + 1) % step == 0:
                    print(f"      … {k + 1:,}/{n:,}", flush=True)
        categories = []
        for val in aqi_values:
            cat_idx = min(4, _aqi_value_to_category_index(val, category_thresholds))
            categories.append(TERM_LABELS[cat_idx])
        return (aqi_values, categories)

    def compare_all(self, particulate_arr, no2_arr, reference_idx, verbose: bool = True) -> dict:
        from sklearn.metrics import accuracy_score, mean_squared_error
        ref     = np.asarray(reference_idx, dtype=int)
        results = {}
        if verbose:
            print(f"\n  Defuzzification comparison on {len(ref):,} samples...", flush=True)
            print("  Pre-building lookup grids for 5 defuzzification methods…", flush=True)
        for method in DEFUZZ_METHODS:
            self._ensure_method_lookup(method, grid_size=41, verbose=verbose)
        for method in DEFUZZ_METHODS:
            if verbose:
                print(f"    Evaluating {DEFUZZ_METHODS[method]}…", flush=True)
            aqi_vals, cats = self.evaluate_method(method, particulate_arr, no2_arr, verbose=False)
            from .evaluation import _label_to_idx
            pred_idx = np.array([_label_to_idx(c) for c in cats])
            valid    = np.isfinite(aqi_vals)
            acc      = float(accuracy_score(ref[valid], pred_idx[valid]))
            rmse     = float(np.sqrt(mean_squared_error(
                ref[valid].astype(float), pred_idx[valid].astype(float)
            )))
            results[method] = {
                "label":    DEFUZZ_METHODS[method],
                "accuracy": acc,
                "rmse":     rmse,
                "aqi_vals": aqi_vals,
                "pred_idx": pred_idx,
            }
            if verbose:
                print(
                    f"    {DEFUZZ_METHODS[method]:<28}: "
                    f"acc={acc*100:.2f}%  RMSE={rmse:.4f}"
                )
        return results


# ===========================================================================
class ThreeInputFIS:
    """
    Three-input Mamdani FIS: PM (pm10 or pm25) + NO2 + SO2.
    Rule base: 5^3 = 125 rules, output = max(PM term, NO2 term, SO2 term).
    Uses a manual inference engine with optional 3-D lookup grid.
    """

    def __init__(self, particulate_mode: str = "pm10", resolution: int = 501) -> None:
        if particulate_mode not in ("pm25", "pm10"):
            raise ValueError("particulate_mode must be 'pm25' or 'pm10'")
        self.particulate_mode = particulate_mode
        self.resolution       = resolution
        self._p_max           = 250.0 if particulate_mode == "pm25" else 500.0
        self.category_thresholds = None
        self._lookup: RegularGridInterpolator | None = None
        self._build_universes()
        self._build_mf_arrays()

    def _build_universes(self) -> None:
        self._p_u   = np.linspace(0.0, self._p_max, self.resolution)
        self._n_u   = np.linspace(0.0, 200.0,      self.resolution)
        self._s_u   = np.linspace(0.0, 1000.0,     self.resolution)
        self._aqi_u = np.linspace(0.0, 5.0,        self.resolution)

    def _build_mf_arrays(self) -> None:
        p_params      = PM25_MF_PARAMS if self.particulate_mode == "pm25" else PM10_MF_PARAMS
        self._p_mfs   = self._build_mf_set(p_params,    self._p_u)
        self._n_mfs   = self._build_mf_set(NO2_MF_PARAMS, self._n_u)
        self._s_mfs   = self._build_mf_set(SO2_MF_PARAMS, self._s_u)
        self._aqi_mfs = self._build_mf_set(AQI_MF_PARAMS, self._aqi_u)

    @staticmethod
    def _build_mf_set(params: dict, universe: np.ndarray) -> dict:
        mfs = {}
        for term, (mf_type, p) in params.items():
            mfs[term] = fuzz.trimf(universe, p) if mf_type == "trimf" else fuzz.trapmf(universe, p)
        return mfs

    def _infer_single(self, p_val: float, n_val: float, s_val: float) -> float:
        p_grades = {t: float(fuzz.interp_membership(self._p_u,   self._p_mfs[t],   p_val)) for t in TERM_KEYS}
        n_grades = {t: float(fuzz.interp_membership(self._n_u,   self._n_mfs[t],   n_val)) for t in TERM_KEYS}
        s_grades = {t: float(fuzz.interp_membership(self._s_u,   self._s_mfs[t],   s_val)) for t in TERM_KEYS}
        agg = np.zeros(self.resolution, dtype=float)
        for i, pt in enumerate(TERM_KEYS):
            for j, nt in enumerate(TERM_KEYS):
                for k, st in enumerate(TERM_KEYS):
                    strength = min(p_grades[pt], n_grades[nt], s_grades[st])
                    if strength < 1e-9:
                        continue
                    out_term = TERM_KEYS[max(i, j, k)]
                    agg = np.maximum(agg, np.minimum(self._aqi_mfs[out_term], strength))
        total = np.trapz(agg, self._aqi_u)
        if total < 1e-9:
            return 2.5
        return float(np.clip(np.trapz(agg * self._aqi_u, self._aqi_u) / total, 0.0, 5.0))

    def build_lookup(self, grid_size: int = 21, verbose: bool = True) -> None:
        if verbose:
            total = grid_size ** 3
            print(f"    3-Input FIS lookup {grid_size}^3 = {total:,} pts...", flush=True)
        p_grid = np.linspace(0.0, self._p_max, grid_size)
        n_grid = np.linspace(0.0, 200.0,      grid_size)
        s_grid = np.linspace(0.0, 1000.0,     grid_size)
        values = np.zeros((grid_size, grid_size, grid_size), dtype=float)
        total  = grid_size ** 3
        count  = 0
        for i, nv in enumerate(n_grid):
            for j, pv in enumerate(p_grid):
                for k, sv in enumerate(s_grid):
                    values[i, j, k] = self._infer_single(pv, nv, sv)
                    count += 1
            if verbose and (i + 1) % max(1, grid_size // 5) == 0:
                print(f"    ... {count:,}/{total:,}", flush=True)
        self._lookup = RegularGridInterpolator(
            (n_grid, p_grid, s_grid), values, bounds_error=False, fill_value=None
        )
        if verbose:
            print("    3-Input FIS lookup ready.", flush=True)

    def _category_index(self, aqi_val: float) -> int:
        return min(4, _aqi_value_to_category_index(aqi_val, self.category_thresholds))

    def evaluate(self, p: float, no2: float, so2: float):
        p   = float(np.clip(p,   0.0, self._p_max))
        no2 = float(np.clip(no2, 0.0, 200.0))
        so2 = float(np.clip(so2, 0.0, 1000.0))
        if self._lookup is not None:
            aqi_val = float(np.clip(self._lookup([[no2, p, so2]])[0], 0.0, 5.0))
        else:
            aqi_val = self._infer_single(p, no2, so2)
        return (aqi_val, TERM_LABELS[self._category_index(aqi_val)])

    def evaluate_batch(self, p_arr, no2_arr, so2_arr, show_progress: bool = False):
        p_arr   = np.asarray(p_arr,   dtype=float)
        no2_arr = np.asarray(no2_arr, dtype=float)
        so2_arr = np.asarray(so2_arr, dtype=float)
        n = len(p_arr)
        if self._lookup is not None:
            pts = np.column_stack([
                np.clip(no2_arr, 0.0, 200.0),
                np.clip(p_arr,   0.0, self._p_max),
                np.clip(so2_arr, 0.0, 1000.0),
            ])
            aqi_values = np.clip(self._lookup(pts).ravel(), 0.0, 5.0)
            categories = [TERM_LABELS[self._category_index(v)] for v in aqi_values]
            return (aqi_values, categories)
        aqi_values = np.empty(n)
        categories = []
        step = max(1, n // 10)
        for k, (p, nv, sv) in enumerate(zip(p_arr, no2_arr, so2_arr)):
            try:
                val, cat = self.evaluate(p, nv, sv)
            except Exception:
                val, cat = np.nan, "Unknown"
            aqi_values[k] = val
            categories.append(cat)
            if show_progress and (k + 1) % step == 0:
                print(f"    ... {k+1:,}/{n:,}", flush=True)
        return (aqi_values, categories)
