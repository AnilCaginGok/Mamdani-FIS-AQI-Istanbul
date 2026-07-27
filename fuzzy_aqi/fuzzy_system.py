import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
TERM_KEYS = ['Good', 'Moderate', 'USG', 'Unhealthy', 'Very_Unhealthy']
TERM_LABELS = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy']
TERM_COLORS = ['#2ECC71', '#F1C40F', '#E67E22', '#E74C3C', '#8E44AD']
PM25_MF_PARAMS = {'Good': ('trapmf', [0, 0, 10, 12]), 'Moderate': ('trimf', [10, 24, 35]), 'USG': ('trimf', [28, 45, 55]), 'Unhealthy': ('trimf', [50, 103, 150]), 'Very_Unhealthy': ('trapmf', [140, 165, 250, 250])}
PM10_MF_PARAMS = {'Good': ('trapmf', [0, 0, 50, 58]), 'Moderate': ('trimf', [48, 100, 158]), 'USG': ('trimf', [145, 205, 258]), 'Unhealthy': ('trimf', [245, 305, 358]), 'Very_Unhealthy': ('trapmf', [345, 400, 500, 500])}
NO2_MF_PARAMS = {'Good': ('trapmf', [0, 0, 25, 45]), 'Moderate': ('trimf', [35, 65, 95]), 'USG': ('trimf', [80, 110, 140]), 'Unhealthy': ('trimf', [120, 150, 175]), 'Very_Unhealthy': ('trapmf', [170, 190, 200, 200])}
AQI_MF_PARAMS = {'Good': ('trapmf', [0, 0, 0.5, 1.5]), 'Moderate': ('trimf', [0.5, 1.5, 2.5]), 'USG': ('trimf', [1.5, 2.5, 3.5]), 'Unhealthy': ('trimf', [2.5, 3.5, 4.5]), 'Very_Unhealthy': ('trapmf', [3.5, 4.5, 5, 5])}
AQI_CATEGORY_CENTERS = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

def _aqi_value_to_category_index(aqi_val: float, thresholds=None) -> int:
    v = float(aqi_val)
    if thresholds is not None and len(thresholds) >= 4:
        t = thresholds
        if v < t[0]:
            return 0
        if v < t[1]:
            return 1
        if v < t[2]:
            return 2
        if v < t[3]:
            return 3
        return 4
    return int(np.argmin(np.abs(AQI_CATEGORY_CENTERS - v)))

def _add_mfs(variable, params_dict, universe):
    for (term, (mf_type, params)) in params_dict.items():
        if mf_type == 'trimf':
            variable[term] = fuzz.trimf(universe, params)
        elif mf_type == 'trapmf':
            variable[term] = fuzz.trapmf(universe, params)

class MamdaniFIS:

    def __init__(self, resolution: int=1001, particulate_mode: str='pm25'):
        if particulate_mode not in ('pm25', 'pm10'):
            raise ValueError("particulate_mode must be 'pm25' or 'pm10'")
        self.resolution = resolution
        self.particulate_mode = particulate_mode
        self._particulate_key = 'PM25' if particulate_mode == 'pm25' else 'PM10'
        self._particulate_max = 250.0 if particulate_mode == 'pm25' else 500.0
        self.bias_correction = 0.0
        self.category_thresholds = None
        self._lookup = None
        self._lookup_p = None
        self._lookup_n = None
        self._lookup_values = None
        self._build()

    def _clear_lookup(self) -> None:
        self._lookup = None
        self._lookup_p = None
        self._lookup_values = None

    def set_bias_correction(self, bias: float):
        self.bias_correction = float(bias)
        self._clear_lookup()

    def set_category_thresholds(self, thresholds):
        self.category_thresholds = np.asarray(thresholds, dtype=float)
        if len(self.category_thresholds) != 4:
            raise ValueError('category_thresholds must have length 4')

    def _category_index(self, aqi_val: float) -> int:
        return min(4, _aqi_value_to_category_index(aqi_val, self.category_thresholds))

    def build_lookup(self, grid_size: int=101, verbose: bool=False):
        if verbose:
            print(f'    Lookup grid {grid_size}x{grid_size}…', flush=True)
        p_grid = np.linspace(0.0, self._particulate_max, grid_size)
        n_grid = np.linspace(0.0, 200.0, grid_size)
        values = np.zeros((grid_size, grid_size), dtype=float)
        for (i, nv) in enumerate(n_grid):
            for (j, pv) in enumerate(p_grid):
                values[i, j] = self._simulate_raw(pv, nv)
        try:
            from scipy.interpolate import RegularGridInterpolator
            self._lookup = RegularGridInterpolator((n_grid, p_grid), values, bounds_error=False, fill_value=None)
        except ImportError:
            self._lookup = None
            self._lookup_values = values
        self._lookup_p = p_grid
        self._lookup_n = n_grid
        if verbose:
            print('    Lookup ready.', flush=True)

    def _simulate_raw(self, particulate: float, no2: float) -> float:
        self._sim.input[self._particulate_key] = float(np.clip(particulate, 0.0, self._particulate_max))
        self._sim.input['NO2'] = float(np.clip(no2, 0.0, 200.0))
        self._sim.compute()
        return float(np.clip(float(self._sim.output['AQI']) - self.bias_correction, 0.0, 5.0))

    def _lookup_value(self, particulate: float, no2: float) -> float:
        if self._lookup is not None and hasattr(self._lookup, '__call__'):
            v = float(self._lookup([no2, particulate]))
            if np.isnan(v):
                return self._simulate_raw(particulate, no2)
            return float(np.clip(v, 0.0, 5.0))
        if self._lookup_p is not None:
            j = int(np.clip(np.searchsorted(self._lookup_p, particulate), 0, len(self._lookup_p) - 1))
            i = int(np.clip(np.searchsorted(self._lookup_n, no2), 0, len(self._lookup_n) - 1))
            return float(self._lookup_values[i, j])
        return self._simulate_raw(particulate, no2)

    def evaluate(self, particulate: float, no2: float):
        if self._lookup is not None or self._lookup_p is not None:
            aqi_val = self._lookup_value(particulate, no2)
        else:
            aqi_val = self._simulate_raw(particulate, no2)
        cat_idx = self._category_index(aqi_val)
        return (aqi_val, TERM_LABELS[cat_idx])

    def evaluate_batch(self, particulate_arr, no2_arr, show_progress: bool=False):
        particulate_arr = np.asarray(particulate_arr, dtype=float)
        no2_arr = np.asarray(no2_arr, dtype=float)
        n = len(particulate_arr)
        if self._lookup is not None:
            points = np.column_stack([no2_arr, particulate_arr])
            aqi_values = np.asarray(self._lookup(points), dtype=float).ravel()
            aqi_values = np.clip(aqi_values, 0.0, 5.0)
            bad = ~np.isfinite(aqi_values)
            if bad.any():
                for k in np.where(bad)[0]:
                    aqi_values[k] = self._simulate_raw(particulate_arr[k], no2_arr[k])
            categories = [TERM_LABELS[self._category_index(v)] for v in aqi_values]
            if show_progress:
                print(f'    … {n:,} samples (lookup batch)', flush=True)
            return (aqi_values, categories)
        if self._lookup_p is not None and self._lookup_values is not None:
            j_idx = np.searchsorted(self._lookup_p, particulate_arr)
            j_idx = np.clip(j_idx, 0, len(self._lookup_p) - 1)
            i_idx = np.searchsorted(self._lookup_n, no2_arr)
            i_idx = np.clip(i_idx, 0, len(self._lookup_n) - 1)
            aqi_values = self._lookup_values[i_idx, j_idx].astype(float)
            aqi_values = np.clip(aqi_values, 0.0, 5.0)
            categories = [TERM_LABELS[self._category_index(v)] for v in aqi_values]
            if show_progress:
                print(f'    … {n:,} samples (grid batch)', flush=True)
            return (aqi_values, categories)
        aqi_values = np.empty(n)
        categories = []
        step = max(1, n // 20)
        for (k, (p, val_n)) in enumerate(zip(particulate_arr, no2_arr)):
            try:
                (val, cat) = self.evaluate(p, val_n)
            except Exception:
                (val, cat) = (np.nan, 'Unknown')
            aqi_values[k] = val
            categories.append(cat)
            if show_progress and (k + 1) % step == 0:
                print(f'    … {k + 1:,}/{n:,} ({100 * (k + 1) / n:.0f}%)', flush=True)
        return (aqi_values, categories)

    def get_rule_matrix(self):
        mat = np.zeros((5, 5), dtype=int)
        for i in range(5):
            for j in range(5):
                mat[i, j] = max(i, j)
        return mat

    def get_mf_values(self):
        out = {'no2': {'universe': self._no2.universe, 'mfs': {t: self._no2[t].mf for t in TERM_KEYS}}, 'aqi': {'universe': self._aqi.universe, 'mfs': {t: self._aqi[t].mf for t in TERM_KEYS}}}
        key = 'pm25' if self.particulate_mode == 'pm25' else 'pm10'
        out[key] = {'universe': self._particulate.universe, 'mfs': {t: self._particulate[t].mf for t in TERM_KEYS}}
        return out

    def _build(self):
        if self.particulate_mode == 'pm25':
            p_u = np.linspace(0, 250, self.resolution)
            p_params = PM25_MF_PARAMS
            p_label = 'PM25'
        else:
            p_u = np.linspace(0, 500, self.resolution)
            p_params = PM10_MF_PARAMS
            p_label = 'PM10'
        no2_u = np.linspace(0, 200, self.resolution)
        aqi_u = np.linspace(0, 5, self.resolution)
        self._particulate = ctrl.Antecedent(p_u, p_label)
        self._no2 = ctrl.Antecedent(no2_u, 'NO2')
        self._aqi = ctrl.Consequent(aqi_u, 'AQI', defuzzify_method='centroid')
        _add_mfs(self._particulate, p_params, p_u)
        _add_mfs(self._no2, NO2_MF_PARAMS, no2_u)
        _add_mfs(self._aqi, AQI_MF_PARAMS, aqi_u)
        rules = []
        for (i, p_term) in enumerate(TERM_KEYS):
            for (j, n_term) in enumerate(TERM_KEYS):
                out_term = TERM_KEYS[max(i, j)]
                rules.append(ctrl.Rule(antecedent=self._particulate[p_term] & self._no2[n_term], consequent=self._aqi[out_term]))
        self._cs = ctrl.ControlSystem(rules)
        self._sim = ctrl.ControlSystemSimulation(self._cs)
        if self.particulate_mode == 'pm25':
            self._pm25 = self._particulate

    def print_rule_table(self):
        short = ['G', 'M', 'U', 'H', 'VU']
        p_name = 'PM2.5' if self.particulate_mode == 'pm25' else 'PM10'
        col_title = f'{p_name} \\ NO2'
        header = f'{col_title:>14}' + ''.join((f'{s:>6}' for s in short))
        print(header)
        print('-' * (14 + 6 * 5))
        for (i, row_label) in enumerate(short):
            row = f'{row_label:>14}'
            for j in range(5):
                row += f'{short[max(i, j)]:>6}'
            print(row)
