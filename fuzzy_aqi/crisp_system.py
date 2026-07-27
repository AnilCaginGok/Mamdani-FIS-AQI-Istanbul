import numpy as np
CATEGORY_INDEX = {'Good': 0, 'Moderate': 1, 'USG': 2, 'Unhealthy': 3, 'Very Unhealthy': 4}
CATEGORY_LABELS = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy']
PM25_BREAKPOINTS = [(0.0, 12.0, 0, 50, 0), (12.1, 35.4, 51, 100, 1), (35.5, 55.4, 101, 150, 2), (55.5, 150.4, 151, 200, 3), (150.5, 250.4, 201, 300, 4)]
PM10_BREAKPOINTS = [(0.0, 54.0, 0, 50, 0), (55.0, 154.0, 51, 100, 1), (155.0, 254.0, 101, 150, 2), (255.0, 354.0, 151, 200, 3), (355.0, 424.0, 201, 300, 4)]
NO2_BREAKPOINTS = [(0.0, 101.4, 0, 50, 0), (101.5, 191.3, 51, 100, 1), (191.4, 200.0, 101, 150, 2)]
NO2_BREAKPOINTS_FULL = [(0.0, 101.4, 0, 50, 0), (101.5, 191.3, 51, 100, 1), (191.4, 688.0, 101, 150, 2), (688.1, 1241.5, 151, 200, 3), (1241.6, 2000.0, 201, 300, 4)]

def _epa_sub_index(concentration: float, breakpoints: list) -> tuple:
    for (c_lo, c_hi, aqi_lo, aqi_hi, cat_idx) in breakpoints:
        if c_lo <= concentration <= c_hi:
            aqi = (aqi_hi - aqi_lo) / (c_hi - c_lo) * (concentration - c_lo) + aqi_lo
            return (aqi, cat_idx)
    (c_lo, c_hi, aqi_lo, aqi_hi, cat_idx) = breakpoints[-1]
    aqi = (aqi_hi - aqi_lo) / (c_hi - c_lo) * (concentration - c_lo) + aqi_lo
    return (float(np.clip(aqi, aqi_lo, aqi_hi + 50)), min(4, cat_idx))

def _aqi_to_index(aqi_value: float) -> float:
    if aqi_value <= 50:
        return aqi_value / 50.0
    elif aqi_value <= 100:
        return 1.0 + (aqi_value - 50) / 50.0
    elif aqi_value <= 150:
        return 2.0 + (aqi_value - 100) / 50.0
    elif aqi_value <= 200:
        return 3.0 + (aqi_value - 150) / 50.0
    else:
        return 4.0 + (aqi_value - 200) / 100.0

class CrispEPA:

    def __init__(self, particulate_mode: str='pm25'):
        if particulate_mode not in ('pm25', 'pm10'):
            raise ValueError("particulate_mode must be 'pm25' or 'pm10'")
        self.particulate_mode = particulate_mode

    def evaluate(self, particulate: float, no2: float):
        if self.particulate_mode == 'pm10':
            (p_aqi, p_cat, _) = self._pm10_sub_index(particulate)
        else:
            (p_aqi, p_cat, _) = self._pm25_sub_index(particulate)
        (no2_aqi, no2_cat, _) = self._no2_sub_index(no2)
        if p_aqi >= no2_aqi:
            final_aqi = p_aqi
            final_cat = p_cat
        else:
            final_aqi = no2_aqi
            final_cat = no2_cat
        aqi_index = _aqi_to_index(final_aqi)
        return (aqi_index, CATEGORY_LABELS[min(4, final_cat)])

    def evaluate_batch(self, particulate_arr, no2_arr):
        particulate_arr = np.asarray(particulate_arr, dtype=float)
        no2_arr = np.asarray(no2_arr, dtype=float)
        indices = np.empty(len(particulate_arr))
        categories = []
        for (k, (p, n)) in enumerate(zip(particulate_arr, no2_arr)):
            (idx, cat) = self.evaluate(p, n)
            indices[k] = idx
            categories.append(cat)
        return (indices, categories)

    def category_index(self, particulate: float, no2: float) -> int:
        (_, cat_label) = self.evaluate(particulate, no2)
        return CATEGORY_LABELS.index(cat_label)

    @staticmethod
    def _pm25_sub_index(pm25: float):
        (aqi, cat_idx) = _epa_sub_index(max(0.0, pm25), PM25_BREAKPOINTS)
        return (aqi, cat_idx, CATEGORY_LABELS[cat_idx])

    @staticmethod
    def _pm10_sub_index(pm10: float):
        (aqi, cat_idx) = _epa_sub_index(max(0.0, pm10), PM10_BREAKPOINTS)
        return (aqi, cat_idx, CATEGORY_LABELS[cat_idx])

    @staticmethod
    def _no2_sub_index(no2: float):
        (aqi, cat_idx) = _epa_sub_index(max(0.0, no2), NO2_BREAKPOINTS_FULL)
        return (aqi, cat_idx, CATEGORY_LABELS[min(4, cat_idx)])

    def get_pm25_aqi(self, pm25_arr):
        return np.array([_epa_sub_index(p, PM25_BREAKPOINTS)[0] for p in pm25_arr])

    def get_pm10_aqi(self, pm10_arr):
        return np.array([_epa_sub_index(p, PM10_BREAKPOINTS)[0] for p in pm10_arr])

    def get_no2_aqi(self, no2_arr):
        return np.array([_epa_sub_index(n, NO2_BREAKPOINTS_FULL)[0] for n in no2_arr])
