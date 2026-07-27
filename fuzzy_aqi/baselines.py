from __future__ import annotations
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from .evaluation import _label_to_idx, CATEGORY_LABELS
from .crisp_system import CrispEPA

class CrispBaseline:

    def __init__(self, particulate_mode: str='pm25'):
        self.crisp = CrispEPA(particulate_mode=particulate_mode)

    def evaluate_batch(self, particulate_arr, no2_arr):
        return self.crisp.evaluate_batch(particulate_arr, no2_arr)

class KNNBaseline:

    def __init__(self, n_neighbors: int=11):
        self.n_neighbors = n_neighbors
        self._clf = None

    def fit(self, particulate_arr, no2_arr, category_labels):
        X = np.column_stack([np.asarray(particulate_arr, dtype=float), np.asarray(no2_arr, dtype=float)])
        y = np.array([_label_to_idx(c) for c in category_labels])
        self._clf = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights='distance')
        self._clf.fit(X, y)

    def evaluate_batch(self, particulate_arr, no2_arr):
        X = np.column_stack([np.asarray(particulate_arr, dtype=float), np.asarray(no2_arr, dtype=float)])
        idx = self._clf.predict(X)
        indices = np.array([float(i) for i in idx])
        categories = [CATEGORY_LABELS[int(i)] for i in idx]
        return (indices, categories)

def baseline_accuracy(predicted_idx, reference_idx) -> float:
    return float(np.mean(np.asarray(predicted_idx) == np.asarray(reference_idx)))
