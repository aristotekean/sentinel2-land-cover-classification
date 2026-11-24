"""Spectral feature transformer shared by the phase 2 notebook and the models it logs.

The input is the ten Sentinel-2 bands as digital numbers, in this order:
B01, B02, B03, B04, B05, B06, B07, B8A, B11, B12.

The transformer optionally:
- applies ``log1p`` to the bands;
- appends normalised-difference indices, always computed from the raw bands.

Linear and Gaussian models cannot build band ratios on their own; the indices give
them the contrasts that separate vegetation, water and built-up surfaces.

``ClassWeightedClassifier`` adds ``class_weight`` to estimators that only accept
``sample_weight`` in ``fit`` (the neural network), so every model of the comparison
can be tuned with the same class-balancing option.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.utils.class_weight import compute_sample_weight

BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12")

# name: (band a, band b) for (a - b) / (a + b)
INDICES = {
    "NDVI": ("B8A", "B04"),   # vegetation
    "NDWI": ("B03", "B8A"),   # open water (McFeeters)
    "MNDWI": ("B03", "B11"),  # water versus built-up (Xu)
    "NDMI": ("B8A", "B11"),   # moisture
    "NBR": ("B8A", "B12"),    # dark and burnt surfaces
}


class SpectralFeatures(BaseEstimator, TransformerMixin):
    """Bands, optionally log-transformed, plus optional normalised-difference indices."""

    def __init__(self, indices=False, log=False):
        self.indices = indices
        self.log = log

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if X.shape[1] != len(BANDS):
            raise ValueError(f"expected {len(BANDS)} bands, got {X.shape[1]}")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        columns = [np.log1p(np.clip(X, 0, None)) if self.log else X]
        if self.indices:
            band = {name: X[:, position] for position, name in enumerate(BANDS)}
            for a, b in INDICES.values():
                columns.append(((band[a] - band[b]) / (band[a] + band[b] + 1e-6))[:, None])
        return np.hstack(columns)

    def get_feature_names_out(self, input_features=None):
        names = [f"log_{b}" if self.log else b for b in BANDS]
        return np.array(names + (list(INDICES) if self.indices else []), dtype=object)


class ClassWeightedClassifier(ClassifierMixin, BaseEstimator):
    """Fit ``estimator`` with per-sample weights derived from ``class_weight``.

    ``class_weight=None`` fits without weights; ``"balanced"`` weights each pixel by
    n_samples / (n_classes * n_pixels_of_its_class), as ``class_weight`` does in the
    other scikit-learn classifiers.
    """

    def __init__(self, estimator, class_weight=None):
        self.estimator = estimator
        self.class_weight = class_weight

    def fit(self, X, y):
        weights = None if self.class_weight is None else compute_sample_weight(self.class_weight, y)
        self.estimator_ = clone(self.estimator).fit(X, y, sample_weight=weights)
        self.classes_ = self.estimator_.classes_
        self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)
