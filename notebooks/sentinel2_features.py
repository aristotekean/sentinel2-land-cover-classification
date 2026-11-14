"""Spectral feature transformer shared by the phase 2 notebook and the models it logs.

The input is the ten Sentinel-2 bands as digital numbers, in this order:
B01, B02, B03, B04, B05, B06, B07, B8A, B11, B12.

The transformer optionally:
- applies ``log1p`` to the bands;
- appends normalised-difference indices, always computed from the raw bands.

Linear and Gaussian models cannot build band ratios on their own; the indices give
them the contrasts that separate vegetation, water and built-up surfaces.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

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
