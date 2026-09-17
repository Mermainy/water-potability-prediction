"""Water-measurement relationships, fitted independently in each CV fold."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted

from schema import FEATURES


ENGINEERED_FEATURES = [
    "solids_per_conductivity",
    "chloramines_per_organic_carbon",
    "trihalomethanes_per_organic_carbon",
    "solids_conductivity_residual",
]


class WaterFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, denominator_floor=1e-6):
        self.denominator_floor = denominator_floor

    def fit(self, X, y=None):
        # Estimate expected solids using this training fold only, without labels.
        self.solids_model_ = LinearRegression().fit(X[["Conductivity"]], X["Solids"])
        return self

    def transform(self, X):
        check_is_fitted(self, "solids_model_")
        frame = X[FEATURES].copy()
        conductivity = frame["Conductivity"].clip(lower=self.denominator_floor)
        carbon = frame["Organic_carbon"].clip(lower=self.denominator_floor)
        frame["solids_per_conductivity"] = frame["Solids"] / conductivity
        frame["chloramines_per_organic_carbon"] = frame["Chloramines"] / carbon
        # Dataset THMs are in micrograms/L; carbon ppm is treated as mg/L
        # for these dilute-water measurements, so convert THMs to mg/L.
        frame["trihalomethanes_per_organic_carbon"] = frame["Trihalomethanes"] / 1000 / carbon
        frame["solids_conductivity_residual"] = (
            frame["Solids"] - self.solids_model_.predict(frame[["Conductivity"]])
        )
        return frame

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES + ENGINEERED_FEATURES, dtype=object)
