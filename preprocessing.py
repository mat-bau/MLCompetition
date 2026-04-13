# preprocessing.py
# Phase 2 -- Preprocessing Pipeline.
# Builds and returns a sklearn Pipeline with variance filtering,
# imputation, and scaling. The scaler choice is driven by EDA findings.

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# -------------------------------------------------------------------
# Optional custom step: CorrelationFilter
# Used when Phase 1 identifies more than 100 high-correlation pairs.
# Drops one column from each correlated pair (keeps the first).
# -------------------------------------------------------------------
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Remove features that are highly correlated with another feature.

    Parameters
    ----------
    threshold : float
        Absolute Pearson correlation above which one of the two
        features in a pair is dropped.

    Attributes
    ----------
    cols_to_drop_ : list of int
        Column indices (in the fitted data) to drop at transform time.
    """

    def __init__(self, threshold=0.98):
        self.threshold = threshold

    def fit(self, X, y=None):
        # Convert to DataFrame for convenient correlation computation.
        df = pd.DataFrame(X)
        corr_matrix = df.corr().abs()

        # Inspect the upper triangle only to avoid double-counting.
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Identify columns where any correlation exceeds the threshold.
        # These are the columns to drop (the second member of each pair).
        self.cols_to_drop_ = [
            col for col in upper.columns
            if any(upper[col] > self.threshold)
        ]
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X)
        df_filtered = df.drop(columns=self.cols_to_drop_, errors="ignore")
        return df_filtered.values

    def get_feature_names_out(self, input_features=None):
        # Required by some sklearn utilities.
        return np.array(input_features) if input_features is not None else None


def build_preprocessor(eda_summary):
    """Build and return the preprocessing pipeline based on EDA findings.

    Decision rules applied here:
    - If outlier_frac < 0.001 (less than 0.1% of values are outliers),
      use StandardScaler; otherwise use RobustScaler.
    - If n_high_corr_pairs > 100, add a CorrelationFilter step after
      the variance filter.

    Parameters
    ----------
    eda_summary : dict returned by eda.run_eda()

    Returns
    -------
    preprocessor : sklearn Pipeline
    """

    # Choose scaler based on outlier fraction from EDA.
    outlier_frac = eda_summary.get("outlier_frac", 1.0)
    if outlier_frac < 0.001:
        scaler = StandardScaler()
        scaler_name = "StandardScaler (outlier fraction < 0.1%)"
    else:
        scaler = RobustScaler()
        scaler_name = "RobustScaler (outlier fraction >= 0.1%)"

    print(f"Scaler selected: {scaler_name}")

    # Decide whether to include a correlation filter step.
    n_high_corr_pairs = eda_summary.get("n_high_corr_pairs", 0)
    use_corr_filter = n_high_corr_pairs > 100

    if use_corr_filter:
        print(f"CorrelationFilter added: {n_high_corr_pairs} high-correlation pairs found.")
        steps = [
            ("variance_filter", VarianceThreshold(threshold=0.0)),
            ("corr_filter",     CorrelationFilter(threshold=0.98)),
            ("imputer",         SimpleImputer(strategy="median")),
            ("scaler",          scaler),
        ]
    else:
        print(f"CorrelationFilter skipped: only {n_high_corr_pairs} high-correlation pairs.")
        steps = [
            ("variance_filter", VarianceThreshold(threshold=0.0)),
            ("imputer",         SimpleImputer(strategy="median")),
            ("scaler",          scaler),
        ]

    preprocessor = Pipeline(steps)
    return preprocessor


if __name__ == "__main__":
    # Quick smoke test with a dummy EDA summary.
    dummy_summary = {"outlier_frac": 0.005, "n_high_corr_pairs": 50}
    preprocessor = build_preprocessor(dummy_summary)
    print(preprocessor)