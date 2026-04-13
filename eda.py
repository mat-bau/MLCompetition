# eda.py
# Phase 1 -- Exploratory Data Analysis.
# All checks are run here and a summary dict is returned.
# No modeling code lives in this file.

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from config import (
    IMBALANCE_THRESHOLD,
    NEAR_CONSTANT_VARIANCE,
    HIGH_CORR_THRESHOLD,
    OUTLIER_STD_THRESHOLD,
    DIST_SHIFT_THRESHOLD,
)


def check_shapes(train_df, test_df, labels_series):
    """Verify expected shapes and print a summary."""
    print("=" * 60)
    print("SHAPE VERIFICATION")
    print(f"  Train features : {train_df.shape}  (expected 3000 x 1024)")
    print(f"  Test  features : {test_df.shape}   (expected 1000 x 1024)")
    print(f"  Labels         : {labels_series.shape}  (expected 3000 x 1)")
    assert train_df.shape[1] == test_df.shape[1], \
        "Train and test have different numbers of features."


def check_label_distribution(labels_series):
    """Count classes and flag imbalance if one class is below threshold."""
    counts = labels_series.value_counts()
    total  = len(labels_series)
    print("=" * 60)
    print("LABEL DISTRIBUTION")
    for label, count in counts.items():
        pct = 100.0 * count / total
        print(f"  {label:10s} : {count:5d}  ({pct:.1f}%)")

    minority_frac = counts.min() / total
    is_imbalanced = minority_frac < IMBALANCE_THRESHOLD
    if is_imbalanced:
        print(f"  WARNING: minority class is {minority_frac:.1%} -- imbalanced dataset.")
        print("  All models must use class_weight='balanced'.")
    else:
        print(f"  Balance OK: minority class is {minority_frac:.1%}.")
    return is_imbalanced


def check_missing_values(train_df, test_df, labels_series):
    """Report total NaN counts per file."""
    print("=" * 60)
    print("MISSING VALUES")
    nan_train  = train_df.isnull().sum().sum()
    nan_test   = test_df.isnull().sum().sum()
    nan_labels = labels_series.isnull().sum()
    print(f"  Train features NaN count : {nan_train}")
    print(f"  Test  features NaN count : {nan_test}")
    print(f"  Labels         NaN count : {nan_labels}")
    return {"nan_train": nan_train, "nan_test": nan_test, "nan_labels": nan_labels}


def check_dtypes(train_df):
    """Verify all feature columns are numeric."""
    print("=" * 60)
    print("DATA TYPES")
    non_float = train_df.dtypes[train_df.dtypes == object]
    if len(non_float) > 0:
        print(f"  WARNING: {len(non_float)} columns are object type: {list(non_float.index)}")
    else:
        print(f"  All {train_df.shape[1]} feature columns are numeric -- OK.")


def check_value_ranges(train_df):
    """Report global min, max, mean, std across all features."""
    print("=" * 60)
    print("FEATURE VALUE RANGES")
    global_min  = train_df.values.min()
    global_max  = train_df.values.max()
    global_mean = train_df.values.mean()
    global_std  = train_df.values.std()
    print(f"  Global min  : {global_min:.4f}")
    print(f"  Global max  : {global_max:.4f}")
    print(f"  Global mean : {global_mean:.4f}")
    print(f"  Global std  : {global_std:.4f}")
    return {"min": global_min, "max": global_max, "mean": global_mean, "std": global_std}


def check_constant_features(train_df):
    """Find zero-variance and near-constant features."""
    print("=" * 60)
    print("CONSTANT / NEAR-CONSTANT FEATURES")

    variances = train_df.var(axis=0)

    zero_var_cols = variances[variances == 0.0].index.tolist()
    near_const_cols = variances[
        (variances > 0.0) & (variances < NEAR_CONSTANT_VARIANCE)
    ].index.tolist()

    print(f"  Zero-variance features     : {len(zero_var_cols)}")
    print(f"  Near-constant (var < {NEAR_CONSTANT_VARIANCE}) : {len(near_const_cols)}")

    if len(zero_var_cols) > 100:
        print("  NOTE: More than 100 zero-variance features -- removal is strongly advised.")

    return zero_var_cols, near_const_cols


def check_high_correlation(train_df, zero_var_cols):
    """Flag feature pairs with absolute Pearson correlation above threshold.

    Computation is O(n_features^2). Skips zero-variance columns to avoid
    division-by-zero in the correlation calculation.
    """
    print("=" * 60)
    print(f"HIGH CORRELATION PAIRS (threshold = {HIGH_CORR_THRESHOLD})")

    # Drop zero-variance columns before computing correlations.
    cols_to_use = [c for c in train_df.columns if c not in zero_var_cols]
    corr_matrix = train_df[cols_to_use].corr().abs()

    # Extract upper triangle only to avoid counting each pair twice.
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    high_corr_pairs = [
        (col, row)
        for col in upper.columns
        for row in upper.index
        if upper.loc[row, col] > HIGH_CORR_THRESHOLD
    ]

    print(f"  High-correlation pairs found : {len(high_corr_pairs)}")
    if len(high_corr_pairs) > 100:
        print("  NOTE: More than 100 pairs -- consider adding a CorrelationFilter step.")

    return high_corr_pairs


def check_outliers(train_df):
    """Count features and values with outliers beyond OUTLIER_STD_THRESHOLD std."""
    print("=" * 60)
    print(f"OUTLIER CHECK (threshold = {OUTLIER_STD_THRESHOLD} std deviations)")

    col_means = train_df.mean(axis=0)
    col_stds  = train_df.std(axis=0)

    # Avoid division by zero for constant columns.
    col_stds_safe = col_stds.replace(0, np.nan)

    z_scores = (train_df - col_means) / col_stds_safe
    outlier_mask = z_scores.abs() > OUTLIER_STD_THRESHOLD

    n_outlier_features = outlier_mask.any(axis=0).sum()
    n_outlier_values   = outlier_mask.sum().sum()
    outlier_frac = n_outlier_values / (train_df.shape[0] * train_df.shape[1])

    print(f"  Features with at least one outlier : {n_outlier_features}")
    print(f"  Total outlier values               : {n_outlier_values}")
    print(f"  Fraction of all values             : {outlier_frac:.4%}")

    if outlier_frac < 0.001:
        print("  Outlier fraction < 0.1% -- StandardScaler may be acceptable.")
    else:
        print("  Meaningful outlier presence -- RobustScaler is recommended.")

    return n_outlier_features, n_outlier_values, outlier_frac


def check_distribution_shift(train_df, test_df):
    """Flag features where train and test mean differ by more than DIST_SHIFT_THRESHOLD stds."""
    print("=" * 60)
    print(f"TRAIN/TEST DISTRIBUTION SHIFT (threshold = {DIST_SHIFT_THRESHOLD} train std)")

    train_means = train_df.mean(axis=0)
    train_stds  = train_df.std(axis=0).replace(0, np.nan)
    test_means  = test_df.mean(axis=0)

    mean_diff = (test_means - train_means).abs() / train_stds
    shifted_features = mean_diff[mean_diff > DIST_SHIFT_THRESHOLD]

    print(f"  Features with distribution shift : {len(shifted_features)}")
    if len(shifted_features) > 50:
        print("  WARNING: More than 50 shifted features -- test set may be out-of-distribution.")
        print("  BCRhat confidence should be reduced accordingly.")

    return len(shifted_features)


def run_eda(train_df, test_df, labels_series):
    """Run all Phase 1 checks and return a summary dictionary.

    Parameters
    ----------
    train_df       : pd.DataFrame of train features
    test_df        : pd.DataFrame of test features
    labels_series  : pd.Series of string labels

    Returns
    -------
    summary : dict with all key findings
    """
    print("\n" + "#" * 60)
    print("# PHASE 1 -- EXPLORATORY DATA ANALYSIS")
    print("#" * 60)

    check_shapes(train_df, test_df, labels_series)
    is_imbalanced = check_label_distribution(labels_series)
    nan_counts    = check_missing_values(train_df, test_df, labels_series)
    check_dtypes(train_df)
    value_ranges  = check_value_ranges(train_df)
    zero_var_cols, near_const_cols = check_constant_features(train_df)
    high_corr_pairs = check_high_correlation(train_df, zero_var_cols)
    n_outlier_features, n_outlier_values, outlier_frac = check_outliers(train_df)
    n_shifted = check_distribution_shift(train_df, test_df)

    summary = {
        "is_imbalanced"      : is_imbalanced,
        "nan_counts"         : nan_counts,
        "value_ranges"       : value_ranges,
        "zero_var_cols"      : zero_var_cols,
        "near_const_cols"    : near_const_cols,
        "high_corr_pairs"    : high_corr_pairs,
        "n_high_corr_pairs"  : len(high_corr_pairs),
        "n_outlier_features" : n_outlier_features,
        "n_outlier_values"   : n_outlier_values,
        "outlier_frac"       : outlier_frac,
        "n_shifted_features" : n_shifted,
    }

    print("\n" + "=" * 60)
    print("EDA SUMMARY")
    print(f"  Imbalanced dataset         : {is_imbalanced}")
    print(f"  Zero-variance features     : {len(zero_var_cols)}")
    print(f"  Near-constant features     : {len(near_const_cols)}")
    print(f"  High-corr pairs (>{HIGH_CORR_THRESHOLD}) : {len(high_corr_pairs)}")
    print(f"  Features with outliers     : {n_outlier_features}")
    print(f"  Shifted features (train/test): {n_shifted}")
    print("=" * 60 + "\n")

    return summary


if __name__ == "__main__":
    from data_loader import load_data
    _, _, _, train_df, test_df, labels_series = load_data()
    summary = run_eda(train_df, test_df, labels_series)