# main.py
# Master script for the A5 Toxicity Classification competition.
# Runs all phases in order. Any phase can be commented out or replaced
# without breaking the rest, as each phase passes its outputs explicitly.
#
# Run from the directory that contains the three CSV data files:
#   python main.py

import warnings
import time
import numpy as np

# Suppress common non-critical sklearn convergence warnings during search.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from data_loader import load_data
from eda import run_eda
from preprocessing import build_preprocessor
from models import (
    run_baseline,
    tune_xgboost,
    tune_svm,
    tune_mlp,
    build_stacking_ensemble,
    select_best_model,
)
from feature_selection import run_feature_selection
from evaluation import run_evaluation
from predict import run_predictions
from config import N_JOBS, PLOTS_DIR, _n_physical


# ====================================================================
# Helpers
# ====================================================================

def _phase_header(n, title):
    """Print a consistent phase header with a timestamp."""
    ts = time.strftime("%H:%M:%S")
    print(f"\n{'#' * 60}")
    print(f"# PHASE {n} -- {title}")
    print(f"# Started at {ts}")
    print("#" * 60)


def _phase_footer(title, elapsed):
    """Print a consistent phase footer with elapsed time."""
    mins, secs = divmod(int(elapsed), 60)
    print(f"\n  ✓ Phase '{title}' completed in {mins}m {secs:02d}s")


# ====================================================================
# Main
# ====================================================================

def main():
    pipeline_start = time.time()

    print("=" * 60)
    print("  A5 TOXICITY CLASSIFICATION -- FULL PIPELINE")
    print(f"  Mac Studio: {_n_physical} logical CPUs, n_jobs={N_JOBS}")
    print(f"  Plots directory: {PLOTS_DIR}")
    print(f"  Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # PHASE 1 -- Data Loading and Exploration
    # ------------------------------------------------------------------
    _phase_header(1, "LOADING DATA AND EXPLORATORY DATA ANALYSIS")
    t1 = time.time()

    print("\n  Loading CSV files...")
    X_train, y_train, X_test, train_df, test_df, labels_series = load_data()
    print(f"  X_train : {X_train.shape}   X_test : {X_test.shape}   y_train : {y_train.shape}")

    eda_summary = run_eda(train_df, test_df, labels_series, y_train=y_train)
    is_imbalanced = eda_summary["is_imbalanced"]

    _phase_footer("LOADING DATA AND EDA", time.time() - t1)

    # ------------------------------------------------------------------
    # PHASE 2 -- Preprocessing Pipeline
    # ------------------------------------------------------------------
    _phase_header(2, "PREPROCESSING PIPELINE")
    t2 = time.time()

    preprocessor = build_preprocessor(eda_summary)

    _phase_footer("PREPROCESSING PIPELINE", time.time() - t2)

    # ------------------------------------------------------------------
    # PHASE 3 -- Baseline Models
    # ------------------------------------------------------------------
    _phase_header(3, "BASELINE MODELS")
    t3 = time.time()

    baseline_results, lr_pipeline, rf_pipeline = run_baseline(
        preprocessor, X_train, y_train, is_imbalanced
    )

    # Abort early if both baselines are too weak (possible data issue).
    if (baseline_results["lr"]["mean"] < 0.65 and
            baseline_results["rf"]["mean"] < 0.65):
        raise RuntimeError(
            "Both baselines are below 0.65 BCR. "
            "Verify data loading and label alignment before continuing."
        )

    _phase_footer("BASELINE MODELS", time.time() - t3)

    # ------------------------------------------------------------------
    # PHASE 4 -- Hyperparameter Tuning
    # ------------------------------------------------------------------
    _phase_header(4, "HYPERPARAMETER TUNING")
    t4 = time.time()

    xgb_search = tune_xgboost(preprocessor, X_train, y_train)
    svm_search = tune_svm(preprocessor, X_train, y_train, with_probability=True)
    mlp_search = tune_mlp(preprocessor, X_train, y_train)

    ensemble, ensemble_scores = build_stacking_ensemble(
        xgb_search, svm_search, mlp_search,
        preprocessor, X_train, y_train,
    )

    best_pipeline, best_name, summary_table = select_best_model(
        baseline_results, xgb_search, svm_search,
        mlp_search, ensemble, ensemble_scores,
    )

    # XGBoost feature importance plot.
    if xgb_search is not None:
        try:
            from plots import plot_xgb_feature_importance
            plot_xgb_feature_importance(xgb_search, top_n=30)
        except Exception:
            pass

    # Determine the best CV BCR for Phase 5.
    if best_name == "Stacking Ensemble":
        phase4_bcr = ensemble_scores.mean()
    elif best_name == "XGBoost (tuned)" and xgb_search is not None:
        phase4_bcr = xgb_search.best_score_
    elif best_name == "SVM RBF (tuned)" and svm_search is not None:
        phase4_bcr = svm_search.best_score_
    elif best_name == "MLP (tuned)" and mlp_search is not None:
        phase4_bcr = mlp_search.best_score_
    else:
        phase4_bcr = max(
            baseline_results["lr"]["mean"],
            baseline_results["rf"]["mean"],
        )

    _phase_footer("HYPERPARAMETER TUNING", time.time() - t4)

    # ------------------------------------------------------------------
    # PHASE 5 -- Feature Selection
    # ------------------------------------------------------------------
    _phase_header(5, "FEATURE SELECTION")
    t5 = time.time()

    final_pipeline, selection_description = run_feature_selection(
        preprocessor, best_pipeline, X_train, y_train, baseline_bcr=phase4_bcr
    )
    print(f"\n  Feature selection decision: {selection_description}")

    _phase_footer("FEATURE SELECTION", time.time() - t5)

    # ------------------------------------------------------------------
    # PHASE 6 -- Final Model Training and BCRhat Estimation
    # ------------------------------------------------------------------
    _phase_header(6, "FINAL MODEL TRAINING AND BCRHAT ESTIMATION")
    t6 = time.time()

    fitted_pipeline, bcr_hat, sigma, fold_bcr_scores, optimal_threshold = run_evaluation(
        final_pipeline, X_train, y_train
    )

    _phase_footer("FINAL MODEL TRAINING", time.time() - t6)

    # ------------------------------------------------------------------
    # PHASE 7 -- Generate and Save Predictions
    # ------------------------------------------------------------------
    _phase_header(7, "GENERATING PREDICTIONS")
    t7 = time.time()

    run_predictions(fitted_pipeline, X_test, threshold=optimal_threshold)

    _phase_footer("GENERATING PREDICTIONS", time.time() - t7)

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    total_elapsed = time.time() - pipeline_start
    total_mins, total_secs = divmod(int(total_elapsed), 60)

    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Best model             : {best_name}")
    print(f"  Feature selection      : {selection_description}")
    print(f"  BCRhat (to submit)     : {bcr_hat:.4f}")
    print(f"  Optimal threshold      : {optimal_threshold:.2f}")
    print(f"  sigma                  : {sigma:.4f}")
    print(f"  Per-fold BCR scores    : {np.array2string(fold_bcr_scores, precision=4)}")
    print(f"  Confidence interval    : "
          f"[{bcr_hat - 1.96 * sigma:.4f}, {bcr_hat + 1.96 * sigma:.4f}]")
    print(f"  Plots saved to         : {PLOTS_DIR}/")
    print(f"  Total pipeline time    : {total_mins}m {total_secs:02d}s")
    print("=" * 60)
    print(f"\n  Submit 'predictions.csv' and BCRhat = {round(bcr_hat, 4)}")


if __name__ == "__main__":
    main()
