# main.py
# Master script for the A5 Toxicity Classification competition.
# Runs all phases in order. Any phase can be commented out or replaced
# without breaking the rest, as each phase passes its outputs explicitly.
#
# Run from the directory that contains the three CSV data files:
#   python main.py

import warnings
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
    build_voting_ensemble,
    select_best_model,
)
from feature_selection import run_feature_selection
from evaluation import run_evaluation
from predict import run_predictions


def main():
    print("=" * 60)
    print("A5 TOXICITY CLASSIFICATION -- FULL PIPELINE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # PHASE 1 -- Data Loading and Exploration
    # ------------------------------------------------------------------
    print("\nLoading data...")
    X_train, y_train, X_test, train_df, test_df, labels_series = load_data()

    eda_summary = run_eda(train_df, test_df, labels_series)

    is_imbalanced = eda_summary["is_imbalanced"]

    # ------------------------------------------------------------------
    # PHASE 2 -- Preprocessing Pipeline
    # ------------------------------------------------------------------
    preprocessor = build_preprocessor(eda_summary)

    # ------------------------------------------------------------------
    # PHASE 3 -- Baseline Models
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # PHASE 4 -- Hyperparameter Tuning
    # ------------------------------------------------------------------

    # Tune XGBoost.
    xgb_search = tune_xgboost(preprocessor, X_train, y_train)

    # Tune SVM with probability=True so it can be used in soft voting.
    svm_search = tune_svm(preprocessor, X_train, y_train, with_probability=True)

    # Tune MLP.
    mlp_search = tune_mlp(preprocessor, X_train, y_train)

    # Build soft voting ensemble from the best single models.
    ensemble, ensemble_scores = build_voting_ensemble(
        xgb_search, svm_search, mlp_search,
        preprocessor, X_train, y_train,
    )

    # Select the overall best pipeline.
    best_pipeline, best_name, summary_table = select_best_model(
        baseline_results, xgb_search, svm_search,
        mlp_search, ensemble, ensemble_scores,
    )

    # The best BCR from Phase 4 is used as the baseline for Phase 5.
    # Extract the best CV BCR mean from the chosen model.
    # For search objects, use best_score_; for ensemble, use ensemble_scores.
    if best_name == "Voting Ensemble":
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

    # ------------------------------------------------------------------
    # PHASE 5 -- Feature Selection
    # ------------------------------------------------------------------
    final_pipeline, selection_description = run_feature_selection(
        preprocessor, best_pipeline, X_train, y_train, baseline_bcr=phase4_bcr
    )
    print(f"Feature selection decision: {selection_description}")

    # ------------------------------------------------------------------
    # PHASE 6 -- Final Model Training and BCRhat Estimation
    # ------------------------------------------------------------------
    fitted_pipeline, bcr_hat, sigma, fold_bcr_scores = run_evaluation(
        final_pipeline, X_train, y_train
    )

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"Best model             : {best_name}")
    print(f"Feature selection      : {selection_description}")
    print(f"BCRhat (to submit)     : {bcr_hat:.4f}")
    print(f"sigma                  : {sigma:.4f}")
    print(f"Per-fold BCR scores    : {np.array2string(fold_bcr_scores, precision=4)}")
    print(f"Confidence interval    : "
          f"[{bcr_hat - 1.96 * sigma:.4f}, {bcr_hat + 1.96 * sigma:.4f}]")
    print("=" * 60)

    # ------------------------------------------------------------------
    # PHASE 7 -- Generate and Save Predictions
    # ------------------------------------------------------------------
    run_predictions(fitted_pipeline, X_test)

    print("\nPipeline complete. Submit 'predictions.csv' and BCRhat =", round(bcr_hat, 4))


if __name__ == "__main__":
    main()