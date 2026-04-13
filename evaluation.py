# evaluation.py
# Phase 6 -- Final Model Training and BCRhat Estimation.
# Computes out-of-fold BCR, per-fold BCR, sigma, and fits the final model.
# Generates confusion matrix and fold BCR plots.

import time
import numpy as np
from copy import deepcopy

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import CV_N_SPLITS, CV_RANDOM_STATE


def _scan_threshold(y_true, oof_probas):
    """Find the decision threshold that maximises BCR on OOF predictions.

    Parameters
    ----------
    y_true     : np.ndarray of true binary labels
    oof_probas : np.ndarray of shape (n_samples,) — P(class=1) for each sample

    Returns
    -------
    optimal_threshold : float in [0.05, 0.95]
    optimal_bcr       : float, BCR at that threshold
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    bcr_at_t = np.array([
        balanced_accuracy_score(y_true, (oof_probas >= t).astype(int))
        for t in thresholds
    ])
    best_idx          = int(np.argmax(bcr_at_t))
    optimal_threshold = float(thresholds[best_idx])
    optimal_bcr       = float(bcr_at_t[best_idx])
    return optimal_threshold, optimal_bcr


def compute_oof_bcr(pipeline, X_train, y_train):
    """Collect out-of-fold predictions and compute the OOF BCR.

    Uses predict_proba to enable threshold scanning.  Falls back to
    predict (threshold=0.5) if the pipeline does not expose probabilities.

    Parameters
    ----------
    pipeline : sklearn Pipeline (preprocessor + model)
    X_train  : np.ndarray
    y_train  : np.ndarray of int

    Returns
    -------
    bcr_hat           : float, BCR at the optimal threshold
    oof_preds         : np.ndarray of OOF predictions (at optimal threshold)
    optimal_threshold : float, the threshold that maximises BCR
    """
    cv = StratifiedKFold(
        n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE
    )

    print("  Collecting out-of-fold predictions (this re-trains the model "
          f"{CV_N_SPLITS}× on sub-splits)...")
    t0 = time.time()

    try:
        oof_probas = cross_val_predict(
            pipeline, X_train, y_train,
            cv=cv, method="predict_proba",
            verbose=1,
        )[:, 1]

        # BCR at default threshold for reference.
        bcr_default = balanced_accuracy_score(
            y_train, (oof_probas >= 0.5).astype(int)
        )

        # Scan thresholds and pick the best one.
        optimal_threshold, optimal_bcr = _scan_threshold(y_train, oof_probas)
        oof_preds = (oof_probas >= optimal_threshold).astype(int)

        print(f"  OOF done in {time.time() - t0:.1f}s")
        print(f"  BCR @ threshold=0.50       : {bcr_default:.4f}")
        print(f"  BCR @ threshold={optimal_threshold:.2f} (optimal): {optimal_bcr:.4f}")
        return optimal_bcr, oof_preds, optimal_threshold

    except (AttributeError, ValueError):
        # Pipeline does not support predict_proba — fall back to predict.
        print("  [WARNING] predict_proba not available; using default threshold 0.5.")
        oof_preds = cross_val_predict(
            pipeline, X_train, y_train,
            cv=cv, method="predict",
            verbose=1,
        )
        bcr_hat = balanced_accuracy_score(y_train, oof_preds)
        print(f"  OOF done in {time.time() - t0:.1f}s  |  BCRhat = {bcr_hat:.4f}")
        return bcr_hat, oof_preds, 0.5


def compute_per_fold_bcr(pipeline, X_train, y_train):
    """Compute per-fold BCR and the standard error of the mean.

    Parameters
    ----------
    pipeline : sklearn Pipeline (preprocessor + model)
    X_train  : np.ndarray
    y_train  : np.ndarray of int

    Returns
    -------
    fold_bcr_scores : np.ndarray of shape (CV_N_SPLITS,)
    sigma           : float, standard error of the mean fold BCR
    """
    cv = StratifiedKFold(
        n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE
    )

    fold_bcr_scores = []
    print(f"  Computing per-fold BCR ({CV_N_SPLITS} folds):")
    t_total = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        t_fold = time.time()
        fold_pipeline = deepcopy(pipeline)
        fold_pipeline.fit(X_train[train_idx], y_train[train_idx])
        fold_pred = fold_pipeline.predict(X_train[val_idx])
        fold_bcr  = balanced_accuracy_score(y_train[val_idx], fold_pred)
        fold_bcr_scores.append(fold_bcr)
        elapsed = time.time() - t_fold
        print(f"    Fold {fold_idx}/{CV_N_SPLITS}  BCR={fold_bcr:.4f}  ({elapsed:.1f}s)")

    fold_bcr_scores = np.array(fold_bcr_scores)
    sigma = fold_bcr_scores.std() / np.sqrt(len(fold_bcr_scores))
    print(f"  Per-fold evaluation done in {time.time() - t_total:.1f}s")
    return fold_bcr_scores, sigma


def train_final_model(pipeline, X_train, y_train):
    """Fit the chosen pipeline on the entire training set.

    Parameters
    ----------
    pipeline : sklearn Pipeline
    X_train  : np.ndarray
    y_train  : np.ndarray of int

    Returns
    -------
    fitted_pipeline : the same pipeline, now fitted on all training data
    """
    print("  Fitting final model on all training data...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"  Final model fitted in {time.time() - t0:.1f}s.")
    return pipeline


def run_evaluation(pipeline, X_train, y_train):
    """Run Phase 6: OOF BCR estimation, sigma, confusion matrix, and final model.

    Parameters
    ----------
    pipeline : sklearn Pipeline (best pipeline from Phase 5)
    X_train  : np.ndarray
    y_train  : np.ndarray of int

    Returns
    -------
    fitted_pipeline : pipeline fitted on all training data
    bcr_hat         : float, the number to submit as BCR estimate
    sigma           : float, standard error of the mean fold BCR
    fold_bcr_scores : np.ndarray of per-fold BCR scores
    """

    bcr_hat, oof_preds, optimal_threshold = compute_oof_bcr(pipeline, X_train, y_train)
    fold_bcr_scores, sigma = compute_per_fold_bcr(pipeline, X_train, y_train)

    fold_mean = fold_bcr_scores.mean()

    print(f"\n  BCRhat (OOF)             : {bcr_hat:.4f}")
    print(f"  Per-fold BCR scores      : {np.array2string(fold_bcr_scores, precision=4)}")
    print(f"  Fold mean BCR            : {fold_mean:.4f}")
    print(f"  Fold std                 : {fold_bcr_scores.std():.4f}")
    print(f"  sigma (std error of mean): {sigma:.4f}")
    print(f"  95% CI                   : "
          f"[{bcr_hat - 1.96 * sigma:.4f}, {bcr_hat + 1.96 * sigma:.4f}]")

    # Classification report (gives precision / recall per class).
    print("\n  Classification Report (OOF predictions):")
    report = classification_report(y_train, oof_preds,
                                   target_names=["negative (0)", "positive (1)"])
    for line in report.splitlines():
        print(f"    {line}")

    # Sanity checks.
    if abs(bcr_hat - fold_mean) > 0.01:
        print("  WARNING: OOF BCRhat and mean fold BCR differ by more than 0.01.")
        print("  Investigate whether fold-level variance is unusually high.")
    if bcr_hat > 0.97:
        print("  WARNING: BCRhat > 0.97 -- verify there is no leakage in the pipeline.")
    if sigma > 0.02:
        print("  WARNING: sigma > 0.02 -- consider using 10-fold CV for more stable estimates.")

    # Plots.
    try:
        from plots import plot_confusion_matrix, plot_fold_bcr_scores
        plot_confusion_matrix(y_train, oof_preds,
                              title="Confusion Matrix (OOF predictions)",
                              filename="eval_01_confusion_matrix.png")
        plot_fold_bcr_scores(fold_bcr_scores, bcr_hat)
    except Exception as exc:
        print(f"  [plot] WARNING: Could not generate evaluation plots: {exc}")

    print(f"\n  Optimal decision threshold  : {optimal_threshold:.2f}")

    # Fit the final model on all training data.
    fitted_pipeline = train_final_model(deepcopy(pipeline), X_train, y_train)

    return fitted_pipeline, bcr_hat, sigma, fold_bcr_scores, optimal_threshold
