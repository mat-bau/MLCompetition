# evaluation.py
# Phase 6 -- Final Model Training and BCRhat Estimation.
# Computes out-of-fold BCR, per-fold BCR, sigma, and fits the final model.

import numpy as np
from copy import deepcopy

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from config import CV_N_SPLITS, CV_RANDOM_STATE


def compute_oof_bcr(pipeline, X_train, y_train):
    """Collect out-of-fold predictions and compute the OOF BCR.

    Each sample is predicted by a model trained on the other folds only,
    making this an unbiased estimate of generalization performance.

    Parameters
    ----------
    pipeline : sklearn Pipeline (preprocessor + model)
    X_train  : np.ndarray
    y_train  : np.ndarray of int

    Returns
    -------
    bcr_hat    : float, balanced accuracy on all OOF predictions
    oof_preds  : np.ndarray of OOF predictions
    """
    cv = StratifiedKFold(
        n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE
    )

    oof_preds = cross_val_predict(
        pipeline, X_train, y_train,
        cv=cv, method="predict",
    )

    bcr_hat = balanced_accuracy_score(y_train, oof_preds)
    return bcr_hat, oof_preds


def compute_per_fold_bcr(pipeline, X_train, y_train):
    """Compute per-fold BCR and the standard error of the mean.

    This is used to estimate sigma = std(fold BCRs) / sqrt(n_folds),
    which represents uncertainty in BCRhat due to finite sample size.

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

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_fold_train = X_train[train_idx]
        X_fold_val   = X_train[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val   = y_train[val_idx]

        # Clone the pipeline to avoid state leakage between folds.
        fold_pipeline = deepcopy(pipeline)
        fold_pipeline.fit(X_fold_train, y_fold_train)
        fold_pred = fold_pipeline.predict(X_fold_val)
        fold_bcr  = balanced_accuracy_score(y_fold_val, fold_pred)
        fold_bcr_scores.append(fold_bcr)

    fold_bcr_scores = np.array(fold_bcr_scores)
    sigma = fold_bcr_scores.std() / np.sqrt(len(fold_bcr_scores))
    return fold_bcr_scores, sigma


def train_final_model(pipeline, X_train, y_train):
    """Fit the chosen pipeline on the entire training set.

    This is the model used to generate predictions on the test set.
    It must not be evaluated on training data (that would be optimistic).

    Parameters
    ----------
    pipeline : sklearn Pipeline
    X_train  : np.ndarray
    y_train  : np.ndarray of int

    Returns
    -------
    fitted_pipeline : the same pipeline, now fitted on all training data
    """
    print("Fitting final model on all training data...")
    pipeline.fit(X_train, y_train)
    print("Final model fitted.")
    return pipeline


def run_evaluation(pipeline, X_train, y_train):
    """Run Phase 6: OOF BCR estimation, sigma, and final model training.

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
    print("\n" + "#" * 60)
    print("# PHASE 6 -- BCRhat ESTIMATION AND FINAL TRAINING")
    print("#" * 60)

    print("Computing out-of-fold predictions...")
    bcr_hat, oof_preds = compute_oof_bcr(pipeline, X_train, y_train)

    print("Computing per-fold BCR scores...")
    fold_bcr_scores, sigma = compute_per_fold_bcr(pipeline, X_train, y_train)

    fold_mean = fold_bcr_scores.mean()

    print(f"\nBCRhat (OOF)             : {bcr_hat:.4f}")
    print(f"Per-fold BCR scores      : {fold_bcr_scores}")
    print(f"Fold mean BCR            : {fold_mean:.4f}")
    print(f"Fold std                 : {fold_bcr_scores.std():.4f}")
    print(f"sigma (std error of mean): {sigma:.4f}")
    print(f"95% confidence interval  : "
          f"[{bcr_hat - 1.96 * sigma:.4f}, {bcr_hat + 1.96 * sigma:.4f}]")

    # Sanity check: large discrepancy between OOF BCR and mean fold BCR
    # suggests high fold-to-fold variance or a methodological issue.
    if abs(bcr_hat - fold_mean) > 0.01:
        print("WARNING: OOF BCRhat and mean fold BCR differ by more than 0.01.")
        print("Investigate whether fold-level variance is unusually high.")

    # Sanity check: BCRhat above 0.97 may indicate leakage or an easy dataset.
    if bcr_hat > 0.97:
        print("WARNING: BCRhat > 0.97 -- verify there is no leakage in the pipeline.")

    # Sanity check: high sigma suggests 10-fold CV would give a better estimate.
    if sigma > 0.02:
        print("WARNING: sigma > 0.02 -- consider using 10-fold CV for more stable estimates.")

    # Fit the final model on all training data.
    fitted_pipeline = train_final_model(deepcopy(pipeline), X_train, y_train)

    return fitted_pipeline, bcr_hat, sigma, fold_bcr_scores