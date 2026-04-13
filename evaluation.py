# evaluation.py
# Phase 6 -- Final Model Training and BCRhat Estimation.
#
# Improvements over baseline (WCCI 2006 Performance Prediction Challenge):
#
# 1. Independent CV folds (EVAL_RANDOM_STATE != CV_RANDOM_STATE) to avoid
#    optimism bias from reusing tuning folds for performance estimation.
# 2. 10-fold CV (EVAL_N_SPLITS=10) for a lower-variance estimate.
# 3. Theoretical σ from the WCCI 2006 BER variance formula.
# 4. BCR shrinkage: predicted_BCR = bcr_hat - alpha * sigma  (corrects
#    the well-documented tendency of CV estimates to be over-optimistic).
# 5. Enriched fold analysis: skewness, anomaly detection, fold ranking.

import time
import numpy as np
from copy import deepcopy

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import (
    CV_N_SPLITS,
    CV_RANDOM_STATE,
    EVAL_N_SPLITS,
    EVAL_RANDOM_STATE,
    SHRINKAGE_ALPHA,
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

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


def compute_ber_variance(y_true, oof_preds):
    """Theoretical BER variance from WCCI 2006 (Guyon et al.).

    BER = 0.5 * (FNR + FPR)
    Var(BER) ≈ 0.25 * (FNR*(1-FNR)/n_pos + FPR*(1-FPR)/n_neg)

    This variance formula assumes independent Bernoulli predictions and
    is an exact lower bound; CV folds introduce some correlation so the
    empirical sigma may be slightly larger.

    Returns
    -------
    sigma_theoretical : float, sqrt(Var(BER))  — interpreted as BCR uncertainty
    """
    cm = confusion_matrix(y_true, oof_preds)
    tn, fp, fn, tp = cm.ravel()
    n_pos = tp + fn
    n_neg = tn + fp

    fnr = fn / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)

    var_ber = 0.25 * (
        fnr * (1 - fnr) / max(n_pos, 1) +
        fpr * (1 - fpr) / max(n_neg, 1)
    )
    return float(np.sqrt(var_ber))


def estimate_predicted_bcr(bcr_hat, sigma, alpha=None):
    """BCR shrinkage: corrects the optimism bias of cross-validation.

    WCCI 2006 showed that participants consistently over-estimated their
    model's performance.  A conservative correction:

        predicted_BCR = bcr_hat - alpha * sigma

    The alpha value scales with instability (high sigma → more pessimistic).

    Parameters
    ----------
    bcr_hat : float  — OOF BCR estimate
    sigma   : float  — uncertainty (std error or theoretical)
    alpha   : float or None — if None, uses the adaptive competition strategy

    Returns
    -------
    predicted_bcr : float
    alpha_used    : float
    """
    if alpha is not None:
        return float(bcr_hat - alpha * sigma), float(alpha)

    # Adaptive competition strategy.
    if sigma > 0.03:
        alpha_used = 1.0      # high variance → strong pessimism
    else:
        alpha_used = 0.5      # moderate pessimism
    return float(bcr_hat - alpha_used * sigma), float(alpha_used)


def _analyse_folds(fold_scores):
    """Compute descriptive statistics and detect anomalous folds.

    Returns
    -------
    stats : dict with keys: mean, std, skewness, anomalous_folds (list of 1-based indices)
    """
    from scipy.stats import skew as scipy_skew

    arr    = np.array(fold_scores)
    mean   = float(arr.mean())
    std    = float(arr.std())
    sk     = float(scipy_skew(arr))
    # Fold is anomalous if it deviates more than 2 std from the mean.
    anomalous = [i + 1 for i, v in enumerate(arr) if abs(v - mean) > 2 * std]

    return {"mean": mean, "std": std, "skewness": sk, "anomalous_folds": anomalous}


# -------------------------------------------------------------------
# Main evaluation functions
# -------------------------------------------------------------------

def compute_oof_bcr(pipeline, X_train, y_train):
    """Collect out-of-fold predictions on INDEPENDENT folds (EVAL_*).

    Uses EVAL_N_SPLITS and EVAL_RANDOM_STATE — deliberately different from the
    tuning CV (CV_N_SPLITS / CV_RANDOM_STATE) to reduce optimism bias.

    Parameters
    ----------
    pipeline : sklearn-compatible Pipeline
    X_train  : np.ndarray
    y_train  : np.ndarray of int

    Returns
    -------
    bcr_hat           : float, BCR at the optimal threshold
    oof_preds         : np.ndarray of OOF predictions (at optimal threshold)
    optimal_threshold : float
    """
    cv = StratifiedKFold(
        n_splits=EVAL_N_SPLITS, shuffle=True, random_state=EVAL_RANDOM_STATE
    )

    print(f"  Collecting OOF predictions ({EVAL_N_SPLITS}-fold, "
          f"seed={EVAL_RANDOM_STATE} — independent from tuning folds)...")
    t0 = time.time()

    try:
        oof_probas = cross_val_predict(
            pipeline, X_train, y_train,
            cv=cv, method="predict_proba",
            verbose=1,
        )[:, 1]

        bcr_default                   = balanced_accuracy_score(
            y_train, (oof_probas >= 0.5).astype(int)
        )
        optimal_threshold, optimal_bcr = _scan_threshold(y_train, oof_probas)
        oof_preds = (oof_probas >= optimal_threshold).astype(int)

        print(f"  OOF done in {time.time() - t0:.1f}s")
        print(f"  BCR @ threshold=0.50          : {bcr_default:.4f}")
        print(f"  BCR @ threshold={optimal_threshold:.2f} (optimal) : {optimal_bcr:.4f}")
        return optimal_bcr, oof_preds, optimal_threshold

    except (AttributeError, ValueError):
        print("  [WARNING] predict_proba not available; using default threshold 0.5.")
        oof_preds = cross_val_predict(
            pipeline, X_train, y_train,
            cv=cv, method="predict", verbose=1,
        )
        bcr_hat = balanced_accuracy_score(y_train, oof_preds)
        print(f"  OOF done in {time.time() - t0:.1f}s  |  BCRhat = {bcr_hat:.4f}")
        return bcr_hat, oof_preds, 0.5


def compute_per_fold_bcr(pipeline, X_train, y_train):
    """Compute per-fold BCR on independent evaluation folds (EVAL_*).

    Returns
    -------
    fold_bcr_scores : np.ndarray of shape (EVAL_N_SPLITS,)
    sigma_empirical : float, standard error of the mean fold BCR
    fold_stats      : dict from _analyse_folds
    """
    cv = StratifiedKFold(
        n_splits=EVAL_N_SPLITS, shuffle=True, random_state=EVAL_RANDOM_STATE
    )

    fold_bcr_scores = []
    print(f"  Computing per-fold BCR ({EVAL_N_SPLITS} folds, seed={EVAL_RANDOM_STATE}):")
    t_total = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        t_fold = time.time()
        fold_pipeline = deepcopy(pipeline)
        fold_pipeline.fit(X_train[train_idx], y_train[train_idx])
        fold_pred = fold_pipeline.predict(X_train[val_idx])
        fold_bcr  = balanced_accuracy_score(y_train[val_idx], fold_pred)
        fold_bcr_scores.append(fold_bcr)
        elapsed = time.time() - t_fold
        print(f"    Fold {fold_idx:2d}/{EVAL_N_SPLITS}  BCR={fold_bcr:.4f}  ({elapsed:.1f}s)")

    fold_bcr_scores = np.array(fold_bcr_scores)
    sigma_empirical = fold_bcr_scores.std() / np.sqrt(len(fold_bcr_scores))
    fold_stats      = _analyse_folds(fold_bcr_scores)

    print(f"  Per-fold evaluation done in {time.time() - t_total:.1f}s")

    # Fold ranking (best → worst).
    ranked = np.argsort(fold_bcr_scores)[::-1]
    print(f"  Fold ranking (best→worst): {[int(i+1) for i in ranked]}")

    if fold_stats["anomalous_folds"]:
        print(f"  WARNING: Anomalous folds detected (>2σ from mean): "
              f"{fold_stats['anomalous_folds']}")

    print(f"  Skewness of fold scores  : {fold_stats['skewness']:+.3f}")

    return fold_bcr_scores, sigma_empirical, fold_stats


def train_final_model(pipeline, X_train, y_train):
    """Fit the chosen pipeline on the entire training set."""
    print("  Fitting final model on all training data...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    print(f"  Final model fitted in {time.time() - t0:.1f}s.")
    return pipeline


def run_evaluation(pipeline, X_train, y_train):
    """Run Phase 6: independent OOF BCR, sigma (empirical + theoretical),
    shrinkage, enriched fold analysis, confusion matrix, and final fit.

    Returns
    -------
    fitted_pipeline    : pipeline fitted on all training data
    bcr_hat            : float, OOF BCR at optimal threshold
    sigma_used         : float, max(sigma_empirical, sigma_theoretical)
    fold_bcr_scores    : np.ndarray
    optimal_threshold  : float
    predicted_bcr      : float, shrinkage-corrected estimate to submit
    sigma_theoretical  : float
    """
    bcr_hat, oof_preds, optimal_threshold = compute_oof_bcr(pipeline, X_train, y_train)
    fold_bcr_scores, sigma_empirical, fold_stats = compute_per_fold_bcr(
        pipeline, X_train, y_train
    )

    # Theoretical σ (WCCI 2006 formula).
    sigma_theoretical = compute_ber_variance(y_train, oof_preds)

    # Use the more conservative (larger) of the two sigma estimates.
    sigma_used = max(sigma_empirical, sigma_theoretical)

    # BCR shrinkage — adaptive + show 3 alphas for reference.
    predicted_bcr, alpha_used = estimate_predicted_bcr(bcr_hat, sigma_used)

    fold_mean = fold_bcr_scores.mean()

    print(f"\n  {'─'*55}")
    print(f"  BCRhat (OOF, optimal threshold)  : {bcr_hat:.4f}")
    print(f"  Per-fold BCR scores              : "
          f"{np.array2string(fold_bcr_scores, precision=4)}")
    print(f"  Fold mean BCR                    : {fold_mean:.4f}")
    print(f"  Fold std                         : {fold_bcr_scores.std():.4f}")
    print(f"  σ empirical  (std/√k)            : {sigma_empirical:.4f}")
    print(f"  σ theoretical (WCCI formula)     : {sigma_theoretical:.4f}")
    print(f"  σ used (max of two)              : {sigma_used:.4f}")
    print(f"  95% CI                           : "
          f"[{bcr_hat - 1.96*sigma_used:.4f}, {bcr_hat + 1.96*sigma_used:.4f}]")
    print(f"\n  BCR Shrinkage (alpha comparison):")
    for alpha_test in [0.3, 0.5, 1.0]:
        est, _ = estimate_predicted_bcr(bcr_hat, sigma_used, alpha=alpha_test)
        marker = " ← adaptive choice" if abs(alpha_test - alpha_used) < 0.01 else ""
        print(f"    α={alpha_test:.1f} → predicted_BCR = {est:.4f}{marker}")
    print(f"\n  predicted_BCR (to submit)        : {predicted_bcr:.4f}")
    print(f"  BER_guess    (to submit)         : {1 - predicted_bcr:.4f}")
    print(f"  Optimal threshold                : {optimal_threshold:.2f}")
    print(f"  {'─'*55}")

    # Classification report.
    print("\n  Classification Report (OOF predictions):")
    report = classification_report(y_train, oof_preds,
                                   target_names=["negative (0)", "positive (1)"])
    for line in report.splitlines():
        print(f"    {line}")

    # Sanity checks.
    if abs(bcr_hat - fold_mean) > 0.01:
        print("  WARNING: OOF BCRhat and mean fold BCR differ by > 0.01.")
    if bcr_hat > 0.97:
        print("  WARNING: BCRhat > 0.97 — verify there is no leakage.")
    if sigma_used > 0.025:
        print("  WARNING: σ > 0.025 — estimate is noisy; trust predicted_BCR over bcr_hat.")

    # Plots.
    try:
        from plots import plot_confusion_matrix, plot_fold_bcr_scores
        plot_confusion_matrix(y_train, oof_preds,
                              title="Confusion Matrix (OOF predictions)",
                              filename="eval_01_confusion_matrix.png")
        plot_fold_bcr_scores(fold_bcr_scores, bcr_hat)
    except Exception as exc:
        print(f"  [plot] WARNING: Could not generate evaluation plots: {exc}")

    fitted_pipeline = train_final_model(deepcopy(pipeline), X_train, y_train)

    return (fitted_pipeline, bcr_hat, sigma_used, fold_bcr_scores,
            optimal_threshold, predicted_bcr, sigma_theoretical)
