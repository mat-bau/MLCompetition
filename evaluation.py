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


def compute_p_score(bcr_real, bcr_submitted, sigma):
    """Competition performance metric P (WCCI 2006 / LINFO2262 A5).

    P = BCR - Δ(BCR) · [1 - exp(-Δ(BCR) / σ)]

    Parameters
    ----------
    bcr_real      : float — true test BCR (or our best proxy for it)
    bcr_submitted : float — the BCRhat value we submit on Inginious
    sigma         : float — theoretical error bar on BCR

    Returns
    -------
    P : float  (maximised when bcr_submitted = bcr_real, i.e. Δ=0 → P=BCR)
    """
    delta = abs(bcr_real - bcr_submitted)
    denom = max(sigma, 1e-10)
    return float(bcr_real - delta * (1.0 - np.exp(-delta / denom)))


def analyse_p_score_range(bcr_hat, predicted_bcr, sigma_used, sigma_theoretical):
    """Show how many competition points you gain or lose depending on what
    BCRhat you submit.

    Uses bcr_hat (OOF estimate) as the best available proxy for the unknown
    true test BCR.  Computes P for four scenarios:

    ┌─────────────────────────────┬─────────────────────────────────┐
    │ Scenario                    │ BCRhat submitted                │
    ├─────────────────────────────┼─────────────────────────────────┤
    │ Perfect prediction          │ bcr_hat  (Δ=0)                  │
    │ Our calibrated estimate     │ predicted_bcr  (shrinkage)      │
    │ CI lower bound (pessimistic)│ bcr_hat - 1.96·σ                │
    │ CI upper bound (optimistic) │ bcr_hat + 1.96·σ                │
    └─────────────────────────────┴─────────────────────────────────┘

    The σ used for the P formula is sigma_theoretical (WCCI 2006), which
    matches the competition definition.

    Returns
    -------
    results : dict keyed by scenario name → {'submitted': float, 'P': float}
    """
    sigma_p = max(sigma_theoretical, 1e-10)   # σ in the P formula
    ci_low  = bcr_hat - 1.96 * sigma_used
    ci_high = bcr_hat + 1.96 * sigma_used

    scenarios = {
        "Perfect prediction (Δ=0)"    : bcr_hat,
        "Our estimate (shrinkage)"     : predicted_bcr,
        "CI lower bound (pessimistic)" : ci_low,
        "CI upper bound (optimistic)"  : ci_high,
    }

    results = {}
    print(f"\n  {'─'*62}")
    print(f"  COMPETITION P-SCORE ANALYSIS  (BCR_real proxy = {bcr_hat:.4f})")
    print(f"  σ (theoretical, used in P formula) = {sigma_p:.6f}")
    print(f"  {'─'*62}")
    print(f"  {'Scenario':<35} {'Submitted BCRhat':>16} {'P score':>9}")
    print(f"  {'-'*62}")

    for name, submitted in scenarios.items():
        p = compute_p_score(bcr_hat, submitted, sigma_p)
        results[name] = {"submitted": round(float(submitted), 6),
                         "P": round(float(p), 6)}
        marker = "  ← SUBMIT THIS" if name == "Our estimate (shrinkage)" else ""
        print(f"  {name:<35} {submitted:>16.4f} {p:>9.4f}{marker}")

    # Interpretation.
    p_best  = results["Perfect prediction (Δ=0)"]["P"]
    p_ours  = results["Our estimate (shrinkage)"]["P"]
    p_low   = results["CI lower bound (pessimistic)"]["P"]
    p_high  = results["CI upper bound (optimistic)"]["P"]
    loss_ours = p_best - p_ours

    print(f"  {'─'*62}")
    print(f"  Max possible P (perfect prediction) : {p_best:.4f}")
    print(f"  Our estimate loses                   : {loss_ours:.4f} pts vs perfect")
    print(f"  Range if submitting CI extremes      : "
          f"[{min(p_low, p_high):.4f}, {max(p_low, p_high):.4f}]")
    print(f"  {'─'*62}")

    return results


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

        preds_at_05                    = (oof_probas >= 0.5).astype(int)
        optimal_threshold, optimal_bcr = _scan_threshold(y_train, oof_probas)
        oof_preds                      = (oof_probas >= optimal_threshold).astype(int)

        # --- balanced_accuracy_score comparison table ---
        bcr_05           = balanced_accuracy_score(y_train, preds_at_05)
        bcr_05_adj       = balanced_accuracy_score(y_train, preds_at_05, adjusted=True)
        bcr_opt          = balanced_accuracy_score(y_train, oof_preds)
        bcr_opt_adj      = balanced_accuracy_score(y_train, oof_preds, adjusted=True)

        print(f"  OOF done in {time.time() - t0:.1f}s")
        print(f"\n  balanced_accuracy_score comparison (OOF):")
        print(f"  {'Threshold':<12} {'adjusted':>8} {'BCR':>8}  note")
        print(f"  {'-'*52}")
        print(f"  {'0.50 (def)':12} {'False':>8} {bcr_05:>8.4f}  ← what sklearn CV reports as 'balanced_accuracy'")
        print(f"  {'0.50 (def)':12} {'True':>8} {bcr_05_adj:>8.4f}  = (BCR-0.5)/0.5, chance-corrected")
        print(f"  {f'{optimal_threshold:.2f} (opt)':12} {'False':>8} {bcr_opt:>8.4f}  ← what we optimise and submit")
        print(f"  {f'{optimal_threshold:.2f} (opt)':12} {'True':>8} {bcr_opt_adj:>8.4f}  chance-corrected at optimal threshold")
        print(f"  {'-'*52}")
        print(f"  Gain from threshold tuning : {bcr_opt - bcr_05:+.4f}")
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


def compute_per_fold_bcr(pipeline, X_train, y_train, optimal_threshold=0.5):
    """Compute per-fold BCR on independent evaluation folds (EVAL_*).

    Uses predict_proba + optimal_threshold (same threshold found in compute_oof_bcr)
    so that per-fold BCR is consistent with the OOF estimate.  Falls back to
    predict() at 0.5 if the pipeline has no predict_proba.

    Parameters
    ----------
    optimal_threshold : float, default 0.5
        Decision threshold from compute_oof_bcr.  Each fold's BCR is computed at
        both 0.5 (sklearn default) and this value — the difference is printed so
        the threshold gain is visible per-fold.

    Returns
    -------
    fold_bcr_scores : np.ndarray of shape (EVAL_N_SPLITS,) — BCR at optimal_threshold
    sigma_empirical : float, standard error of the mean fold BCR
    fold_stats      : dict from _analyse_folds
    """
    cv = StratifiedKFold(
        n_splits=EVAL_N_SPLITS, shuffle=True, random_state=EVAL_RANDOM_STATE
    )

    fold_bcr_scores    = []
    fold_bcr_at_05     = []
    use_proba          = optimal_threshold != 0.5

    print(f"  Computing per-fold BCR ({EVAL_N_SPLITS} folds, seed={EVAL_RANDOM_STATE}, "
          f"threshold={optimal_threshold:.2f}):")
    t_total = time.time()

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train), 1):
        t_fold = time.time()
        fold_pipeline = deepcopy(pipeline)
        fold_pipeline.fit(X_train[train_idx], y_train[train_idx])

        y_val = y_train[val_idx]

        if use_proba:
            try:
                fold_probas  = fold_pipeline.predict_proba(X_train[val_idx])[:, 1]
                fold_pred_05  = (fold_probas >= 0.5).astype(int)
                fold_pred_opt = (fold_probas >= optimal_threshold).astype(int)
            except AttributeError:
                fold_pred_opt = fold_pipeline.predict(X_train[val_idx])
                fold_pred_05  = fold_pred_opt
        else:
            fold_pred_opt = fold_pipeline.predict(X_train[val_idx])
            fold_pred_05  = fold_pred_opt

        bcr_opt = balanced_accuracy_score(y_val, fold_pred_opt)
        bcr_05  = balanced_accuracy_score(y_val, fold_pred_05)
        fold_bcr_scores.append(bcr_opt)
        fold_bcr_at_05.append(bcr_05)
        elapsed = time.time() - t_fold

        delta_str = f"  Δ={bcr_opt - bcr_05:+.4f}" if use_proba else ""
        print(f"    Fold {fold_idx:2d}/{EVAL_N_SPLITS}  "
              f"BCR@{optimal_threshold:.2f}={bcr_opt:.4f}  "
              f"BCR@0.50={bcr_05:.4f}{delta_str}  ({elapsed:.1f}s)")

    fold_bcr_scores = np.array(fold_bcr_scores)
    fold_bcr_at_05  = np.array(fold_bcr_at_05)
    sigma_empirical = fold_bcr_scores.std() / np.sqrt(len(fold_bcr_scores))
    fold_stats      = _analyse_folds(fold_bcr_scores)

    print(f"  Per-fold evaluation done in {time.time() - t_total:.1f}s")

    if use_proba:
        mean_gain = (fold_bcr_scores - fold_bcr_at_05).mean()
        print(f"  Mean gain from threshold tuning (per fold): {mean_gain:+.4f}")

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
        pipeline, X_train, y_train, optimal_threshold=optimal_threshold
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
    # -----------------------------------------------------------------
    # BCR vs BER clarification
    # -----------------------------------------------------------------
    # What to submit on Inginious:
    #   Question 2 → BCRhat = predicted_bcr  (balanced classification rate)
    # BER (balanced error rate = 1 - BCR) is kept as a diagnostic metric
    # only; do NOT submit it.
    # -----------------------------------------------------------------
    print(f"\n  ╔══════════════════════════════════════════════════════╗")
    print(f"  ║  SUBMIT ON INGINIOUS (Question 2)                   ║")
    print(f"  ║  BCRhat = {predicted_bcr:.4f}                               ║")
    print(f"  ╚══════════════════════════════════════════════════════╝")
    print(f"  Optimal threshold (for predictions)  : {optimal_threshold:.2f}")
    print(f"  BER (diagnostic only, do NOT submit) : {1 - predicted_bcr:.4f}")
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

    # P-score analysis.
    p_analysis = analyse_p_score_range(bcr_hat, predicted_bcr,
                                        sigma_used, sigma_theoretical)

    # Plots.
    try:
        from plots import (plot_confusion_matrix, plot_fold_bcr_scores,
                           plot_p_score_analysis)
        plot_confusion_matrix(y_train, oof_preds,
                              title="Confusion Matrix (OOF predictions)",
                              filename="eval_01_confusion_matrix.png")
        plot_fold_bcr_scores(fold_bcr_scores, bcr_hat)
        plot_p_score_analysis(bcr_hat, predicted_bcr, sigma_used)
    except Exception as exc:
        print(f"  [plot] WARNING: Could not generate evaluation plots: {exc}")

    fitted_pipeline = train_final_model(deepcopy(pipeline), X_train, y_train)

    return (fitted_pipeline, bcr_hat, sigma_used, fold_bcr_scores,
            optimal_threshold, predicted_bcr, sigma_theoretical, p_analysis)
