# feature_selection.py
# Phase 5 -- Feature Selection
#
# Seven strategies are evaluated, all scored with a LogisticRegression
# reference model for a consistent, apples-to-apples comparison:
#
#   Filter (univariate)
#     1. f_classif      -- ANOVA F-test (parametric, assumes Gaussian)
#     2. mutual_info    -- Mutual information (non-parametric, information theory)
#     3. mann_whitney   -- Mann-Whitney U (non-parametric, rank-based)
#   Dimensionality reduction
#     4. PCA            -- linear projection to k components
#   Wrapper
#     5. RFE            -- Recursive Feature Elimination with LR (coef_ ranking)
#   Embedded
#     6. L1 LogReg      -- SelectFromModel with L1-penalised LogisticRegression
#     7. LinearSVC L1   -- SelectFromModel with L1-penalised LinearSVC

import time
import numpy as np
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel,
)
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

from models import get_cv
from config import (
    SELECT_K_VALUES, RFE_K_VALUES, EMBEDDED_C_VALUES,
    PCA_N_VALUES, RANDOM_STATE, N_JOBS,
)


# ====================================================================
# Helpers
# ====================================================================

def _lr_ref():
    """Fast balanced LogisticRegression used as the reference final estimator."""
    return LogisticRegression(
        class_weight="balanced", max_iter=2000,
        random_state=RANDOM_STATE, n_jobs=N_JOBS,
    )


def _eval_pipeline(preprocessor, selector, selector_name, ref_model=None):
    """preprocessor → selector → ref_model.

    ref_model defaults to LR but should be the actual best model's estimator
    (e.g. the SVC or LDA object extracted from best_model_pipeline) so that
    the comparison is fair: selector + real_model vs real_model with all features.
    """
    model = deepcopy(ref_model) if ref_model is not None else _lr_ref()
    return Pipeline([
        ("preprocessor", deepcopy(preprocessor)),
        (selector_name,  selector),
        ("model",        model),
    ])


def _apply_pipeline(preprocessor, selector, selector_name, best_model_pipeline):
    """preprocessor → selector → best_model  (final application pipeline).

    If the best model is a StackingClassifier (no .steps), falls back to LR
    because injecting a selector into a stacking ensemble would cause double
    preprocessing inside each base estimator.
    """
    if hasattr(best_model_pipeline, "steps"):
        model_name, model_obj = best_model_pipeline.steps[-1]
        return Pipeline([
            ("preprocessor", deepcopy(preprocessor)),
            (selector_name,  selector),
            (model_name,     deepcopy(model_obj)),
        ])
    # Ensemble fallback: selector + LR is evaluated vs. baseline below.
    return Pipeline([
        ("preprocessor", deepcopy(preprocessor)),
        (selector_name,  selector),
        ("model",        _lr_ref()),
    ])


def _mann_whitney_score(X, y):
    """SelectKBest-compatible score function using Mann-Whitney U p-values.

    Returns -log10(p) as the score (higher = more discriminative).
    Non-parametric: no Gaussian assumption; robust to heavy-tailed features.
    """
    from scipy.stats import mannwhitneyu
    classes = np.unique(y)
    X0, X1 = X[y == classes[0]], X[y == classes[1]]
    scores  = np.empty(X.shape[1])
    pvalues = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        res = mannwhitneyu(X0[:, j], X1[:, j], alternative="two-sided")
        scores[j]  = res.statistic
        pvalues[j] = res.pvalue
    pvalues = np.clip(pvalues, 1e-300, 1.0)
    return -np.log10(pvalues), pvalues


def _cv_score(pipeline, X, y):
    """Return (mean, std) balanced accuracy over the standard CV splits.

    sklearn raises ValueError when ALL folds fail even with error_score=np.nan,
    so we catch that here and return (nan, nan) — the caller skips this config.
    """
    try:
        s = cross_val_score(
            pipeline, X, y,
            cv=get_cv(), scoring="balanced_accuracy", n_jobs=N_JOBS,
            error_score=np.nan,
        )
        if np.all(np.isnan(s)):
            return float("nan"), float("nan")
        return float(np.nanmean(s)), float(np.nanstd(s))
    except Exception:
        return float("nan"), float("nan")


# ====================================================================
# Evaluation functions — one per strategy family
# ====================================================================

def _eval_filter(preprocessor, X, y, score_func, label, ref_model=None):
    """Evaluate SelectKBest for every k in SELECT_K_VALUES.

    Uses ref_model as the downstream classifier so the comparison is fair:
    selector + same_model vs same_model with all features.
    """
    print(f"\n  [{label}]  SelectKBest  k={SELECT_K_VALUES}")
    results = []
    for k in SELECT_K_VALUES:
        t0   = time.time()
        sel  = SelectKBest(score_func=score_func, k=k)
        pipe = _eval_pipeline(preprocessor, sel, "selector", ref_model)
        mean, std = _cv_score(pipe, X, y)
        results.append({"k": k, "mean": mean, "std": std, "label": label})
        if np.isnan(mean):
            print(f"    k={str(k):<6}  skipped (model failed — n_features > n_minority_samples?)")
        else:
            print(f"    k={str(k):<6}  BCR={mean:.4f} ± {std:.4f}  ({time.time()-t0:.1f}s)")
    return results


def _eval_pca(preprocessor, X, y, ref_model=None):
    """Evaluate PCA for every n in PCA_N_VALUES."""
    print(f"\n  [PCA]  n_components={PCA_N_VALUES}")
    results = []
    for n in PCA_N_VALUES:
        t0   = time.time()
        sel  = PCA(n_components=n, random_state=RANDOM_STATE)
        pipe = _eval_pipeline(preprocessor, sel, "pca", ref_model)
        mean, std = _cv_score(pipe, X, y)
        results.append({"k": n, "mean": mean, "std": std, "label": "PCA"})
        print(f"    n={n:<4}  BCR={mean:.4f} ± {std:.4f}  ({time.time()-t0:.1f}s)")
    return results


def _eval_rfe(preprocessor, X, y, ref_model=None):
    """Wrapper: RFE with LogisticRegression as ranker, ref_model as classifier.

    RFE uses LR coef_ to rank and eliminate features (step=100) — the ranker
    is always LR because it needs coef_. The downstream classifier is ref_model
    (the actual best model) so the BCR is comparable to the baseline.
    """
    print(f"\n  [RFE (LR ranker)]  n_features={RFE_K_VALUES}  step=100")
    results = []
    for k in RFE_K_VALUES:
        t0  = time.time()
        rfe = RFE(
            estimator=LogisticRegression(
                class_weight="balanced", max_iter=1000,
                random_state=RANDOM_STATE, n_jobs=1,
            ),
            n_features_to_select=k,
            step=100,
        )
        pipe = _eval_pipeline(preprocessor, rfe, "rfe", ref_model)
        mean, std = _cv_score(pipe, X, y)
        results.append({"k": k, "mean": mean, "std": std, "label": "RFE (LR)"})
        print(f"    k={k:<4}  BCR={mean:.4f} ± {std:.4f}  ({time.time()-t0:.1f}s)")
    return results


def _eval_embedded(preprocessor, X, y, inner_estimator_fn, label, ref_model=None):
    """Embedded: SelectFromModel with an L1-penalised estimator.

    Crash guard: C=0.001 can zero all coefficients → 0 features selected →
    downstream model fails. We catch this and report NaN for that config.
    """
    print(f"\n  [{label}]  SelectFromModel  C={EMBEDDED_C_VALUES}")
    results = []

    # Fit once on full preprocessed data to count n_features per C value.
    prep_pipe = Pipeline([("preprocessor", deepcopy(preprocessor))])
    X_prep    = prep_pipe.fit_transform(X, y)

    for c in EMBEDDED_C_VALUES:
        t0    = time.time()
        inner = inner_estimator_fn(c)

        # Check n_features on full data first — skip if 0 (model would crash).
        try:
            sel_probe = SelectFromModel(inner_estimator_fn(c))
            sel_probe.fit(X_prep, y)
            n_sel = int(sel_probe.get_support().sum())
        except Exception:
            n_sel = 0

        if n_sel == 0:
            print(f"    C={c:<7}  n_features=0  → skipped (all coefs zeroed by L1)")
            results.append({"k": 0, "C": c, "mean": float("nan"),
                            "std": float("nan"), "label": label})
            continue

        sel  = SelectFromModel(inner_estimator_fn(c))
        pipe = _eval_pipeline(preprocessor, sel, "selector", ref_model)
        try:
            mean, std = _cv_score(pipe, X, y)
        except ValueError:
            mean, std = float("nan"), float("nan")

        results.append({"k": n_sel, "C": c, "mean": mean, "std": std, "label": label})
        print(f"    C={c:<7}  n_features≈{n_sel:<5}  "
              f"BCR={mean:.4f} ± {std:.4f}  ({time.time()-t0:.1f}s)")
    return results


# ====================================================================
# Main entry point
# ====================================================================

def run_feature_selection(preprocessor, best_model_pipeline, X_train, y_train,
                          baseline_bcr):
    """Evaluate all seven feature selection strategies and adopt the best one.

    A strategy is adopted only if it improves BCR by > 0.005 over baseline.
    All strategies are evaluated using LR as the reference model so that
    comparisons are consistent across methods.

    For single-model pipelines: the best selector is injected into the actual
    best model (SVM, MLP, etc.) for the final pipeline.
    For StackingClassifier (no .steps): reports analysis but keeps ensemble
    unchanged to avoid double-preprocessing inside base estimators.

    Parameters
    ----------
    preprocessor        : sklearn Pipeline (preprocessing steps)
    best_model_pipeline : best pipeline from Phase 4 (may be ensemble)
    X_train, y_train    : training data
    baseline_bcr        : float, Phase 4 best CV BCR

    Returns
    -------
    final_pipeline        : pipeline to use in Phase 6
    selection_description : human-readable string for reporting
    """
    is_ensemble = not hasattr(best_model_pipeline, "steps")

    # Feature selection evaluation always uses LR as the scoring model so that
    # ALL k values can be compared (some models like QDA fail when k > n_minority).
    # The winning selector is then applied to the actual best model for the final
    # pipeline, so evaluation and deployment use different (appropriate) classifiers.
    #
    # LR baseline: what LR scores on all features — this is the comparison bar.
    lr_scores = cross_val_score(
        Pipeline([("preprocessor", deepcopy(preprocessor)), ("model", _lr_ref())]),
        X_train, y_train,
        cv=get_cv(), scoring="balanced_accuracy", n_jobs=N_JOBS,
        error_score=np.nan,
    )
    lr_baseline = float(np.nanmean(lr_scores))
    ref_model   = None   # → _eval_pipeline always uses LR
    ref_baseline = lr_baseline

    actual_model_name = (best_model_pipeline.steps[-1][0]
                         if not is_ensemble else "Stacking Ensemble")
    print(f"  Best Phase 4 model   : {actual_model_name}  (BCR={baseline_bcr:.4f})")
    print(f"  LR baseline (all k)  : {lr_baseline:.4f}  ← evaluation reference")
    print(f"  Selectors scored with LR; winner applied to {actual_model_name} for final pipeline.")

    t0 = time.time()
    print(f"  Baseline BCR (all features): {ref_baseline:.4f}")
    print(f"  Improvement threshold to adopt a selector: 0.005")

    # ------------------------------------------------------------------
    # 1–3. Filter methods
    # ------------------------------------------------------------------
    fc_results = _eval_filter(preprocessor, X_train, y_train, f_classif,
                               "f_classif", ref_model)
    mi_results = _eval_filter(preprocessor, X_train, y_train, mutual_info_classif,
                               "mutual_info", ref_model)
    mw_results = _eval_filter(preprocessor, X_train, y_train, _mann_whitney_score,
                               "Mann-Whitney", ref_model)

    # ------------------------------------------------------------------
    # 4. PCA
    # ------------------------------------------------------------------
    pca_results = _eval_pca(preprocessor, X_train, y_train, ref_model)

    # ------------------------------------------------------------------
    # 5. Wrapper — RFE (ranker=LR, classifier=ref_model)
    # ------------------------------------------------------------------
    rfe_results = _eval_rfe(preprocessor, X_train, y_train, ref_model)

    # ------------------------------------------------------------------
    # 6–7. Embedded — L1 regularisation
    # ------------------------------------------------------------------
    l1lr_results = _eval_embedded(
        preprocessor, X_train, y_train,
        lambda c: LogisticRegression(
            penalty="l1", solver="liblinear", C=c,
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE,
        ),
        "Embedded L1-LR",
        ref_model,
    )
    l1svc_results = _eval_embedded(
        preprocessor, X_train, y_train,
        lambda c: LinearSVC(
            penalty="l1", dual=False, C=c,
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE,
        ),
        "Embedded L1-SVC",
        ref_model,
    )

    # ------------------------------------------------------------------
    # Collect all results and find the global best
    # ------------------------------------------------------------------
    all_results = fc_results + mi_results + mw_results + pca_results \
                  + rfe_results + l1lr_results + l1svc_results

    try:
        from plots import plot_feature_selection_results
        plot_feature_selection_results(
            fc_results, mi_results, mw_results,
            pca_results, rfe_results, l1lr_results, l1svc_results,
            baseline_bcr,
        )
    except Exception as exc:
        print(f"  [plot] Feature selection plot skipped: {exc}")

    valid = [r for r in all_results if not (isinstance(r["mean"], float)
                                             and np.isnan(r["mean"]))]
    if not valid:
        print("  All strategies failed. Keeping original pipeline.")
        return best_model_pipeline, "No feature selection (all strategies failed)"

    best        = max(valid, key=lambda r: r["mean"])
    improvement = best["mean"] - ref_baseline

    # Summary table
    strategies = [
        ("f_classif",       fc_results),
        ("mutual_info",     mi_results),
        ("Mann-Whitney",    mw_results),
        ("PCA",             pca_results),
        ("RFE (LR)",        rfe_results),
        ("Embedded L1-LR",  l1lr_results),
        ("Embedded L1-SVC", l1svc_results),
    ]
    print(f"\n  {'Strategy':<22} {'Best k':>7} {'LR+sel BCR':>11} {'vs LR':>8} {'vs '+actual_model_name[:12]:>15}")
    print("  " + "-" * 66)
    for name, res in strategies:
        valid_res = [r for r in res if not (isinstance(r["mean"], float)
                                            and np.isnan(r["mean"]))]
        if not valid_res:
            print(f"  {name:<22} {'—':>7} {'all k failed':>11}")
            continue
        b = max(valid_res, key=lambda r: r["mean"])
        delta_lr     = b["mean"] - lr_baseline
        delta_actual = b["mean"] - baseline_bcr
        marker = " ← BEST" if b is best else ""
        print(f"  {name:<22} {str(b['k']):>7} {b['mean']:>11.4f} {delta_lr:>+8.4f} {delta_actual:>+15.4f}{marker}")
    print("  " + "-" * 66)
    print(f"  {'LR (all features)':<22} {'all':>7} {lr_baseline:>11.4f}  (reference)")
    print(f"  {actual_model_name:<22} {'all':>7} {baseline_bcr:>11.4f}  (target to beat)")

    # ------------------------------------------------------------------
    # Decision: adopt or skip
    # ------------------------------------------------------------------
    THRESHOLD = 0.005

    # Adoption decision: the selector+LR must beat the ACTUAL best model (baseline_bcr),
    # not just LR alone. If selector+LR > actual_model, it will likely help even more
    # when combined with the actual model (which is already stronger than LR).
    improvement_vs_actual = best["mean"] - baseline_bcr

    if improvement_vs_actual <= THRESHOLD:
        print(f"\n  Best selector+LR = {best['mean']:.4f}  vs  {actual_model_name} = {baseline_bcr:.4f}")
        print(f"  Improvement vs actual model: {improvement_vs_actual:+.4f}  (threshold={THRESHOLD})")
        print("  No selector beats the actual model — keeping the best Phase 4 pipeline unchanged.")
        print(f"  Feature selection done in {time.time()-t0:.1f}s")
        return best_model_pipeline, "No feature selection (all features)"

    # Build the winning selector
    label      = best["label"]
    best_k     = best["k"]
    print(f"\n  Adopting: {label} with k={best_k}  (improvement={improvement:+.4f})")

    if label == "f_classif":
        sel, sel_name = SelectKBest(f_classif, k=best_k), "selector"
    elif label == "mutual_info":
        sel, sel_name = SelectKBest(mutual_info_classif, k=best_k), "selector"
    elif label == "Mann-Whitney":
        sel, sel_name = SelectKBest(_mann_whitney_score, k=best_k), "selector"
    elif label == "PCA":
        sel, sel_name = PCA(n_components=best_k, random_state=RANDOM_STATE), "pca"
    elif label == "RFE (LR)":
        sel, sel_name = RFE(
            estimator=LogisticRegression(
                class_weight="balanced", max_iter=1000,
                random_state=RANDOM_STATE, n_jobs=1,
            ),
            n_features_to_select=best_k, step=100,
        ), "rfe"
    elif label == "Embedded L1-LR":
        best_c = best["C"]
        sel, sel_name = SelectFromModel(LogisticRegression(
            penalty="l1", solver="liblinear", C=best_c,
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE,
        )), "selector"
    else:  # Embedded L1-SVC
        best_c = best["C"]
        sel, sel_name = SelectFromModel(LinearSVC(
            penalty="l1", dual=False, C=best_c,
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE,
        )), "selector"

    final_pipeline = _apply_pipeline(preprocessor, sel, sel_name, best_model_pipeline)

    if is_ensemble:
        desc = f"{label} k={best_k} + LR (ensemble kept for analysis only)"
    else:
        model_name = best_model_pipeline.steps[-1][0]
        desc = f"{label} k={best_k} + {model_name}"

    print(f"  Feature selection done in {time.time()-t0:.1f}s")
    return final_pipeline, desc
