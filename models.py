# models.py
# Phase 3 -- Baseline Models
# Phase 4 -- Model Selection and Hyperparameter Tuning
#
# All models are wrapped in full pipelines (preprocessor + model) so that
# no data leakage occurs during cross-validation.
#
# Mac Studio optimisations:
#   - n_jobs=N_JOBS uses all 16 CPU cores for parallelism
#   - XGBoost uses tree_method='hist' (fastest CPU method)
#   - nthread set explicitly to leverage all physical cores
#   - verbose=2 on RandomizedSearchCV prints each trial result

import time
import numpy as np
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)

try:
    import imblearn  # noqa: F401  -- presence check only
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("WARNING: imbalanced-learn not installed. SMOTE will be skipped.")

try:
    import xgboost  # noqa: F401  -- presence check only
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("WARNING: xgboost not installed. XGBoost model will be skipped.")

from config import (
    CV_N_SPLITS,
    CV_RANDOM_STATE,
    RANDOM_STATE,
    XGB_N_ITER,
    SVM_N_ITER,
    MLP_N_ITER,
    N_JOBS,
    ROBUSTNESS_SEEDS,
)


def _make_pipeline(preprocessor, model):
    """Build a pipeline with SMOTE (when imblearn is available) between preprocessor and model.

    Using ImbPipeline ensures SMOTE is applied only on the training fold during
    cross-validation, never on the validation fold — no data leakage.
    """
    if IMBLEARN_AVAILABLE:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        # imblearn does not allow a Pipeline as an intermediate step, so we
        # flatten the preprocessor's steps directly into the ImbPipeline.
        return ImbPipeline(
            [(name, deepcopy(step)) for name, step in preprocessor.steps]
            + [("smote", SMOTE(random_state=RANDOM_STATE)),
               ("model", model)]
        )
    return Pipeline([
        ("preprocessor", deepcopy(preprocessor)),
        ("model",        model),
    ])


def get_cv():
    """Return the standard StratifiedKFold splitter used throughout the project."""
    return StratifiedKFold(
        n_splits=CV_N_SPLITS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )


def _progress_bar(current, total, prefix="", bar_len=30):
    """Print an inline ASCII progress bar."""
    filled = int(bar_len * current / max(total, 1))
    bar    = "█" * filled + "░" * (bar_len - filled)
    pct    = 100 * current / max(total, 1)
    print(f"\r  {prefix} [{bar}] {pct:5.1f}%  ({current}/{total})", end="", flush=True)
    if current >= total:
        print()   # newline on completion


# -------------------------------------------------------------------
# Phase 3 -- Baselines
# -------------------------------------------------------------------

def _cv_with_progress(pipeline, X, y, cv, scoring, n_jobs, label):
    """cross_val_score wrapper that prints fold-level progress."""
    skf    = list(cv.split(X, y))
    n_folds = len(skf)
    scores  = []
    t_start = time.time()

    print(f"\n  Training {label} ({n_folds}-fold CV):")
    for fold_idx, (train_idx, val_idx) in enumerate(skf, 1):
        t_fold = time.time()
        fold_pipe = deepcopy(pipeline)
        fold_pipe.fit(X[train_idx], y[train_idx])
        from sklearn.metrics import balanced_accuracy_score
        preds = fold_pipe.predict(X[val_idx])
        score = balanced_accuracy_score(y[val_idx], preds)
        scores.append(score)
        elapsed = time.time() - t_fold
        print(f"    Fold {fold_idx}/{n_folds}  BCR={score:.4f}  ({elapsed:.1f}s)")

    arr = np.array(scores)
    total_elapsed = time.time() - t_start
    print(f"  → {label} done in {total_elapsed:.1f}s  |  "
          f"BCR = {arr.mean():.4f} ± {arr.std():.4f}")
    return arr


def run_baseline(preprocessor, X_train, y_train, is_imbalanced):
    """Train and cross-validate Logistic Regression and Random Forest baselines.

    Parameters
    ----------
    preprocessor  : fitted-compatible sklearn Pipeline (from preprocessing.py)
    X_train       : np.ndarray of shape (n_samples, n_features)
    y_train       : np.ndarray of int labels
    is_imbalanced : bool, whether the dataset is flagged as imbalanced

    Returns
    -------
    results : dict with keys 'lr' and 'rf', each containing 'mean' and 'std'
    lr_pipeline : the Logistic Regression pipeline (for reuse)
    rf_pipeline : the Random Forest pipeline (for reuse)
    """
    cv = get_cv()

    # Logistic Regression baseline.
    lr_pipeline = Pipeline([
        ("preprocessor", deepcopy(preprocessor)),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )),
    ])

    # Random Forest baseline — n_jobs uses all Mac Studio cores.
    rf_pipeline = Pipeline([
        ("preprocessor", deepcopy(preprocessor)),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
        )),
    ])

    lr_scores = _cv_with_progress(lr_pipeline, X_train, y_train, cv,
                                   "balanced_accuracy", N_JOBS, "Logistic Regression")
    rf_scores = _cv_with_progress(rf_pipeline, X_train, y_train, cv,
                                   "balanced_accuracy", N_JOBS, "Random Forest")

    print(f"\n  LR  BCR: {lr_scores.mean():.4f} +/- {lr_scores.std():.4f}")
    print(f"  RF  BCR: {rf_scores.mean():.4f} +/- {rf_scores.std():.4f}")

    if lr_scores.mean() < 0.65 and rf_scores.mean() < 0.65:
        print("  WARNING: Both baselines are below 0.65 BCR.")
        print("  Check data loading and label alignment before proceeding.")

    if lr_scores.mean() > 0.90:
        print("  NOTE: LR BCR > 0.90 -- data appears nearly linearly separable.")
        print("  Less time needed on non-linear models in Phase 4.")

    results = {
        "lr": {"mean": lr_scores.mean(), "std": lr_scores.std(), "scores": lr_scores},
        "rf": {"mean": rf_scores.mean(), "std": rf_scores.std(), "scores": rf_scores},
    }
    return results, lr_pipeline, rf_pipeline


# -------------------------------------------------------------------
# Phase 4 -- Hyperparameter Tuning
# -------------------------------------------------------------------

def tune_xgboost(preprocessor, X_train, y_train):
    """Tune XGBoost with RandomizedSearchCV.

    Mac Studio optimisations:
      - tree_method='hist': fastest tree construction on modern CPUs
      - nthread=-1: use all logical cores inside XGBoost
      - device='cpu': explicit (avoids deprecation warnings in XGB 2.x)

    Returns
    -------
    search : fitted RandomizedSearchCV object
    """
    if not XGB_AVAILABLE:
        print("  XGBoost not available -- skipping.")
        return None

    import xgboost as xgb

    print(f"\n  Tuning XGBoost  ({XGB_N_ITER} iterations x {CV_N_SPLITS} folds "
          f"= {XGB_N_ITER * CV_N_SPLITS} fits)...")
    t0 = time.time()

    xgb_pipeline = _make_pipeline(preprocessor, xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",      # fastest CPU tree builder
        device="cpu",
        nthread=-1,              # use all cores inside XGBoost
        random_state=RANDOM_STATE,
    ))

    xgb_param_dist = {
        "model__n_estimators"     : [200, 400, 600, 800, 1000],
        "model__max_depth"        : [3, 4, 5, 6, 7],
        "model__learning_rate"    : [0.005, 0.01, 0.05, 0.1, 0.2],
        "model__subsample"        : [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree" : [0.4, 0.5, 0.7, 0.9, 1.0],
        "model__scale_pos_weight" : [1, 3, 5, 7, 9, 12],   # ratio réel = 9
        "model__reg_alpha"        : [0, 0.01, 0.1, 0.5],
        "model__reg_lambda"       : [0.5, 1, 2, 5],
        "model__min_child_weight" : [1, 3, 5],
        "model__gamma"            : [0, 0.1, 0.3],
    }

    search = RandomizedSearchCV(
        xgb_pipeline, xgb_param_dist,
        n_iter=XGB_N_ITER,
        scoring="balanced_accuracy",
        cv=get_cv(),
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        refit=True,
        verbose=2,              # print each trial result
    )
    search.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"\n  XGBoost tuning done in {elapsed:.1f}s")
    print(f"  XGBoost best CV BCR: {search.best_score_:.4f}")
    print(f"  XGBoost best params:")
    for k, v in search.best_params_.items():
        print(f"    {k:<35} = {v}")

    try:
        from plots import plot_hyperparameter_search_results
        plot_hyperparameter_search_results(search, "XGBoost", "model__n_estimators",
                                           "model_03_xgb_search.png")
    except Exception:
        pass

    return search


def tune_svm(preprocessor, X_train, y_train, with_probability=True):
    """Tune SVM with RBF kernel using RandomizedSearchCV.

    Parameters
    ----------
    with_probability : bool
        If True, fits the SVM with probability=True (needed for soft voting).

    Returns
    -------
    search : fitted RandomizedSearchCV object
    """
    print(f"\n  Tuning SVM (RBF kernel)  ({SVM_N_ITER} iterations × {CV_N_SPLITS} folds "
          f"= {SVM_N_ITER * CV_N_SPLITS} fits)...")
    t0 = time.time()

    svm_pipeline = _make_pipeline(preprocessor, SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=with_probability,
        random_state=RANDOM_STATE,
        cache_size=2000,         # larger cache improves speed with 1024-d data
    ))

    svm_param_dist = {
        "model__C"     : np.logspace(-2, 3, 30),
        "model__gamma" : np.logspace(-4, 0, 30),
    }

    search = RandomizedSearchCV(
        svm_pipeline, svm_param_dist,
        n_iter=SVM_N_ITER,
        scoring="balanced_accuracy",
        cv=get_cv(),
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        refit=True,
        verbose=2,
    )
    search.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"\n  SVM tuning done in {elapsed:.1f}s")
    print(f"  SVM best CV BCR: {search.best_score_:.4f}")
    print(f"  SVM best params:")
    for k, v in search.best_params_.items():
        print(f"    {k:<35} = {v:.6g}")

    try:
        from plots import plot_hyperparameter_search_results
        plot_hyperparameter_search_results(search, "SVM (RBF)", "model__C",
                                           "model_04_svm_search.png")
    except Exception:
        pass

    return search


def tune_mlp(preprocessor, X_train, y_train):
    """Tune MLP with RandomizedSearchCV.

    Returns
    -------
    search : fitted RandomizedSearchCV object
    """
    print(f"\n  Tuning MLP  ({MLP_N_ITER} iterations × {CV_N_SPLITS} folds "
          f"= {MLP_N_ITER * CV_N_SPLITS} fits)...")
    t0 = time.time()

    mlp_pipeline = _make_pipeline(preprocessor, MLPClassifier(
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
        verbose=False,
    ))

    mlp_param_dist = {
        "model__hidden_layer_sizes" : [
            (512,), (256,), (128,),
            (256, 128), (512, 256), (128, 64),
            (256, 128, 64), (512, 256, 128),
        ],
        "model__alpha"              : [0.0001, 0.001, 0.01, 0.1],
        "model__learning_rate_init" : [0.0005, 0.001, 0.005, 0.01],
        "model__activation"         : ["relu", "tanh"],
        "model__learning_rate"      : ["constant", "adaptive"],
    }

    # n_jobs=1: MLPClassifier's internal BLAS threading conflicts with outer parallelism.
    search = RandomizedSearchCV(
        mlp_pipeline, mlp_param_dist,
        n_iter=MLP_N_ITER,
        scoring="balanced_accuracy",
        cv=get_cv(),
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        verbose=2,
    )
    search.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"\n  MLP tuning done in {elapsed:.1f}s")
    print(f"  MLP best CV BCR: {search.best_score_:.4f}")
    print(f"  MLP best params:")
    for k, v in search.best_params_.items():
        print(f"    {k:<35} = {v}")

    try:
        from plots import plot_hyperparameter_search_results
        plot_hyperparameter_search_results(search, "MLP", "model__hidden_layer_sizes",
                                           "model_05_mlp_search.png")
    except Exception:
        pass

    return search


def build_stacking_ensemble(xgb_search, svm_search, mlp_search,
                            preprocessor, X_train, y_train):
    """Combine the three best estimators in a StackingClassifier.

    Each base pipeline (preprocessor + SMOTE + model) is already self-contained.
    A LogisticRegression meta-learner is trained on the out-of-fold predictions
    produced by those base pipelines, which is strictly more powerful than
    the simple probability average used by VotingClassifier.

    Returns
    -------
    ensemble : StackingClassifier (not yet re-fitted on all data here)
    ensemble_scores : np.ndarray of per-fold BCR scores
    """
    print("\n  Building stacking ensemble (LogReg meta-learner)...")
    t0 = time.time()

    estimators = []
    if xgb_search is not None:
        estimators.append(("xgb", xgb_search.best_estimator_))
    if svm_search is not None:
        estimators.append(("svm", svm_search.best_estimator_))
    if mlp_search is not None:
        estimators.append(("mlp", mlp_search.best_estimator_))

    if len(estimators) < 2:
        print("  Not enough models for stacking -- returning None.")
        return None, np.array([])

    meta_learner = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=get_cv(),
        n_jobs=1,
        passthrough=False,
    )

    print(f"  Evaluating stacking ensemble over {CV_N_SPLITS} folds...")
    ensemble_scores = cross_val_score(
        ensemble, X_train, y_train,
        cv=get_cv(), scoring="balanced_accuracy", n_jobs=1,
        verbose=1,
    )

    elapsed = time.time() - t0
    print(f"\n  Stacking ensemble done in {elapsed:.1f}s")
    print(f"  Stacking Ensemble BCR: {ensemble_scores.mean():.4f} +/- {ensemble_scores.std():.4f}")
    return ensemble, ensemble_scores


def select_best_model(baseline_results, xgb_search, svm_search,
                      mlp_search, ensemble, ensemble_scores):
    """Compare all candidates and return the best pipeline.

    Selection rules:
    - If the ensemble BCR is more than 0.005 better than the best single
      model, prefer the ensemble.
    - Otherwise, prefer the single model with the highest mean BCR.
    - If two models are within 0.005 of each other, prefer the one with
      lower standard deviation (more stable).

    Returns
    -------
    best_pipeline : the chosen pipeline (not yet fitted on all data)
    best_name     : string name of the chosen model
    summary_table : list of dicts for printing / plotting
    """
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON TABLE")
    print(f"  {'Model':<30} {'CV BCR Mean':>12} {'CV BCR Std':>11}")
    print("  " + "-" * 55)

    candidates = []

    def _add(name, search_or_scores, is_ensemble=False):
        if search_or_scores is None:
            return
        if is_ensemble:
            scores   = search_or_scores
            mean     = scores.mean()
            std      = scores.std()
            pipeline = ensemble
        else:
            mean     = search_or_scores.best_score_
            std      = search_or_scores.cv_results_["std_test_score"][
                           search_or_scores.best_index_]
            pipeline = search_or_scores.best_estimator_

        candidates.append({"name": name, "mean": mean, "std": std, "pipeline": pipeline})
        print(f"  {name:<30} {mean:>12.4f} {std:>11.4f}")

    for key, label in [("lr", "Logistic Regression"), ("rf", "Random Forest")]:
        r = baseline_results[key]
        candidates.append({"name": label, "mean": r["mean"], "std": r["std"], "pipeline": None})
        print(f"  {label:<30} {r['mean']:>12.4f} {r['std']:>11.4f}")

    _add("XGBoost (tuned)", xgb_search)
    _add("SVM RBF (tuned)", svm_search)
    _add("MLP (tuned)",     mlp_search)

    if ensemble is not None and len(ensemble_scores) > 0:
        _add("Stacking Ensemble", ensemble_scores, is_ensemble=True)

    print("  " + "-" * 55)

    # Generate model comparison plot (all candidates that have a score).
    try:
        from plots import plot_model_comparison
        plot_model_comparison([c for c in candidates if c["mean"] > 0])
    except Exception:
        pass

    runnable = [c for c in candidates if c["pipeline"] is not None]
    if not runnable:
        raise RuntimeError("No tuned models available. Check Phase 4 execution.")

    best_single = max(runnable, key=lambda c: c["mean"])
    ensemble_candidate = next(
        (c for c in runnable if c["name"] == "Stacking Ensemble"), None
    )

    if (ensemble_candidate is not None and
            ensemble_candidate["mean"] > best_single["mean"] + 0.005):
        chosen = ensemble_candidate
    else:
        threshold      = best_single["mean"] - 0.005
        close_candidates = [c for c in runnable if c["mean"] >= threshold]
        chosen         = min(close_candidates, key=lambda c: c["std"])

    print(f"\n  Chosen model: {chosen['name']} "
          f"(BCR={chosen['mean']:.4f}, std={chosen['std']:.4f})")

    return chosen["pipeline"], chosen["name"], candidates


def evaluate_robustness(pipeline, X_train, y_train):
    """Estimate model stability by running CV with multiple random seeds.

    Each seed produces a different StratifiedKFold split; the spread of
    mean BCR across seeds reflects how sensitive the model is to data
    partitioning (i.e. luck in the CV draw).

    A small robustness_score (std across seeds) means the estimate is
    stable and trustworthy; a large value signals high sensitivity to
    the choice of folds.

    Parameters
    ----------
    pipeline  : sklearn-compatible Pipeline (not yet fitted)
    X_train   : np.ndarray
    y_train   : np.ndarray of int

    Returns
    -------
    mean_across_seeds  : float
    robustness_score   : float, std of mean BCR across seeds
    seed_results       : list of dicts {"seed", "mean", "std"}
    """
    print(f"\n  Multi-seed robustness evaluation ({len(ROBUSTNESS_SEEDS)} seeds):")
    t0 = time.time()

    seed_results = []
    for seed in ROBUSTNESS_SEEDS:
        cv = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=seed)
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="balanced_accuracy", n_jobs=1,
        )
        seed_results.append({"seed": seed, "mean": float(scores.mean()),
                              "std": float(scores.std())})
        print(f"    seed={seed:<6}  BCR={scores.mean():.4f} ± {scores.std():.4f}")

    means             = np.array([r["mean"] for r in seed_results])
    mean_across_seeds = float(means.mean())
    robustness_score  = float(means.std())   # low = stable, high = sensitive

    print(f"\n  Mean BCR across seeds   : {mean_across_seeds:.4f}")
    print(f"  Robustness score (std)  : {robustness_score:.4f}  "
          f"({'stable' if robustness_score < 0.01 else 'moderate' if robustness_score < 0.02 else 'unstable'})")
    print(f"  Multi-seed evaluation done in {time.time() - t0:.1f}s")

    return mean_across_seeds, robustness_score, seed_results
