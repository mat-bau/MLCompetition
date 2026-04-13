# models.py
# Phase 3 -- Baseline Models
# Phase 4 -- Model Selection and Hyperparameter Tuning
#
# All models are wrapped in full pipelines (preprocessor + model) so that
# no data leakage occurs during cross-validation.

import numpy as np
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
)

try:
    import xgboost as xgb
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
)


def get_cv():
    """Return the standard StratifiedKFold splitter used throughout the project."""
    return StratifiedKFold(
        n_splits=CV_N_SPLITS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )


# -------------------------------------------------------------------
# Phase 3 -- Baselines
# -------------------------------------------------------------------

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

    cw_lr = "balanced"
    cw_rf = "balanced_subsample"

    # Logistic Regression baseline.
    # max_iter=2000 avoids convergence warnings on 1024-dimensional data.
    lr_pipeline = Pipeline([
        ("preprocessor", deepcopy(preprocessor)),
        ("model", LogisticRegression(
            class_weight=cw_lr,
            max_iter=2000,
            random_state=RANDOM_STATE,
        )),
    ])

    # Random Forest baseline with no tuning.
    rf_pipeline = Pipeline([
        ("preprocessor", deepcopy(preprocessor)),
        ("model", RandomForestClassifier(
            n_estimators=100,
            class_weight=cw_rf,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    print("Training Logistic Regression baseline...")
    lr_scores = cross_val_score(
        lr_pipeline, X_train, y_train,
        cv=cv, scoring="balanced_accuracy", n_jobs=-1,
    )

    print("Training Random Forest baseline...")
    rf_scores = cross_val_score(
        rf_pipeline, X_train, y_train,
        cv=cv, scoring="balanced_accuracy", n_jobs=-1,
    )

    print(f"LR  BCR: {lr_scores.mean():.4f} +/- {lr_scores.std():.4f}")
    print(f"RF  BCR: {rf_scores.mean():.4f} +/- {rf_scores.std():.4f}")

    # Decision point: if both baselines are below 0.65, raise a warning.
    if lr_scores.mean() < 0.65 and rf_scores.mean() < 0.65:
        print("WARNING: Both baselines are below 0.65 BCR.")
        print("Check data loading and label alignment before proceeding.")

    # Decision point: if LR BCR > 0.90, the data is nearly linearly separable.
    if lr_scores.mean() > 0.90:
        print("NOTE: LR BCR > 0.90 -- data appears nearly linearly separable.")
        print("Less time needed on non-linear models in Phase 4.")

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

    Returns
    -------
    search : fitted RandomizedSearchCV object
    """
    if not XGB_AVAILABLE:
        print("XGBoost not available -- skipping.")
        return None

    print("\nTuning XGBoost...")

    xgb_pipeline = Pipeline([
        ("preprocessor", deepcopy(preprocessor)),
        ("model", xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    # Search space covers the most impactful XGBoost hyperparameters.
    # n_estimators and learning_rate interact: more trees need a lower rate.
    # scale_pos_weight handles class imbalance inside XGBoost.
    xgb_param_dist = {
        "model__n_estimators"     : [200, 400, 600, 800],
        "model__max_depth"        : [3, 4, 5, 6],
        "model__learning_rate"    : [0.01, 0.05, 0.1, 0.2],
        "model__subsample"        : [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree" : [0.5, 0.7, 0.9, 1.0],
        "model__scale_pos_weight" : [1, 2, 3],
        "model__reg_alpha"        : [0, 0.01, 0.1],
        "model__reg_lambda"       : [1, 2, 5],
    }

    search = RandomizedSearchCV(
        xgb_pipeline, xgb_param_dist,
        n_iter=XGB_N_ITER,
        scoring="balanced_accuracy",
        cv=get_cv(),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_score = search.best_score_
    print(f"XGBoost best CV BCR: {best_score:.4f}")
    print(f"XGBoost best params: {search.best_params_}")
    return search


def tune_svm(preprocessor, X_train, y_train, with_probability=True):
    """Tune SVM with RBF kernel using RandomizedSearchCV.

    Parameters
    ----------
    with_probability : bool
        If True, fits the SVM with probability=True (needed for soft voting).
        This makes training slower due to Platt scaling.

    Returns
    -------
    search : fitted RandomizedSearchCV object
    """
    print("\nTuning SVM (RBF kernel)...")

    svm_pipeline = Pipeline([
        ("preprocessor", deepcopy(preprocessor)),
        ("model", SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=with_probability,
            random_state=RANDOM_STATE,
        )),
    ])

    # C and gamma are searched on a log scale because their effect is
    # multiplicative. A wide range is tested first.
    svm_param_dist = {
        "model__C"     : np.logspace(-2, 3, 20),
        "model__gamma" : np.logspace(-4, 0, 20),
    }

    search = RandomizedSearchCV(
        svm_pipeline, svm_param_dist,
        n_iter=SVM_N_ITER,
        scoring="balanced_accuracy",
        cv=get_cv(),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_score = search.best_score_
    print(f"SVM best CV BCR: {best_score:.4f}")
    print(f"SVM best params: {search.best_params_}")
    return search


def tune_mlp(preprocessor, X_train, y_train):
    """Tune MLP with RandomizedSearchCV.

    Returns
    -------
    search : fitted RandomizedSearchCV object
    """
    print("\nTuning MLP...")

    mlp_pipeline = Pipeline([
        ("preprocessor", deepcopy(preprocessor)),
        ("model", MLPClassifier(
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        )),
    ])

    # Three architectural families: shallow-wide, deep-narrow, two-layer.
    mlp_param_dist = {
        "model__hidden_layer_sizes" : [
            (512,), (256,), (128,),
            (256, 128), (512, 256), (128, 64),
            (256, 128, 64),
        ],
        "model__alpha"              : [0.0001, 0.001, 0.01, 0.1],
        "model__learning_rate_init" : [0.001, 0.005, 0.01],
        "model__activation"         : ["relu", "tanh"],
    }

    # n_jobs=1 because MLPClassifier's internal parallelism conflicts with
    # outer parallelism in RandomizedSearchCV.
    search = RandomizedSearchCV(
        mlp_pipeline, mlp_param_dist,
        n_iter=MLP_N_ITER,
        scoring="balanced_accuracy",
        cv=get_cv(),
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        verbose=1,
    )
    search.fit(X_train, y_train)

    best_score = search.best_score_
    print(f"MLP best CV BCR: {best_score:.4f}")
    print(f"MLP best params: {search.best_params_}")
    return search


def build_voting_ensemble(xgb_search, svm_search, mlp_search,
                          preprocessor, X_train, y_train):
    """Combine the three best estimators in a soft VotingClassifier.

    The sub-estimators already contain preprocessors in their pipelines,
    so no additional preprocessor is needed at the ensemble level.

    Returns
    -------
    ensemble : fitted VotingClassifier
    ensemble_scores : np.ndarray of per-fold BCR scores
    """
    print("\nBuilding soft voting ensemble...")

    estimators = []
    if xgb_search is not None:
        estimators.append(("xgb", xgb_search.best_estimator_))
    if svm_search is not None:
        estimators.append(("svm", svm_search.best_estimator_))
    if mlp_search is not None:
        estimators.append(("mlp", mlp_search.best_estimator_))

    if len(estimators) < 2:
        print("Not enough models for an ensemble -- returning None.")
        return None, np.array([])

    ensemble = VotingClassifier(
        estimators=estimators,
        voting="soft",
        n_jobs=1,
    )

    # Evaluate the ensemble with cross-validation.
    # n_jobs=1 because sub-estimators may already use threading internally.
    ensemble_scores = cross_val_score(
        ensemble, X_train, y_train,
        cv=get_cv(), scoring="balanced_accuracy", n_jobs=1,
    )

    print(f"Ensemble BCR: {ensemble_scores.mean():.4f} +/- {ensemble_scores.std():.4f}")
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
    summary_table : list of dicts for printing
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print(f"{'Model':<30} {'CV BCR Mean':>12} {'CV BCR Std':>11}")
    print("-" * 55)

    candidates = []

    def _add(name, search_or_scores, is_ensemble=False):
        """Helper to add a candidate to the comparison list."""
        if search_or_scores is None:
            return
        if is_ensemble:
            scores = search_or_scores
            pipeline = ensemble
        else:
            scores = search_or_scores.cv_results_["mean_test_score"]
            # best_score_ is the single best mean; compute std from all results
            mean = search_or_scores.best_score_
            # std across the CV folds of the best run is not directly stored
            # so we report the std of all trials as a proxy
            std = search_or_scores.cv_results_["std_test_score"][
                search_or_scores.best_index_
            ]
            pipeline = search_or_scores.best_estimator_
            candidates.append({
                "name": name,
                "mean": mean,
                "std": std,
                "pipeline": pipeline,
            })
            print(f"{name:<30} {mean:>12.4f} {std:>11.4f}")
            return

        mean = scores.mean()
        std  = scores.std()
        candidates.append({
            "name": name,
            "mean": mean,
            "std": std,
            "pipeline": pipeline,
        })
        print(f"{name:<30} {mean:>12.4f} {std:>11.4f}")

    # Add baselines.
    for key, label in [("lr", "Logistic Regression"), ("rf", "Random Forest")]:
        r = baseline_results[key]
        candidates.append({
            "name": label,
            "mean": r["mean"],
            "std": r["std"],
            "pipeline": None,  # pipelines from run_baseline are passed separately
        })
        print(f"{label:<30} {r['mean']:>12.4f} {r['std']:>11.4f}")

    # Add tuned models.
    _add("XGBoost (tuned)", xgb_search)
    _add("SVM RBF (tuned)", svm_search)
    _add("MLP (tuned)",     mlp_search)

    if ensemble is not None and len(ensemble_scores) > 0:
        _add("Voting Ensemble", ensemble_scores, is_ensemble=True)

    print("-" * 55)

    # Filter out candidates with no pipeline (baselines without refit).
    runnable = [c for c in candidates if c["pipeline"] is not None]

    if not runnable:
        raise RuntimeError("No tuned models available. Check Phase 4 execution.")

    # Find the best single model by mean BCR.
    best_single = max(runnable, key=lambda c: c["mean"])

    # Check if the ensemble is meaningfully better.
    ensemble_candidate = next(
        (c for c in runnable if c["name"] == "Voting Ensemble"), None
    )

    if (ensemble_candidate is not None and
            ensemble_candidate["mean"] > best_single["mean"] + 0.005):
        chosen = ensemble_candidate
    else:
        # Among candidates within 0.005 of the best, pick lowest std.
        threshold = best_single["mean"] - 0.005
        close_candidates = [c for c in runnable if c["mean"] >= threshold]
        chosen = min(close_candidates, key=lambda c: c["std"])

    print(f"\nChosen model: {chosen['name']} "
          f"(BCR={chosen['mean']:.4f}, std={chosen['std']:.4f})")

    return chosen["pipeline"], chosen["name"], candidates