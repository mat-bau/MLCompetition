# models.py
# Phase 3 -- Baseline Models
# Phase 4 -- Model Selection and Hyperparameter Tuning
#
# All models are wrapped in full pipelines (preprocessor + model) so that
# no data leakage occurs during cross-validation.
#
# Mac Studio optimisations:
#   - n_jobs=N_JOBS (-1) uses all 16 CPU cores for parallel CV
#   - XGBoost uses tree_method='hist' (fastest CPU tree builder)
#   - nthread=-1 leverages all physical cores inside XGBoost
#   - verbose=2 on RandomizedSearchCV prints each trial result
#
# Model catalogue (Phase 3 baselines + Phase 4 tuned):
#   Baselines : Logistic Regression, Random Forest, GaussianNB, Extra Trees
#   Tuned     : XGBoost, SVM (RBF), MLP, Random Forest, HistGradientBoosting
#   Ensemble  : StackingClassifier (LogReg meta-learner)
#
# Note on Bayesian Neural Networks: proper BNNs (pyro / torch-based) require
# libraries not in this venv and do not scale to 1024 features × 3000 samples
# without GPU. GaussianNB is used as the practical Bayesian probabilistic baseline.

import time
import numpy as np
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.decomposition import PCA
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
    RF_N_ITER,
    GB_N_ITER,
    GNB_N_ITER,
    LDA_N_ITER,
    QDA_N_ITER,
    RF_EVOLUTION_STEPS,
    N_JOBS,
    ROBUSTNESS_SEEDS,
)


# ====================================================================
# Pipeline helpers
# ====================================================================

def _make_pipeline(preprocessor, model, pca_n_components=None):
    """Build a pipeline with optional PCA and SMOTE (when imblearn is available).

    Using ImbPipeline ensures SMOTE is applied only on the training fold
    during cross-validation, never on the validation fold — no data leakage.
    When imblearn is absent, a plain sklearn Pipeline is returned.

    Parameters
    ----------
    pca_n_components : int or None
        If not None, insert a PCA dimensionality-reduction step after
        preprocessing but before SMOTE. Useful for MLP to reduce the
        1024-feature input to a manageable size.
    """
    if IMBLEARN_AVAILABLE:
        from imblearn.pipeline import Pipeline as ImbPipeline
        from imblearn.over_sampling import SMOTE
        # Flatten preprocessor steps into the ImbPipeline (imblearn requirement).
        steps = [(name, deepcopy(step)) for name, step in preprocessor.steps]
        if pca_n_components is not None:
            steps.append(("pca", PCA(n_components=pca_n_components,
                                     random_state=RANDOM_STATE)))
        steps += [("smote", SMOTE(random_state=RANDOM_STATE)), ("model", model)]
        return ImbPipeline(steps)

    steps = [("preprocessor", deepcopy(preprocessor))]
    if pca_n_components is not None:
        steps.append(("pca", PCA(n_components=pca_n_components,
                                  random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps)


def get_cv():
    """Return the standard StratifiedKFold splitter used throughout the project."""
    return StratifiedKFold(
        n_splits=CV_N_SPLITS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )


def _cv_with_progress(pipeline, X, y, cv, label):
    """cross_val_score wrapper that prints fold-level progress and timing."""
    skf     = list(cv.split(X, y))
    n_folds = len(skf)
    scores  = []
    t_start = time.time()

    print(f"\n  Training {label} ({n_folds}-fold CV):")
    for fold_idx, (train_idx, val_idx) in enumerate(skf, 1):
        t_fold    = time.time()
        fold_pipe = deepcopy(pipeline)
        fold_pipe.fit(X[train_idx], y[train_idx])
        from sklearn.metrics import balanced_accuracy_score
        preds = fold_pipe.predict(X[val_idx])
        score = balanced_accuracy_score(y[val_idx], preds)
        scores.append(score)
        elapsed = time.time() - t_fold
        print(f"    Fold {fold_idx}/{n_folds}  BCR={score:.4f}  ({elapsed:.1f}s)")

    arr           = np.array(scores)
    total_elapsed = time.time() - t_start
    print(f"  → {label} done in {total_elapsed:.1f}s  |  "
          f"BCR = {arr.mean():.4f} ± {arr.std():.4f}")
    return arr


# ====================================================================
# Phase 3 -- Baselines
# ====================================================================

def run_baseline(preprocessor, X_train, y_train, is_imbalanced):
    """Train and cross-validate four quick baselines.

    Baselines (no hyperparameter tuning):
    - Logistic Regression    : linear, probabilistic, interpretable
    - Random Forest (200)    : non-linear ensemble reference
    - GaussianNB             : Bayesian probabilistic model (fast lower bound)
    - Extra Trees (200)      : lower-variance ensemble; often beats RF

    Standardisation note:
        Scaler choice (RobustScaler vs StandardScaler) is made in Phase 2 based
        on the outlier fraction measured by EDA. RobustScaler is selected on this
        dataset (606/1024 features have at least one outlier at 5σ). The
        preprocessor passed in already encodes this decision — no extra scaling
        is applied here.

    Parameters
    ----------
    preprocessor  : sklearn Pipeline (from preprocessing.py)
    X_train       : np.ndarray of shape (n_samples, n_features)
    y_train       : np.ndarray of int labels
    is_imbalanced : bool (used for future extension; all baselines already use
                    class_weight='balanced')

    Returns
    -------
    results    : dict keyed by model short-name, each value = {mean, std, scores}
    pipelines  : dict keyed by model short-name → pipeline object
    """
    cv = get_cv()

    lr_pipeline = Pipeline([
        ("preprocessor", deepcopy(preprocessor)),
        ("model", LogisticRegression(
            class_weight="balanced", max_iter=2000,
            random_state=RANDOM_STATE, n_jobs=N_JOBS,
        )),
    ])

    rf_pipeline = Pipeline([
        ("preprocessor", deepcopy(preprocessor)),
        ("model", RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=N_JOBS,
        )),
    ])

    gnb_pipeline = Pipeline([
        ("preprocessor", deepcopy(preprocessor)),
        ("model", GaussianNB()),
    ])

    et_pipeline = Pipeline([
        ("preprocessor", deepcopy(preprocessor)),
        ("model", ExtraTreesClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=N_JOBS,
        )),
    ])

    lr_scores  = _cv_with_progress(lr_pipeline,  X_train, y_train, cv, "Logistic Regression")
    rf_scores  = _cv_with_progress(rf_pipeline,  X_train, y_train, cv, "Random Forest (200 trees)")
    gnb_scores = _cv_with_progress(gnb_pipeline, X_train, y_train, cv, "GaussianNB (Bayesian)")
    et_scores  = _cv_with_progress(et_pipeline,  X_train, y_train, cv, "Extra Trees (200 trees)")

    print(f"\n  Baseline summary:")
    for name, scores in [("LR", lr_scores), ("RF", rf_scores),
                          ("GNB", gnb_scores), ("ET", et_scores)]:
        print(f"    {name:<4}  BCR = {scores.mean():.4f} ± {scores.std():.4f}")

    if lr_scores.mean() < 0.65 and rf_scores.mean() < 0.65:
        print("  WARNING: Both LR and RF baselines are below 0.65 BCR.")
        print("  Check data loading and label alignment before proceeding.")

    if lr_scores.mean() > 0.90:
        print("  NOTE: LR BCR > 0.90 — data appears nearly linearly separable.")

    results = {
        "lr" : {"mean": lr_scores.mean(),  "std": lr_scores.std(),  "scores": lr_scores},
        "rf" : {"mean": rf_scores.mean(),  "std": rf_scores.std(),  "scores": rf_scores},
        "gnb": {"mean": gnb_scores.mean(), "std": gnb_scores.std(), "scores": gnb_scores},
        "et" : {"mean": et_scores.mean(),  "std": et_scores.std(),  "scores": et_scores},
    }
    pipelines = {
        "lr": lr_pipeline, "rf": rf_pipeline,
        "gnb": gnb_pipeline, "et": et_pipeline,
    }
    return results, pipelines


# ====================================================================
# Phase 4 -- Hyperparameter Tuning
# ====================================================================

def tune_xgboost(preprocessor, X_train, y_train):
    """Tune XGBoost with RandomizedSearchCV (200 iterations).

    Mac Studio optimisations:
      - tree_method='hist' : fastest tree construction on modern CPUs
      - nthread=-1         : use all logical cores inside XGBoost
      - device='cpu'       : explicit (avoids deprecation in XGB 2.x)

    Returns
    -------
    search : fitted RandomizedSearchCV object, or None if XGBoost unavailable
    """
    if not XGB_AVAILABLE:
        print("  XGBoost not available -- skipping.")
        return None

    import xgboost as xgb

    print(f"\n  Tuning XGBoost  ({XGB_N_ITER} iterations × {CV_N_SPLITS} folds "
          f"= {XGB_N_ITER * CV_N_SPLITS} fits)...")
    t0 = time.time()

    xgb_pipeline = _make_pipeline(preprocessor, xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cpu",
        nthread=-1,
        random_state=RANDOM_STATE,
    ))

    xgb_param_dist = {
        "model__n_estimators"     : [100, 200, 400, 600, 800, 1000, 1500],
        "model__max_depth"        : [3, 4, 5, 6, 7, 8],
        "model__learning_rate"    : [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.2],
        "model__subsample"        : [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree" : [0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0],
        "model__scale_pos_weight" : [1, 3, 5, 7, 9, 12],   # class imbalance ratio ≈ 9
        "model__reg_alpha"        : [0, 0.001, 0.01, 0.1, 0.5, 1.0],
        "model__reg_lambda"       : [0.1, 0.5, 1, 2, 5, 10],
        "model__min_child_weight" : [1, 3, 5, 10],
        "model__gamma"            : [0, 0.05, 0.1, 0.3, 0.5],
        "model__max_bin"          : [128, 256, 512],
    }

    search = RandomizedSearchCV(
        xgb_pipeline, xgb_param_dist,
        n_iter=XGB_N_ITER,
        scoring="balanced_accuracy",
        cv=get_cv(),
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        refit=True,
        verbose=2,
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
    """Tune SVM with RBF kernel using RandomizedSearchCV (150 iterations).

    C and gamma are sampled on a logarithmic scale covering 7 decades each.
    Probability calibration (Platt scaling) is enabled for soft voting.

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
        cache_size=2000,   # large cache = faster on high-d data (Mac Studio has plenty of RAM)
    ))

    svm_param_dist = {
        "model__C"     : np.logspace(-3, 4, 50),   # 0.001 → 10 000
        "model__gamma" : np.logspace(-5, 1, 50),   # 1e-5  →  10
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
        from plots import plot_hyperparameter_search_results, plot_cv_scatter
        plot_hyperparameter_search_results(search, "SVM (RBF)", "model__C",
                                           "model_04_svm_search.png")
        plot_cv_scatter(search, "SVM (RBF)", "model__C", "model__gamma",
                        "model_04b_svm_cv_scatter.png")
    except Exception:
        pass

    return search


def tune_mlp(preprocessor, X_train, y_train):
    """Tune MLP with RandomizedSearchCV (150 iterations).

    A PCA(n_components=200) step is inserted before the MLP to reduce the
    1024-feature input. This prevents the MLP from overfitting on the raw
    high-dimensional space, stabilises training, and allows smaller & faster
    architectures. The previous best arch (1024,) was essentially a linear
    layer on raw features; with PCA the MLP can exploit non-linear structure.

    n_jobs=1: MLPClassifier uses BLAS internally; parallelising the outer
    CV loop while BLAS uses multiple threads causes resource contention on
    macOS and degrades performance.

    Returns
    -------
    search : fitted RandomizedSearchCV object
    """
    print(f"\n  Tuning MLP+PCA  ({MLP_N_ITER} iterations × {CV_N_SPLITS} folds "
          f"= {MLP_N_ITER * CV_N_SPLITS} fits)...")
    t0 = time.time()

    # PCA to 200 components before SMOTE and MLP.
    mlp_pipeline = _make_pipeline(preprocessor, MLPClassifier(
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_STATE,
        verbose=False,
    ), pca_n_components=200)

    mlp_param_dist = {
        "model__hidden_layer_sizes" : [
            (200,), (128,), (64,),
            (200, 100), (128, 64), (100, 50),
            (200, 100, 50), (128, 64, 32), (100, 100, 50),
            (200, 200), (100, 100),
        ],
        "model__alpha"              : [1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.5],
        "model__learning_rate_init" : [1e-4, 5e-4, 0.001, 0.005, 0.01, 0.02],
        "model__activation"         : ["relu", "tanh"],
        "model__learning_rate"      : ["constant", "adaptive"],
        "model__batch_size"         : [32, 64, 128, 256, "auto"],
    }

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


# -------------------------------------------------------------------
# RF evolution helper
# -------------------------------------------------------------------

def _rf_evolution_curve(preprocessor, X_train, y_train):
    """Train RF at increasing n_estimators to show BCR growth.

    Used to answer: 'at which point do we stop gaining from adding more trees?'

    Returns
    -------
    steps      : list of int  (n_estimators values)
    bcr_means  : list of float (mean CV BCR at each step)
    """
    print(f"\n  RF evolution curve: {RF_EVOLUTION_STEPS} trees...")
    t0     = time.time()
    cv     = get_cv()
    means  = []

    for n in RF_EVOLUTION_STEPS:
        pipe = Pipeline([
            ("preprocessor", deepcopy(preprocessor)),
            ("model", RandomForestClassifier(
                n_estimators=n, class_weight="balanced_subsample",
                random_state=RANDOM_STATE, n_jobs=N_JOBS,
            )),
        ])
        scores = cross_val_score(pipe, X_train, y_train,
                                  cv=cv, scoring="balanced_accuracy", n_jobs=N_JOBS)
        means.append(float(scores.mean()))
        print(f"    n_estimators={n:<5}  BCR={scores.mean():.4f} ± {scores.std():.4f}")

    print(f"  RF evolution done in {time.time() - t0:.1f}s")

    try:
        from plots import plot_rf_evolution
        plot_rf_evolution(RF_EVOLUTION_STEPS, means)
    except Exception:
        pass

    return RF_EVOLUTION_STEPS, means


def tune_random_forest(preprocessor, X_train, y_train):
    """Tune Random Forest with RandomizedSearchCV (120 iterations).

    Also runs an evolution curve to show BCR vs n_estimators before tuning.

    Returns
    -------
    search : fitted RandomizedSearchCV object
    """
    print(f"\n  Tuning Random Forest  ({RF_N_ITER} iterations × {CV_N_SPLITS} folds "
          f"= {RF_N_ITER * CV_N_SPLITS} fits)...")
    t0 = time.time()

    # Show RF evolution curve first (doubles as a diagnostic).
    _rf_evolution_curve(preprocessor, X_train, y_train)

    rf_pipeline = _make_pipeline(preprocessor, RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=N_JOBS,
    ))

    rf_param_dist = {
        "model__n_estimators"    : [100, 200, 300, 400, 600, 800, 1000],
        "model__max_depth"       : [None, 10, 20, 30, 50],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features"    : ["sqrt", "log2", 0.2, 0.3, 0.5],
        "model__class_weight"    : ["balanced", "balanced_subsample"],
        "model__bootstrap"       : [True, False],
    }

    search = RandomizedSearchCV(
        rf_pipeline, rf_param_dist,
        n_iter=RF_N_ITER,
        scoring="balanced_accuracy",
        cv=get_cv(),
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        refit=True,
        verbose=2,
    )
    search.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"\n  RF tuning done in {elapsed:.1f}s")
    print(f"  RF best CV BCR: {search.best_score_:.4f}")
    print(f"  RF best params:")
    for k, v in search.best_params_.items():
        print(f"    {k:<35} = {v}")

    try:
        from plots import plot_hyperparameter_search_results
        plot_hyperparameter_search_results(search, "Random Forest (tuned)",
                                           "model__n_estimators",
                                           "model_06_rf_search.png")
    except Exception:
        pass

    return search


def tune_hist_gradient_boosting(preprocessor, X_train, y_train):
    """Tune sklearn HistGradientBoostingClassifier with RandomizedSearchCV (150 iter).

    HistGradBoost is sklearn's native fast gradient boosting (similar to
    LightGBM). It handles large feature sets natively, supports class weights,
    and is significantly faster than full GradientBoostingClassifier.

    Returns
    -------
    search : fitted RandomizedSearchCV object
    """
    print(f"\n  Tuning HistGradientBoosting  ({GB_N_ITER} iterations × {CV_N_SPLITS} folds "
          f"= {GB_N_ITER * CV_N_SPLITS} fits)...")
    t0 = time.time()

    gb_pipeline = _make_pipeline(preprocessor, HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    ))

    gb_param_dist = {
        "model__max_iter"           : [100, 200, 300, 500, 800],
        "model__max_depth"          : [None, 3, 4, 5, 7, 10],
        "model__learning_rate"      : [0.005, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
        "model__min_samples_leaf"   : [5, 10, 20, 50, 100],
        "model__l2_regularization"  : [0.0, 0.01, 0.1, 0.5, 1.0, 5.0],
        "model__max_leaf_nodes"     : [15, 31, 63, 127, 255],
        "model__max_features"       : [0.5, 0.7, 0.9, 1.0],
        "model__early_stopping"     : [True, False],
    }

    search = RandomizedSearchCV(
        gb_pipeline, gb_param_dist,
        n_iter=GB_N_ITER,
        scoring="balanced_accuracy",
        cv=get_cv(),
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        refit=True,
        verbose=2,
    )
    search.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"\n  HistGradBoost tuning done in {elapsed:.1f}s")
    print(f"  HistGradBoost best CV BCR: {search.best_score_:.4f}")
    print(f"  HistGradBoost best params:")
    for k, v in search.best_params_.items():
        print(f"    {k:<35} = {v}")

    try:
        from plots import plot_hyperparameter_search_results
        plot_hyperparameter_search_results(search, "HistGradBoost",
                                           "model__max_iter",
                                           "model_07_gb_search.png")
    except Exception:
        pass

    return search


def tune_gaussian_nb(preprocessor, X_train, y_train):
    """Tune GaussianNB — only one parameter: var_smoothing.

    GaussianNB scored 0.7259 BCR with default settings, beating all tuned
    tree-based models. var_smoothing controls the portion of the largest
    variance added to all variances for numerical stability. Tuning it on
    a log scale can recover another +0.01–0.02 BCR.

    Returns
    -------
    search : fitted RandomizedSearchCV object
    """
    print(f"\n  Tuning GaussianNB  ({GNB_N_ITER} iterations × {CV_N_SPLITS} folds "
          f"= {GNB_N_ITER * CV_N_SPLITS} fits)...")
    t0 = time.time()

    gnb_pipeline = Pipeline([
        ("preprocessor", deepcopy(preprocessor)),
        ("model", GaussianNB()),
    ])

    gnb_param_dist = {
        "model__var_smoothing": np.logspace(-12, 0, 200),
    }

    search = RandomizedSearchCV(
        gnb_pipeline, gnb_param_dist,
        n_iter=GNB_N_ITER,
        scoring="balanced_accuracy",
        cv=get_cv(),
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"\n  GaussianNB tuning done in {elapsed:.1f}s")
    print(f"  GaussianNB best CV BCR: {search.best_score_:.4f}")
    print(f"  GaussianNB best params:")
    for k, v in search.best_params_.items():
        print(f"    {k:<35} = {v:.4e}")

    return search


def tune_lda(preprocessor, X_train, y_train):
    """Tune LinearDiscriminantAnalysis with Ledoit-Wolf shrinkage.

    LDA assumes Gaussian classes with shared covariance (unlike GaussianNB
    which assumes diagonal). With shrinkage='auto' (Ledoit-Wolf estimator)
    it is optimal for high-dimensional data and can outperform SVM on
    linearly separable Gaussian distributions.

    solver='svd' is excluded because it does not support shrinkage.
    All (solver, shrinkage) combinations here are valid.

    Returns
    -------
    search : fitted RandomizedSearchCV object
    """
    print(f"\n  Tuning LDA  ({LDA_N_ITER} iterations × {CV_N_SPLITS} folds "
          f"= {LDA_N_ITER * CV_N_SPLITS} fits)...")
    t0 = time.time()

    lda_pipeline = Pipeline([
        ("preprocessor", deepcopy(preprocessor)),
        ("model", LinearDiscriminantAnalysis()),
    ])

    lda_param_dist = {
        "model__solver"    : ["lsqr", "eigen"],
        "model__shrinkage" : (["auto"] +
                              list(np.linspace(0.0, 1.0, 20))),
        "model__tol"       : [1e-5, 1e-4, 1e-3, 1e-2],
    }

    search = RandomizedSearchCV(
        lda_pipeline, lda_param_dist,
        n_iter=LDA_N_ITER,
        scoring="balanced_accuracy",
        cv=get_cv(),
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"\n  LDA tuning done in {elapsed:.1f}s")
    print(f"  LDA best CV BCR: {search.best_score_:.4f}")
    print(f"  LDA best params:")
    for k, v in search.best_params_.items():
        print(f"    {k:<35} = {v}")

    try:
        from plots import plot_hyperparameter_search_results
        plot_hyperparameter_search_results(search, "LDA", "model__shrinkage",
                                           "model_08_lda_search.png")
    except Exception:
        pass

    return search


def tune_qda(preprocessor, X_train, y_train):
    """Tune QuadraticDiscriminantAnalysis via reg_param.

    QDA models separate covariance matrices per class (unlike LDA which
    assumes shared covariance). With 1024 features and only ~300 positive
    samples, the positive-class covariance matrix is rank-deficient;
    reg_param regularises it: reg_param=0 is pure QDA, reg_param=1 gives
    a diagonal (GaussianNB-like) covariance. Tuning reg_param finds the
    optimal blend.

    Returns
    -------
    search : fitted RandomizedSearchCV object
    """
    print(f"\n  Tuning QDA  ({QDA_N_ITER} iterations × {CV_N_SPLITS} folds "
          f"= {QDA_N_ITER * CV_N_SPLITS} fits)...")
    t0 = time.time()

    qda_pipeline = Pipeline([
        ("preprocessor", deepcopy(preprocessor)),
        ("model", QuadraticDiscriminantAnalysis()),
    ])

    qda_param_dist = {
        "model__reg_param" : list(np.linspace(0.0, 1.0, 100)),
        "model__tol"       : [1e-5, 1e-4, 1e-3],
    }

    search = RandomizedSearchCV(
        qda_pipeline, qda_param_dist,
        n_iter=QDA_N_ITER,
        scoring="balanced_accuracy",
        cv=get_cv(),
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)

    elapsed = time.time() - t0
    print(f"\n  QDA tuning done in {elapsed:.1f}s")
    print(f"  QDA best CV BCR: {search.best_score_:.4f}")
    print(f"  QDA best params:")
    for k, v in search.best_params_.items():
        print(f"    {k:<35} = {v}")

    try:
        from plots import plot_hyperparameter_search_results
        plot_hyperparameter_search_results(search, "QDA", "model__reg_param",
                                           "model_09_qda_search.png")
    except Exception:
        pass

    return search


# ====================================================================
# Ensemble
# ====================================================================

def build_stacking_ensemble(xgb_search, svm_search, mlp_search,
                             X_train, y_train,
                             rf_search=None, gb_search=None,
                             gnb_search=None, lda_search=None, qda_search=None):
    """Combine the best estimators in a StackingClassifier.

    Each base pipeline (preprocessor + model) is already self-contained.
    A LogisticRegression meta-learner is trained on out-of-fold predictions.

    Threshold logic: include all models within 0.05 of the best single model.
    This allows diverse learners (Bayesian, kernel, boosting) to contribute
    even if they individually score slightly lower — diversity > marginal score.

    Returns
    -------
    ensemble        : StackingClassifier (not yet re-fitted on all data)
    ensemble_scores : np.ndarray of per-fold BCR scores
    """
    print("\n  Building stacking ensemble (LogReg meta-learner)...")
    t0 = time.time()

    # Collect candidates with their best CV score.
    all_candidates = []
    if xgb_search  is not None:
        all_candidates.append(("xgb",     xgb_search.best_estimator_,  xgb_search.best_score_))
    if svm_search  is not None:
        all_candidates.append(("svm",     svm_search.best_estimator_,  svm_search.best_score_))
    if mlp_search  is not None:
        all_candidates.append(("mlp",     mlp_search.best_estimator_,  mlp_search.best_score_))
    if rf_search   is not None:
        all_candidates.append(("rf",      rf_search.best_estimator_,   rf_search.best_score_))
    if gb_search   is not None:
        all_candidates.append(("gb",      gb_search.best_estimator_,   gb_search.best_score_))
    if gnb_search  is not None:
        all_candidates.append(("gnb",     gnb_search.best_estimator_,  gnb_search.best_score_))
    if lda_search  is not None:
        all_candidates.append(("lda",     lda_search.best_estimator_,  lda_search.best_score_))
    if qda_search  is not None:
        all_candidates.append(("qda",     qda_search.best_estimator_,  qda_search.best_score_))

    if len(all_candidates) < 2:
        print("  Not enough models for stacking -- returning None.")
        return None, np.array([])

    best_score = max(s for _, _, s in all_candidates)
    # Include all models within 0.05 of the best (was 0.02 — too restrictive,
    # caused only SVM to be selected even though XGBoost and GNB were strong).
    estimators = [(name, est) for name, est, score in all_candidates
                  if score >= best_score - 0.05]
    print(f"  Stacking with {len(estimators)} base learners: "
          f"{[n for n, _ in estimators]}")

    meta_learner = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE,
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
        cv=get_cv(), scoring="balanced_accuracy", n_jobs=1, verbose=1,
    )

    elapsed = time.time() - t0
    print(f"\n  Stacking ensemble done in {elapsed:.1f}s")
    print(f"  Stacking Ensemble BCR: {ensemble_scores.mean():.4f} ± {ensemble_scores.std():.4f}")
    return ensemble, ensemble_scores


# ====================================================================
# Model selection
# ====================================================================

def select_best_model(baseline_results, xgb_search, svm_search,
                      mlp_search, ensemble, ensemble_scores,
                      rf_search=None, gb_search=None,
                      gnb_search=None, lda_search=None, qda_search=None):
    """Compare all candidates and return the best pipeline.

    Selection rules:
    - If the ensemble BCR is more than 0.005 better than the best single
      model, prefer the ensemble.
    - Otherwise, prefer the single model with the highest mean BCR.
    - If two models are within 0.005 of each other, prefer the one with
      lower standard deviation (more stable).

    Returns
    -------
    best_pipeline : chosen pipeline
    best_name     : string name
    candidates    : list of dicts for plotting
    """
    print("\n" + "=" * 65)
    print("  MODEL COMPARISON TABLE")
    print(f"  {'Model':<35} {'CV BCR Mean':>12} {'CV BCR Std':>11}")
    print("  " + "-" * 60)

    candidates = []

    def _add(name, search_or_scores, is_ensemble=False):
        if search_or_scores is None:
            return
        if is_ensemble:
            mean     = search_or_scores.mean()
            std      = search_or_scores.std()
            pipeline = ensemble
        else:
            mean     = search_or_scores.best_score_
            std      = search_or_scores.cv_results_["std_test_score"][
                           search_or_scores.best_index_]
            pipeline = search_or_scores.best_estimator_
        candidates.append({"name": name, "mean": mean, "std": std, "pipeline": pipeline})
        marker = " ←" if mean == max(
            (c["mean"] for c in candidates), default=0
        ) else ""
        print(f"  {name:<35} {mean:>12.4f} {std:>11.4f}{marker}")

    # Baselines (no pipeline object kept — they were reference only).
    for key, label in [("lr", "Logistic Regression"), ("rf", "Random Forest (baseline)"),
                        ("gnb", "GaussianNB (Bayesian)"), ("et", "Extra Trees (baseline)")]:
        r = baseline_results.get(key)
        if r is None:
            continue
        candidates.append({"name": label, "mean": r["mean"], "std": r["std"], "pipeline": None})
        print(f"  {label:<35} {r['mean']:>12.4f} {r['std']:>11.4f}")

    _add("XGBoost (tuned)",          xgb_search)
    _add("SVM RBF (tuned)",          svm_search)
    _add("MLP+PCA (tuned)",          mlp_search)
    _add("Random Forest (tuned)",    rf_search)
    _add("HistGradBoost (tuned)",    gb_search)
    _add("GaussianNB (tuned)",       gnb_search)
    _add("LDA (tuned)",              lda_search)
    _add("QDA (tuned)",              qda_search)

    if ensemble is not None and len(ensemble_scores) > 0:
        _add("Stacking Ensemble", ensemble_scores, is_ensemble=True)

    print("  " + "-" * 60)

    try:
        from plots import plot_model_comparison
        plot_model_comparison([c for c in candidates if c["mean"] > 0])
    except Exception:
        pass

    runnable = [c for c in candidates if c["pipeline"] is not None]
    if not runnable:
        raise RuntimeError("No tuned models available. Check Phase 4 execution.")

    best_single        = max(runnable, key=lambda c: c["mean"])
    ensemble_candidate = next((c for c in runnable if c["name"] == "Stacking Ensemble"), None)

    if (ensemble_candidate is not None and
            ensemble_candidate["mean"] > best_single["mean"] + 0.005):
        chosen = ensemble_candidate
    else:
        threshold        = best_single["mean"] - 0.005
        close_candidates = [c for c in runnable if c["mean"] >= threshold]
        chosen           = min(close_candidates, key=lambda c: c["std"])

    print(f"\n  Chosen model: {chosen['name']} "
          f"(BCR={chosen['mean']:.4f}, std={chosen['std']:.4f})")

    return chosen["pipeline"], chosen["name"], candidates


# ====================================================================
# Robustness
# ====================================================================

def evaluate_robustness(pipeline, X_train, y_train):
    """Estimate model stability by running CV with multiple random seeds.

    A small robustness_score (std across seeds) means the estimate is
    stable and trustworthy; a large value signals high sensitivity to
    the choice of folds.

    Returns
    -------
    mean_across_seeds  : float
    robustness_score   : float, std of mean BCR across seeds (lower = better)
    seed_results       : list of dicts {seed, mean, std}
    """
    print(f"\n  Multi-seed robustness ({len(ROBUSTNESS_SEEDS)} seeds):")
    t0 = time.time()

    seed_results = []
    for seed in ROBUSTNESS_SEEDS:
        cv     = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=seed)
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="balanced_accuracy", n_jobs=1,
        )
        seed_results.append({"seed": seed, "mean": float(scores.mean()),
                              "std":  float(scores.std())})
        print(f"    seed={seed:<6}  BCR={scores.mean():.4f} ± {scores.std():.4f}")

    means             = np.array([r["mean"] for r in seed_results])
    mean_across_seeds = float(means.mean())
    robustness_score  = float(means.std())

    label = ("stable"   if robustness_score < 0.01 else
             "moderate" if robustness_score < 0.02 else "unstable")
    print(f"\n  Mean BCR across seeds  : {mean_across_seeds:.4f}")
    print(f"  Robustness score (std) : {robustness_score:.4f}  ({label})")
    print(f"  Multi-seed eval done in {time.time() - t0:.1f}s")

    return mean_across_seeds, robustness_score, seed_results
