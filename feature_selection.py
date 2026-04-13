# feature_selection.py
# Phase 5 -- Feature Engineering and Dimensionality Reduction.
# Evaluates SelectKBest (f_classif, mutual_info_classif) and PCA
# by embedding them in pipelines and comparing CV BCR.

import numpy as np
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score

from models import get_cv
from config import SELECT_K_VALUES, PCA_N_VALUES, RANDOM_STATE


def evaluate_select_k_best(preprocessor, best_model_pipeline,
                            X_train, y_train,
                            score_func=f_classif):
    """Evaluate SelectKBest for a range of k values.

    Parameters
    ----------
    preprocessor          : sklearn Pipeline (preprocessing steps only)
    best_model_pipeline   : the full best pipeline from Phase 4
    X_train, y_train      : training data
    score_func            : f_classif or mutual_info_classif

    Returns
    -------
    results : list of dicts with keys 'k', 'mean', 'std'
    """
    func_name = score_func.__name__
    print(f"\nSelectKBest ({func_name}):")

    # Extract just the final estimator (model step) from the best pipeline.
    # The pipeline already has a preprocessor as its first step, so we need
    # to pull out only the model to rebuild pipelines with a selector in between.
    model_step = best_model_pipeline.steps[-1]  # ('model', estimator)

    results = []
    for k in SELECT_K_VALUES:
        selector = SelectKBest(score_func=score_func, k=k)
        pipeline = Pipeline([
            ("preprocessor", deepcopy(preprocessor)),
            ("selector",     selector),
            (model_step[0],  deepcopy(model_step[1])),
        ])

        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=get_cv(), scoring="balanced_accuracy",
        )
        results.append({"k": k, "mean": scores.mean(), "std": scores.std()})
        print(f"  k={str(k):<6}  BCR={scores.mean():.4f} +/- {scores.std():.4f}")

    return results


def evaluate_pca(preprocessor, best_model_pipeline, X_train, y_train):
    """Evaluate PCA for a range of n_components values.

    Parameters
    ----------
    preprocessor          : sklearn Pipeline (preprocessing steps only)
    best_model_pipeline   : the full best pipeline from Phase 4
    X_train, y_train      : training data

    Returns
    -------
    results : list of dicts with keys 'n', 'mean', 'std'
    """
    print("\nPCA dimensionality reduction:")

    model_step = best_model_pipeline.steps[-1]

    results = []
    for n in PCA_N_VALUES:
        pca = PCA(n_components=n, random_state=RANDOM_STATE)
        pipeline = Pipeline([
            ("preprocessor", deepcopy(preprocessor)),
            ("pca",          pca),
            (model_step[0],  deepcopy(model_step[1])),
        ])

        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=get_cv(), scoring="balanced_accuracy",
        )
        results.append({"n": n, "mean": scores.mean(), "std": scores.std()})
        print(f"  n={n:<4}  BCR={scores.mean():.4f} +/- {scores.std():.4f}")

    return results


def run_feature_selection(preprocessor, best_model_pipeline, X_train, y_train,
                          baseline_bcr):
    """Run all feature selection strategies and decide which (if any) to use.

    A feature selection step is adopted only if it improves CV BCR by more
    than 0.002 over using all features (no selection).

    Parameters
    ----------
    preprocessor          : sklearn Pipeline (preprocessing steps only)
    best_model_pipeline   : the full best pipeline from Phase 4
    X_train, y_train      : training data
    baseline_bcr          : float, CV BCR of the best model with all features

    Returns
    -------
    best_pipeline         : the pipeline to use going forward (may be unchanged)
    selection_description : string describing the chosen strategy
    """

    print(f"Baseline BCR (all features): {baseline_bcr:.4f}")
    print("Improvement threshold to adopt selection: 0.002")

    # Evaluate f_classif SelectKBest.
    fclassif_results = evaluate_select_k_best(
        preprocessor, best_model_pipeline, X_train, y_train, f_classif
    )

    # Evaluate mutual information SelectKBest.
    mi_results = evaluate_select_k_best(
        preprocessor, best_model_pipeline, X_train, y_train, mutual_info_classif
    )

    # Evaluate PCA.
    pca_results = evaluate_pca(preprocessor, best_model_pipeline, X_train, y_train)

    # Collect the single best result from each strategy.
    best_fclassif = max(fclassif_results, key=lambda r: r["mean"])
    best_mi       = max(mi_results,       key=lambda r: r["mean"])
    best_pca      = max(pca_results,      key=lambda r: r["mean"])

    all_best = [
        ("SelectKBest f_classif", best_fclassif["mean"], best_fclassif["std"],
         best_fclassif["k"], "f_classif"),
        ("SelectKBest mutual_info", best_mi["mean"], best_mi["std"],
         best_mi["k"], "mutual_info"),
        ("PCA", best_pca["mean"], best_pca["std"], best_pca["n"], "pca"),
    ]

    print("\nBest result per strategy vs. baseline:")
    print(f"{'Strategy':<30} {'BCR Mean':>10} {'BCR Std':>9} {'Improvement':>12}")
    for name, mean, std, param, _ in all_best:
        improvement = mean - baseline_bcr
        print(f"  {name:<28} {mean:>10.4f} {std:>9.4f} {improvement:>+12.4f}  (param={param})")

    # Adopt the best strategy only if it beats the baseline by more than 0.002.
    improvement_threshold = 0.002
    best_of_all = max(all_best, key=lambda r: r[1])
    best_improvement = best_of_all[1] - baseline_bcr

    if best_improvement > improvement_threshold:
        strategy_name, best_mean, best_std, best_param, strategy_key = best_of_all
        print(f"\nAdopting: {strategy_name} with param={best_param} "
              f"(improvement={best_improvement:+.4f})")

        model_step = best_model_pipeline.steps[-1]

        if strategy_key == "pca":
            selector = PCA(n_components=best_param, random_state=RANDOM_STATE)
            selector_name = "pca"
        elif strategy_key == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_classif, k=best_param)
            selector_name = "selector"
        else:  # f_classif
            selector = SelectKBest(score_func=f_classif, k=best_param)
            selector_name = "selector"

        final_pipeline = Pipeline([
            ("preprocessor", deepcopy(preprocessor)),
            (selector_name,  selector),
            (model_step[0],  deepcopy(model_step[1])),
        ])
        selection_description = f"{strategy_name} k/n={best_param}"

    else:
        print(f"\nNo strategy improves BCR by more than {improvement_threshold}.")
        print("Keeping all features -- using the best Phase 4 pipeline unchanged.")
        final_pipeline = best_model_pipeline
        selection_description = "No feature selection (all features used)"

    return final_pipeline, selection_description