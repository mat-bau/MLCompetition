# plots.py
# Centralized plotting utilities for the A5 Toxicity Classification pipeline.
# All plots are saved to PLOTS_DIR and displayed if a display is available.
# Uses matplotlib + seaborn.

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe on all machines)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from config import PLOTS_DIR, RANDOM_STATE

# Apply a clean, consistent style across all plots.
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titleweight": "bold",
})


def _ensure_plots_dir():
    """Create the plots directory if it does not exist."""
    os.makedirs(PLOTS_DIR, exist_ok=True)


def _savefig(fig, filename):
    """Save a figure to PLOTS_DIR and close it."""
    _ensure_plots_dir()
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [plot] saved → {path}")


# ====================================================================
# EDA PLOTS
# ====================================================================

def plot_class_distribution(labels_series):
    """Bar chart of class counts and percentages.

    Parameters
    ----------
    labels_series : pd.Series of string labels ('positive' / 'negative')
    """
    counts = labels_series.value_counts().sort_index()
    total  = len(labels_series)
    pcts   = 100 * counts / total

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=["#e74c3c", "#3498db"], edgecolor="white", linewidth=0.8)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=11)

    ax.set_title("Class Distribution (Train Labels)")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.15)
    _savefig(fig, "eda_01_class_distribution.png")


def plot_feature_variance_histogram(train_df, zero_var_cols):
    """Histogram of per-feature variances (excluding zero-variance features).

    Parameters
    ----------
    train_df      : pd.DataFrame of train features
    zero_var_cols : list of column names with zero variance
    """
    cols_to_use = [c for c in train_df.columns if c not in zero_var_cols]
    variances   = train_df[cols_to_use].var(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Full distribution
    axes[0].hist(variances, bins=60, color="#2ecc71", edgecolor="white", linewidth=0.5)
    axes[0].set_title("Feature Variance Distribution (all)")
    axes[0].set_xlabel("Variance")
    axes[0].set_ylabel("Number of Features")

    # Zoomed in: bottom 95th percentile
    cutoff = np.percentile(variances, 95)
    axes[1].hist(variances[variances <= cutoff], bins=60,
                 color="#3498db", edgecolor="white", linewidth=0.5)
    axes[1].set_title("Feature Variance Distribution (≤ 95th pct)")
    axes[1].set_xlabel("Variance")
    axes[1].set_ylabel("Number of Features")

    fig.suptitle(f"Feature Variances  |  {len(zero_var_cols)} zero-variance features excluded",
                 fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "eda_02_feature_variances.png")


def plot_correlation_heatmap(train_df, zero_var_cols, n_top=40):
    """Heatmap of correlations among the n_top highest-variance features.

    Parameters
    ----------
    train_df      : pd.DataFrame of train features
    zero_var_cols : list of zero-variance column names to exclude
    n_top         : number of top-variance features to include in the heatmap
    """
    cols_to_use = [c for c in train_df.columns if c not in zero_var_cols]
    variances   = train_df[cols_to_use].var(axis=0).sort_values(ascending=False)
    top_cols    = variances.head(n_top).index.tolist()

    corr = train_df[top_cols].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # upper triangle mask
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.3, linecolor="white",
        annot=False, square=True,
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
    )
    ax.set_title(f"Correlation Heatmap — Top {n_top} Highest-Variance Features",
                 pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=7)
    plt.tight_layout()
    _savefig(fig, "eda_03_correlation_heatmap.png")


def plot_train_test_distribution_shift(train_df, test_df, n_top=30):
    """Scatter of train mean vs test mean for the top shifted features.

    A perfect train=test line is drawn for reference.

    Parameters
    ----------
    train_df : pd.DataFrame of train features
    test_df  : pd.DataFrame of test features
    n_top    : number of features to label (most shifted ones)
    """
    train_means = train_df.mean(axis=0)
    test_means  = test_df.mean(axis=0)
    train_stds  = train_df.std(axis=0).replace(0, np.nan)
    shift       = ((test_means - train_means).abs() / train_stds).fillna(0)

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(train_means, test_means, c=shift, cmap="YlOrRd",
                    s=8, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Shift (|Δmean| / train std)")

    # Identity line
    lo = min(train_means.min(), test_means.min())
    hi = max(train_means.max(), test_means.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="train = test")
    ax.legend(fontsize=9)

    ax.set_title("Train vs Test Feature Means (Distribution Shift)")
    ax.set_xlabel("Train Mean")
    ax.set_ylabel("Test Mean")
    plt.tight_layout()
    _savefig(fig, "eda_04_distribution_shift.png")


def plot_outlier_heatmap(train_df, zero_var_cols, n_top=50):
    """Bar chart of the top features by outlier count.

    Parameters
    ----------
    train_df      : pd.DataFrame of train features
    zero_var_cols : list of zero-variance column names to exclude
    n_top         : number of top features to show
    """
    from config import OUTLIER_STD_THRESHOLD

    cols_to_use  = [c for c in train_df.columns if c not in zero_var_cols]
    sub          = train_df[cols_to_use]
    col_means    = sub.mean(axis=0)
    col_stds     = sub.std(axis=0).replace(0, np.nan)
    z_scores     = (sub - col_means) / col_stds
    outlier_cnts = (z_scores.abs() > OUTLIER_STD_THRESHOLD).sum(axis=0)
    top_outliers = outlier_cnts.sort_values(ascending=False).head(n_top)

    if top_outliers.sum() == 0:
        print("  [plot] No outliers found — skipping outlier bar chart.")
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(top_outliers)), top_outliers.values,
           color="#e67e22", edgecolor="white", linewidth=0.5)
    ax.set_title(f"Top {n_top} Features by Outlier Count  (|z| > {OUTLIER_STD_THRESHOLD})")
    ax.set_xlabel("Feature rank (most outliers → left)")
    ax.set_ylabel("Number of outlier samples")
    ax.xaxis.set_major_formatter(mticker.NullFormatter())
    plt.tight_layout()
    _savefig(fig, "eda_05_outlier_features.png")


def plot_feature_means_by_class(train_df, y_train, n_top=30):
    """Horizontal bar chart: features where class means differ the most.

    Parameters
    ----------
    train_df : pd.DataFrame of raw train features
    y_train  : np.ndarray of int labels (0 / 1)
    n_top    : number of features to show
    """
    df = train_df.copy()
    df["__label__"] = y_train
    means0 = df[df["__label__"] == 0].drop(columns="__label__").mean()
    means1 = df[df["__label__"] == 1].drop(columns="__label__").mean()
    stds   = df.drop(columns="__label__").std().replace(0, np.nan)
    diff   = ((means1 - means0).abs() / stds).fillna(0)
    top    = diff.sort_values(ascending=False).head(n_top)

    fig, ax = plt.subplots(figsize=(8, max(4, n_top * 0.25)))
    ax.barh(range(len(top)), top.values[::-1], color="#9b59b6", edgecolor="white")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1].astype(str), fontsize=8)
    ax.set_title(f"Top {n_top} Features by Class Mean Difference  (|Δμ| / σ)")
    ax.set_xlabel("Standardised Mean Difference")
    plt.tight_layout()
    _savefig(fig, "eda_06_top_discriminative_features.png")


# ====================================================================
# MODEL / EVALUATION PLOTS
# ====================================================================

def plot_model_comparison(all_candidates):
    """Horizontal bar chart comparing all candidate model CV BCR scores.

    Parameters
    ----------
    all_candidates : list of dicts with keys 'name', 'mean', 'std'
    """
    names  = [c["name"] for c in all_candidates]
    means  = [c["mean"] for c in all_candidates]
    stds   = [c["std"]  for c in all_candidates]

    sorted_idx = np.argsort(means)
    names  = [names[i] for i in sorted_idx]
    means  = [means[i] for i in sorted_idx]
    stds   = [stds[i]  for i in sorted_idx]

    colors = ["#e74c3c" if m == max(means) else "#3498db" for m in means]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.55)))
    bars = ax.barh(names, means, xerr=stds, color=colors, edgecolor="white",
                   linewidth=0.8, capsize=4, error_kw={"linewidth": 1.2})
    for bar, m in zip(bars, means):
        ax.text(m + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{m:.4f}", va="center", fontsize=9)

    ax.set_xlim(max(0, min(means) - 0.05), min(1, max(means) + 0.06))
    ax.set_title("Model Comparison — CV Balanced Accuracy (BCR)")
    ax.set_xlabel("Balanced Accuracy (mean ± std over folds)")
    ax.axvline(max(means), color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    _savefig(fig, "model_01_comparison.png")


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", filename="eval_01_confusion_matrix.png"):
    """Annotated confusion matrix heatmap.

    Parameters
    ----------
    y_true    : array-like of true labels (int)
    y_pred    : array-like of predicted labels (int)
    title     : plot title
    filename  : output filename
    """
    from sklearn.metrics import confusion_matrix

    cm     = confusion_matrix(y_true, y_pred)
    labels = ["negative (0)", "positive (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Count"},
        ax=ax,
    )
    ax.set_title(title, pad=12)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    _savefig(fig, filename)


def plot_fold_bcr_scores(fold_bcr_scores, bcr_hat):
    """Bar chart of per-fold BCR scores with the OOF BCR as a reference line.

    Parameters
    ----------
    fold_bcr_scores : np.ndarray of per-fold BCR values
    bcr_hat         : float, overall OOF BCR
    """
    n_folds = len(fold_bcr_scores)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, n_folds + 1), fold_bcr_scores,
           color="#3498db", edgecolor="white", linewidth=0.8)
    ax.axhline(bcr_hat, color="red", linestyle="--", linewidth=1.2,
               label=f"OOF BCRhat = {bcr_hat:.4f}")
    ax.axhline(fold_bcr_scores.mean(), color="#e67e22", linestyle=":",
               linewidth=1.0, label=f"Fold mean = {fold_bcr_scores.mean():.4f}")
    for i, v in enumerate(fold_bcr_scores):
        ax.text(i + 1, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_title("Per-Fold Balanced Accuracy (BCR)")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_xticks(range(1, n_folds + 1))
    ax.set_ylim(max(0, fold_bcr_scores.min() - 0.05), min(1.0, fold_bcr_scores.max() + 0.04))
    ax.legend(fontsize=9)
    plt.tight_layout()
    _savefig(fig, "eval_02_fold_bcr_scores.png")


def plot_xgb_feature_importance(xgb_search, top_n=30):
    """Bar chart of XGBoost feature importances (gain).

    Parameters
    ----------
    xgb_search : fitted RandomizedSearchCV with XGBoost best_estimator_
    top_n      : number of top features to show
    """
    try:
        booster = xgb_search.best_estimator_.named_steps["model"].get_booster()
        importance = booster.get_score(importance_type="gain")
    except Exception:
        print("  [plot] Could not extract XGBoost feature importances — skipping.")
        return

    if not importance:
        return

    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, gains = zip(*sorted_imp)

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.28)))
    ax.barh(range(len(features)), list(gains)[::-1],
            color="#f39c12", edgecolor="white")
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(list(features)[::-1], fontsize=8)
    ax.set_title(f"XGBoost Feature Importance (Gain) — Top {top_n}")
    ax.set_xlabel("Gain")
    plt.tight_layout()
    _savefig(fig, "model_02_xgb_feature_importance.png")


def plot_feature_selection_results(fc_results, mi_results, mw_results,
                                    pca_results, rfe_results,
                                    l1lr_results, l1svc_results,
                                    baseline_bcr):
    """Three-panel comparison of all seven feature selection strategies.

    Panel 1 — Filter methods (f_classif, mutual_info, Mann-Whitney) BCR vs k.
    Panel 2 — Wrapper (RFE) and PCA BCR vs k.
    Panel 3 — Embedded L1 (LR + SVC) BCR vs C (number of features annotated).
    """
    _ensure_plots_dir()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ---- Panel 1: Filter methods ----
    ax = axes[0]
    palette = {"f_classif": "#3498db", "mutual_info": "#e74c3c",
               "Mann-Whitney": "#2ecc71"}
    for res, color in [(fc_results, palette["f_classif"]),
                       (mi_results, palette["mutual_info"]),
                       (mw_results, palette["Mann-Whitney"])]:
        if not res:
            continue
        lbl   = res[0]["label"]
        ks    = [r["k"] if r["k"] != "all" else 1024 for r in res]
        means = [r["mean"] for r in res]
        stds  = [r["std"]  for r in res]
        ax.errorbar(ks, means, yerr=stds, marker="o", label=lbl,
                    color=color, linewidth=1.8, capsize=3)
    ax.axhline(baseline_bcr, linestyle="--", color="gray", linewidth=1.0,
               label=f"Baseline={baseline_bcr:.4f}")
    ax.set_title("Filter Methods (SelectKBest)")
    ax.set_xlabel("k features selected")
    ax.set_ylabel("CV BCR (LR reference)")
    ax.legend(fontsize=8)
    if fc_results:
        ticks = [r["k"] if r["k"] != "all" else 1024 for r in fc_results]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(r["k"]) for r in fc_results], rotation=30)

    # ---- Panel 2: Wrapper (RFE) + PCA ----
    ax = axes[1]
    if rfe_results:
        ks    = [r["k"]    for r in rfe_results]
        means = [r["mean"] for r in rfe_results]
        stds  = [r["std"]  for r in rfe_results]
        ax.errorbar(ks, means, yerr=stds, marker="^", color="#e67e22",
                    linewidth=1.8, capsize=3, label="RFE (LR coef)")
    if pca_results:
        ks    = [r["k"]    for r in pca_results]
        means = [r["mean"] for r in pca_results]
        stds  = [r["std"]  for r in pca_results]
        ax.errorbar(ks, means, yerr=stds, marker="s", color="#9b59b6",
                    linewidth=1.8, capsize=3, label="PCA")
    ax.axhline(baseline_bcr, linestyle="--", color="gray", linewidth=1.0,
               label=f"Baseline={baseline_bcr:.4f}")
    ax.set_title("Wrapper (RFE) & Dimensionality Reduction (PCA)")
    ax.set_xlabel("k features / components")
    ax.set_ylabel("CV BCR (LR reference)")
    ax.legend(fontsize=8)

    # ---- Panel 3: Embedded methods ----
    ax = axes[2]
    for res, color, marker in [(l1lr_results,  "#1abc9c", "o"),
                                (l1svc_results, "#e74c3c", "s")]:
        if not res:
            continue
        lbl   = res[0]["label"]
        cs    = [r["C"]    for r in res]
        means = [r["mean"] for r in res]
        stds  = [r["std"]  for r in res]
        n_feats = [r["k"]  for r in res]
        ax.errorbar(range(len(cs)), means, yerr=stds, marker=marker,
                    color=color, linewidth=1.8, capsize=3, label=lbl)
        for i, (m, n) in enumerate(zip(means, n_feats)):
            ax.annotate(f"n={n}", (i, m), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7, color=color)
    ax.axhline(baseline_bcr, linestyle="--", color="gray", linewidth=1.0,
               label=f"Baseline={baseline_bcr:.4f}")
    ax.set_title("Embedded Methods (SelectFromModel L1)")
    if l1lr_results:
        ax.set_xticks(range(len(l1lr_results)))
        ax.set_xticklabels([f"C={r['C']}" for r in l1lr_results], rotation=30)
    ax.set_xlabel("L1 regularisation strength C")
    ax.set_ylabel("CV BCR (LR reference)")
    ax.legend(fontsize=8)

    fig.suptitle("Feature Selection Strategy Comparison (all 7 methods)",
                 fontweight="bold")
    plt.tight_layout()
    _savefig(fig, "feat_01_selection_comparison.png")


def plot_hyperparameter_search_results(search, model_name, param_name, filename):
    """Strip / scatter plot showing CV BCR across all RandomizedSearchCV trials.

    Parameters
    ----------
    search     : fitted RandomizedSearchCV object
    model_name : display name for the title
    param_name : which hyperparameter to show on x-axis (best-effort)
    filename   : output filename
    """
    results = search.cv_results_
    means   = results["mean_test_score"]
    stds    = results["std_test_score"]
    ranks   = results["rank_test_score"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(range(len(means)), means, c=ranks, cmap="RdYlGn_r",
               s=30, alpha=0.7)
    ax.fill_between(range(len(means)),
                    means - stds, means + stds,
                    alpha=0.15, color="#3498db")
    best_idx = search.best_index_
    ax.scatter(best_idx, means[best_idx], marker="*", s=180,
               color="red", zorder=5, label=f"Best = {means[best_idx]:.4f}")
    ax.set_title(f"{model_name} — RandomizedSearch CV BCR per Trial")
    ax.set_xlabel("Trial index")
    ax.set_ylabel("CV Balanced Accuracy")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _savefig(fig, filename)


def plot_learning_curve(pipeline, X_train, y_train, filename="eval_03_learning_curve.png"):
    """Learning curve: training size vs train/CV BCR.

    Useful for diagnosing overfitting (large gap between train and CV) or
    underfitting (both scores plateau at a low value).

    Parameters
    ----------
    pipeline : sklearn-compatible Pipeline (unfitted)
    X_train  : np.ndarray
    y_train  : np.ndarray of int
    filename : output filename
    """
    from sklearn.model_selection import learning_curve

    train_sizes = np.linspace(0.10, 1.0, 8)

    train_sz, train_scores, cv_scores = learning_curve(
        pipeline, X_train, y_train,
        train_sizes=train_sizes,
        cv=5,
        scoring="balanced_accuracy",
        n_jobs=1,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    cv_mean    = cv_scores.mean(axis=1)
    cv_std     = cv_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_sz, train_mean, "o-", color="#2ecc71", label="Train BCR")
    ax.fill_between(train_sz, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#2ecc71")
    ax.plot(train_sz, cv_mean, "s-", color="#e74c3c", label="CV BCR (5-fold)")
    ax.fill_between(train_sz, cv_mean - cv_std, cv_mean + cv_std,
                    alpha=0.15, color="#e74c3c")

    ax.set_title("Learning Curve — BCR vs Training Set Size")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("Balanced Accuracy")
    ax.legend(fontsize=9)
    ax.set_ylim(max(0, min(cv_mean.min(), train_mean.min()) - 0.05),
                min(1.0, max(cv_mean.max(), train_mean.max()) + 0.04))
    plt.tight_layout()
    _savefig(fig, filename)


def plot_cv_scatter(search, model_name, param_x, param_y, filename):
    """2-D scatter of RandomizedSearch trials: param_x vs param_y, colour = BCR.

    Unlike a heatmap (which requires a uniform grid), this works correctly
    with the non-uniform sampling of RandomizedSearchCV.

    Parameters
    ----------
    search    : fitted RandomizedSearchCV
    model_name: display name for title
    param_x   : key in search.cv_results_["params"], e.g. "model__C"
    param_y   : key in search.cv_results_["params"], e.g. "model__gamma"
    filename  : output filename
    """
    params = search.cv_results_["params"]
    means  = search.cv_results_["mean_test_score"]
    stds   = search.cv_results_["std_test_score"]

    x_vals = []
    y_vals = []
    for p in params:
        xv = p.get(param_x)
        yv = p.get(param_y)
        x_vals.append(float(xv) if xv is not None else np.nan)
        y_vals.append(float(yv) if yv is not None else np.nan)

    x_arr = np.array(x_vals)
    y_arr = np.array(y_vals)

    # Mask trials where both params exist.
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    if mask.sum() < 3:
        print(f"  [plot] {filename}: fewer than 3 trials with both {param_x} "
              f"and {param_y} — skipping scatter.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        x_arr[mask], y_arr[mask],
        c=means[mask], cmap="RdYlGn",
        s=60 / (stds[mask] + 0.01),   # larger = more stable
        alpha=0.75, edgecolors="white", linewidths=0.4,
    )
    plt.colorbar(sc, ax=ax, label="Mean CV BCR")

    best_idx = search.best_index_
    if mask[best_idx]:
        ax.scatter(x_arr[best_idx], y_arr[best_idx],
                   marker="*", s=250, color="red", zorder=5,
                   label=f"Best = {means[best_idx]:.4f}")
        ax.legend(fontsize=9)

    x_label = param_x.replace("model__", "")
    y_label = param_y.replace("model__", "")
    ax.set_xscale("log") if x_arr[mask].min() > 0 and (x_arr[mask].max() / x_arr[mask].min() > 100) else None
    ax.set_yscale("log") if y_arr[mask].min() > 0 and (y_arr[mask].max() / y_arr[mask].min() > 100) else None
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{model_name} — RandomizedSearch: {x_label} vs {y_label}")
    plt.tight_layout()
    _savefig(fig, filename)


def plot_rf_evolution(steps, bcr_means, filename="model_06_rf_evolution.png"):
    """Line chart of BCR vs n_estimators to show how the RF improves as
    more trees are added.

    Parameters
    ----------
    steps     : list of int  — n_estimators values evaluated
    bcr_means : list of float — CV BCR at each step
    filename  : output filename
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, bcr_means, "o-", color="#2ecc71", linewidth=2, markersize=7)
    for x, y in zip(steps, bcr_means):
        ax.text(x, y + 0.001, f"{y:.4f}", ha="center", va="bottom", fontsize=8)

    ax.set_title("Random Forest — BCR vs Number of Trees")
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("CV Balanced Accuracy (BCR)")
    ax.set_xticks(steps)
    ax.set_xticklabels(steps, rotation=30)
    delta = max(bcr_means) - min(bcr_means)
    ax.set_ylim(min(bcr_means) - max(delta * 0.5, 0.01),
                max(bcr_means) + max(delta * 0.5, 0.01))
    plt.tight_layout()
    _savefig(fig, filename)


def plot_p_score_analysis(bcr_hat, predicted_bcr, sigma,
                          filename="eval_04_p_score_analysis.png"):
    """Continuous P-score curve as a function of the submitted BCRhat.

    Shows how many competition points are gained/lost depending on what
    BCRhat value is submitted, with vertical lines marking key scenarios.

    P = BCR - |BCR - BCRhat| * (1 - exp(-|BCR - BCRhat| / σ))

    Parameters
    ----------
    bcr_hat       : float — our OOF BCR (proxy for true test BCR)
    predicted_bcr : float — shrinkage-corrected value we'd submit
    sigma         : float — uncertainty used for CI bounds
    filename      : output filename
    """
    submitted_range = np.linspace(
        max(0.0, bcr_hat - 4 * sigma),
        min(1.0, bcr_hat + 4 * sigma),
        300
    )
    sigma_p = max(sigma, 1e-10)
    delta   = np.abs(bcr_hat - submitted_range)
    p_curve = bcr_hat - delta * (1.0 - np.exp(-delta / sigma_p))

    ci_low  = bcr_hat - 1.96 * sigma
    ci_high = bcr_hat + 1.96 * sigma

    def p_at(val):
        d = abs(bcr_hat - val)
        return bcr_hat - d * (1.0 - np.exp(-d / sigma_p))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(submitted_range, p_curve, color="#3498db", linewidth=2, label="P(BCRhat)")
    ax.fill_between(submitted_range, p_curve, min(p_curve), alpha=0.08, color="#3498db")

    # Mark key scenarios.
    for val, label, color, ls in [
        (bcr_hat,       f"BCR_real proxy\n({bcr_hat:.4f})",        "#2ecc71", "--"),
        (predicted_bcr, f"Our estimate\n({predicted_bcr:.4f}) ← SUBMIT", "#e74c3c", "-"),
        (ci_low,        f"CI low\n({ci_low:.4f})",                  "#e67e22", ":"),
        (ci_high,       f"CI high\n({ci_high:.4f})",                "#9b59b6", ":"),
    ]:
        if 0 <= val <= 1:
            ax.axvline(val, color=color, linestyle=ls, linewidth=1.3, alpha=0.85)
            ax.scatter([val], [p_at(val)], color=color, s=60, zorder=5)
            ax.annotate(f"P={p_at(val):.4f}\n{label}", xy=(val, p_at(val)),
                        xytext=(val + 0.002 * (1 if val < bcr_hat else -1),
                                p_at(val) - 0.008),
                        fontsize=7.5, color=color,
                        ha="left" if val < bcr_hat else "right")

    ax.set_title("Competition P Score vs Submitted BCRhat\n"
                 f"(assumes true test BCR ≈ {bcr_hat:.4f})")
    ax.set_xlabel("Submitted BCRhat (what you write in Q2 on Inginious)")
    ax.set_ylabel("P score")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    _savefig(fig, filename)

