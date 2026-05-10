import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict  # noqa: F401
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

TRAIN_FEATURES_PATH = "./A5_2026_train.csv"
TRAIN_LABELS_PATH   = "./A5_2026_train_labels.csv"
TEST_FEATURES_PATH  = "./A5_2026_test.csv"
PREDICTIONS_PATH    = "./predictions.csv"

LABEL_MAP         = {"positive": 1, "negative": 0}
REVERSE_LABEL_MAP = {1: "positive", 0: "negative"}
RANDOM_STATE      = 42
N_JOBS            = -1


# --- 1. Chargement des donnees -------------------------------------------

train_df  = pd.read_csv(TRAIN_FEATURES_PATH)
labels_df = pd.read_csv(TRAIN_LABELS_PATH, index_col=0)
labels_df.columns = ["label"]
test_df   = pd.read_csv(TEST_FEATURES_PATH)

if not train_df.index.equals(labels_df.index):
    labels_df = labels_df.reset_index(drop=True)

y_train = labels_df["label"].map(LABEL_MAP).values.astype(int)
X_train = train_df.values.astype(np.float64)
X_test  = test_df.values.astype(np.float64)

#print(f"X_train : {X_train.shape}   X_test : {X_test.shape}")
#print(f"Classes : {np.bincount(y_train)}  (0=negatif, 1=positif)")


# --- 2. Preprocessing ----------------------------------------------------

def make_pipeline(model, pca_n=None):
    steps = [
        ("variance_filter", VarianceThreshold(threshold=0.0)),
        ("imputer",         SimpleImputer(strategy="median")),
        ("scaler",          RobustScaler()),
    ]
    if pca_n is not None:
        steps.append(("pca", PCA(n_components=pca_n, random_state=RANDOM_STATE)))
    if SMOTE_AVAILABLE:
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
        return ImbPipeline(steps + [("model", model)])
    return Pipeline(steps + [("model", model)])


cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# --- 3. Comparaison des modeles (hyperparametres fixes, pas de recherche) -
#
# Tous les hyperparametres ci-dessous ont ete trouves par RandomizedSearchCV (voir rapport section 3)

#print("Evaluation des modeles (5-fold CV)...")

# Logistic Regression (baseline)
lr_pipeline = make_pipeline(
    LogisticRegression(class_weight="balanced", max_iter=2000,
                       random_state=RANDOM_STATE, n_jobs=N_JOBS)
)
#lr_scores = cross_val_score(lr_pipeline, X_train, y_train, cv=cv5, scoring="balanced_accuracy", n_jobs=N_JOBS)
#print(f"  LR           BCR = {lr_scores.mean():.4f} +/- {lr_scores.std():.4f}")

# Random Forest (tuned) - 120 iters de recherche
rf_pipeline = make_pipeline(
    RandomForestClassifier(
        n_estimators=1000, max_depth=10, min_samples_split=10,
        min_samples_leaf=4, max_features=0.2, bootstrap=True,
        class_weight=None, random_state=RANDOM_STATE, n_jobs=N_JOBS,
    )
)
#rf_scores = cross_val_score(rf_pipeline, X_train, y_train, cv=cv5, scoring="balanced_accuracy", n_jobs=N_JOBS)
#print(f"  Random Forest BCR = {rf_scores.mean():.4f} +/- {rf_scores.std():.4f}")

# HistGradientBoosting (tuned) - 150 iters de recherche
gb_pipeline = make_pipeline(
    HistGradientBoostingClassifier(
        max_iter=100, max_depth=5, learning_rate=0.01,
        min_samples_leaf=50, max_leaf_nodes=31, max_features=0.5,
        l2_regularization=0.0, early_stopping=True,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
)
#gb_scores = cross_val_score(gb_pipeline, X_train, y_train, cv=cv5, scoring="balanced_accuracy", n_jobs=N_JOBS)
#print(f"  HistGradBoost BCR = {gb_scores.mean():.4f} +/- {gb_scores.std():.4f}")

# XGBoost (tuned) - 200 iters de recherche
if XGB_AVAILABLE:
    xgb_pipeline = make_pipeline(
        xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.001,
            subsample=0.5, colsample_bytree=0.3, gamma=0,
            reg_alpha=0.5, reg_lambda=5, min_child_weight=1,
            max_bin=256, scale_pos_weight=1,
            objective="binary:logistic", tree_method="hist",
            device="cpu", nthread=-1, random_state=RANDOM_STATE,
        )
    )
    #xgb_scores = cross_val_score(xgb_pipeline, X_train, y_train, cv=cv5, scoring="balanced_accuracy", n_jobs=N_JOBS)
    #print(f"  XGBoost       BCR = {xgb_scores.mean():.4f} +/- {xgb_scores.std():.4f}")

# SVM RBF (tuned) - 150 iters de recherche
svm_pipeline = make_pipeline(
    SVC(kernel="rbf", C=2.683, gamma=1.326e-05,
        class_weight="balanced", probability=True,
        random_state=RANDOM_STATE, cache_size=2000)
)
#svm_scores = cross_val_score(svm_pipeline, X_train, y_train, cv=cv5, scoring="balanced_accuracy", n_jobs=N_JOBS)
#print(f"  SVM (RBF)     BCR = {svm_scores.mean():.4f} +/- {svm_scores.std():.4f}")

# GaussianNB (tuned) - 30 iters, var_smoothing sur echelle log
gnb_pipeline = make_pipeline(
    GaussianNB(var_smoothing=5.354e-07)
)
#gnb_scores = cross_val_score(gnb_pipeline, X_train, y_train, cv=cv5, scoring="balanced_accuracy", n_jobs=N_JOBS)
#print(f"  GaussianNB    BCR = {gnb_scores.mean():.4f} +/- {gnb_scores.std():.4f}")

# LDA (tuned) - shrinkage Ledoit-Wolf, 50 iters
lda_pipeline = make_pipeline(
    LinearDiscriminantAnalysis(solver="lsqr", shrinkage=1.0, tol=1e-4)
)
#lda_scores = cross_val_score(lda_pipeline, X_train, y_train, cv=cv5, scoring="balanced_accuracy", n_jobs=N_JOBS)
#print(f"  LDA           BCR = {lda_scores.mean():.4f} +/- {lda_scores.std():.4f}")

# QDA + PCA(100) meilleur modele individuel (BCR = 0.768)
qda_pipeline = make_pipeline(
    QuadraticDiscriminantAnalysis(reg_param=0.8944, tol=0.001),
    pca_n=100,
)
#qda_scores = cross_val_score(qda_pipeline, X_train, y_train, cv=cv5, scoring="balanced_accuracy", n_jobs=N_JOBS)
#print(f"  QDA+PCA(100)  BCR = {qda_scores.mean():.4f} +/- {qda_scores.std():.4f}")

# Gaussian Stack : GNB + LDA + QDA avec meta-learner LogisticRegression
# Les trois modeles Gaussiens sont complementaires (covariances differentes).
gnb_base = deepcopy(gnb_pipeline)
lda_base = deepcopy(lda_pipeline)
qda_base = deepcopy(qda_pipeline)

gaussian_stack = StackingClassifier(
    estimators=[("gnb", gnb_base), ("lda", lda_base), ("qda", qda_base)],
    final_estimator=LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
    ),
    cv=cv5, n_jobs=1, passthrough=False,
)
#stack_scores = cross_val_score(gaussian_stack, X_train, y_train, cv=cv5, scoring="balanced_accuracy", n_jobs=1)
#print(f"  Gaussian Stack BCR = {stack_scores.mean():.4f} +/- {stack_scores.std():.4f}")

# Modele retenu : QDA+PCA(100) 
# le stacking n'apporte pas de gain significatif 
final_pipeline = qda_pipeline


# --- 4. Estimation du BCRhat (OOF, folds independants) ------------------
#
# Seed different de la recherche (99 != 42) : folds independants de la selection

EVAL_N_SPLITS     = 10
EVAL_RANDOM_STATE = 99

eval_cv = StratifiedKFold(n_splits=EVAL_N_SPLITS, shuffle=True, random_state=EVAL_RANDOM_STATE)

oof_probas = cross_val_predict(
    final_pipeline, X_train, y_train,
    cv=eval_cv, method="predict_proba", verbose=1,
)[:, 1]

# Scan du seuil optimal sur les probas OOF (0.5 sous-optimal sur donnees desequilibrees).
thresholds = np.linspace(0.05, 0.95, 91)
bcr_at_t   = np.array([
    balanced_accuracy_score(y_train, (oof_probas >= t).astype(int))
    for t in thresholds
])
best_idx          = int(np.argmax(bcr_at_t))
optimal_threshold = float(thresholds[best_idx])
bcr_hat           = float(bcr_at_t[best_idx])
oof_preds         = (oof_probas >= optimal_threshold).astype(int)

#print(f"BCR seuil=0.50 : {balanced_accuracy_score(y_train, (oof_probas >= 0.5).astype(int)):.4f}")
#print(f"BCR seuil={optimal_threshold:.2f} : {bcr_hat:.4f}")


# --- 5. Sigma et correction conservative (BCR shrinkage) ----------------
#
# sigma_theorique : formule WCCI 2006, Var(BER) = 0.25*(FNR*(1-FNR)/n+ + FPR*(1-FPR)/n-)
# sigma_empirique : std des scores par fold / sqrt(k)
# predicted_BCR = bcr_hat - alpha * sigma  (alpha=0.5, moderate pessimism)
# --> valeur à soumettr

cm = confusion_matrix(y_train, oof_preds)
tn, fp, fn, tp = cm.ravel()
n_pos, n_neg = tp + fn, tn + fp
fnr, fpr     = fn / max(n_pos, 1), fp / max(n_neg, 1)

sigma_theoretical = float(np.sqrt(
    0.25 * (fnr * (1 - fnr) / max(n_pos, 1) + fpr * (1 - fpr) / max(n_neg, 1))
))

fold_scores = []
for train_idx, val_idx in eval_cv.split(X_train, y_train):
    fold_pipe = deepcopy(final_pipeline)
    fold_pipe.fit(X_train[train_idx], y_train[train_idx])
    fold_probas = fold_pipe.predict_proba(X_train[val_idx])[:, 1]
    fold_preds  = (fold_probas >= optimal_threshold).astype(int)
    fold_scores.append(balanced_accuracy_score(y_train[val_idx], fold_preds))

fold_scores     = np.array(fold_scores)
sigma_empirical = fold_scores.std() / np.sqrt(len(fold_scores))

sigma_used    = max(sigma_empirical, sigma_theoretical)
alpha         = 0.5 if sigma_used <= 0.03 else 1.0
predicted_bcr = float(bcr_hat - alpha * sigma_used)

#print(f"BCRhat OOF      : {bcr_hat:.4f}")
#print(f"sigma utilise   : {sigma_used:.4f}  (theorique={sigma_theoretical:.4f}, empirique={sigma_empirical:.4f})")
#print(f"IC 95%%          : [{bcr_hat - 1.96*sigma_used:.4f}, {bcr_hat + 1.96*sigma_used:.4f}]")
#print(f"Scores par fold : {np.round(fold_scores, 4)}")
#print(f"BCRhat a soumettre (Q2) : {predicted_bcr:.4f}")


# --- 6. Entrainement final sur toutes les donnees d'entrainement ---------

fitted_pipeline = deepcopy(final_pipeline)
fitted_pipeline.fit(X_train, y_train)
#print("Entrainement final termine.")


# --- 7. Generation des predictions sur le test set -----------------------

test_probas    = fitted_pipeline.predict_proba(X_test)[:, 1]
test_preds_int = (test_probas >= optimal_threshold).astype(int)
test_labels    = [REVERSE_LABEL_MAP[p] for p in test_preds_int]

output_df = pd.DataFrame({"label": test_labels}, index=range(len(test_labels)))

# index_label="" produit la virgule initiale requise par le format de la competition.
output_df.to_csv(PREDICTIONS_PATH, index=True, index_label="")

#print(f"Predictions ecrites : {PREDICTIONS_PATH}")
#print(output_df["label"].value_counts().to_string())
