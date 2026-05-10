import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Attention : imbalanced-learn non installe, SMOTE desactive.")

TRAIN_FEATURES_PATH = "./A5_2026_train.csv"
TRAIN_LABELS_PATH   = "./A5_2026_train_labels.csv"
TEST_FEATURES_PATH  = "./A5_2026_test.csv"
PREDICTIONS_PATH    = "./predictions.csv"

LABEL_MAP         = {"positive": 1, "negative": 0}
REVERSE_LABEL_MAP = {1: "positive", 0: "negative"}
RANDOM_STATE      = 42


# -----------------------------------------------------------------------
# 1. Chargement des donnees
# -----------------------------------------------------------------------
#print("Chargement des donnees...")

train_df   = pd.read_csv(TRAIN_FEATURES_PATH)
labels_df  = pd.read_csv(TRAIN_LABELS_PATH, index_col=0)
labels_df.columns = ["label"]
test_df    = pd.read_csv(TEST_FEATURES_PATH)

# Alignement index features / labels par position
if not train_df.index.equals(labels_df.index):
    labels_df = labels_df.reset_index(drop=True)

y_train = labels_df["label"].map(LABEL_MAP).values.astype(int)
X_train = train_df.values.astype(np.float64)
X_test  = test_df.values.astype(np.float64)

#print(f"  X_train : {X_train.shape}   X_test : {X_test.shape}")
#print(f"  Classes : {np.bincount(y_train)}  (0=negatif, 1=positif)")


# -----------------------------------------------------------------------
# 2. Preprocessing
# -----------------------------------------------------------------------
preprocess_steps = [
    ("variance_filter", VarianceThreshold(threshold=0.0)),
    ("imputer",         SimpleImputer(strategy="median")),
    ("scaler",          RobustScaler()),
    ("pca",             PCA(n_components=100, random_state=RANDOM_STATE)),
]

# Hyperparametres trouvés par RandomizedSearchCV (200 iterations, 5 folds).
# reg_param melange QDA (0) vers GaussianNB (1) pour regulariser la covariance.
qda_params = {
    "reg_param": 0.8944,
    "tol":       0.001,
}

qda_model = QuadraticDiscriminantAnalysis(**qda_params)

# SMOTE applique uniquement sur le fold d'entrainement dans le CV (pas sur
# le fold de validation), grace à ImbPipeline qui respecte cet ordre.
if SMOTE_AVAILABLE:
    final_pipeline = ImbPipeline(
        preprocess_steps + [
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", qda_model),
        ]
    )
else:
    final_pipeline = Pipeline(
        preprocess_steps + [("model", qda_model)]
    )

#print("\nPipeline final :")
#for name, step in final_pipeline.steps:
#    print(f"  {name} : {step}")


# -----------------------------------------------------------------------
# 3. Estimation du BCRhat
#
# On utilise des folds independants de la recherche d'hyperparametres
# (EVAL_RANDOM_STATE=99 != CV_RANDOM_STATE=42)
# 10 folds = variance plus faible qu'avec 5 folds.
# -----------------------------------------------------------------------
EVAL_N_SPLITS     = 10
EVAL_RANDOM_STATE = 99


eval_cv = StratifiedKFold(
    n_splits=EVAL_N_SPLITS, shuffle=True, random_state=EVAL_RANDOM_STATE
)

# out-of-fold predict. avec proba pour scanner le seuil optimal.
oof_probas = cross_val_predict(
    final_pipeline, X_train, y_train,
    cv=eval_cv, method="predict_proba", verbose=1,
)[:, 1]

# Scan du seuil de decision entre 0.05 et 0.95 pour maximiser le BCR.
thresholds = np.linspace(0.05, 0.95, 91)
bcr_at_t   = np.array([
    balanced_accuracy_score(y_train, (oof_probas >= t).astype(int))
    for t in thresholds
])
best_idx          = int(np.argmax(bcr_at_t))
optimal_threshold = float(thresholds[best_idx])
bcr_hat           = float(bcr_at_t[best_idx])
oof_preds         = (oof_probas >= optimal_threshold).astype(int)

#print(f"\n  BCR seuil = 0.50  : {balanced_accuracy_score(y_train, (oof_probas >= 0.5).astype(int)):.4f}")
#print(f"  BCR seuil={optimal_threshold:.2f}  : {bcr_hat:.4f}  ")


# -----------------------------------------------------------------------
# 4. BCR shrinkage, sigma 
# predicted_BCR = bcr_hat - alpha * sigma
# -----------------------------------------------------------------------
cm = confusion_matrix(y_train, oof_preds)
tn, fp, fn, tp = cm.ravel()
n_pos = tp + fn
n_neg = tn + fp
fnr   = fn / max(n_pos, 1)
fpr   = fp / max(n_neg, 1)

var_ber           = 0.25 * (fnr * (1 - fnr) / max(n_pos, 1) + fpr * (1 - fpr) / max(n_neg, 1))
sigma_theoretical = float(np.sqrt(var_ber))

# sigma empirique : std des scores par fold / sqrt(k)
fold_scores = []
for train_idx, val_idx in eval_cv.split(X_train, y_train):
    fold_pipe = deepcopy(final_pipeline)
    fold_pipe.fit(X_train[train_idx], y_train[train_idx])
    fold_probas = fold_pipe.predict_proba(X_train[val_idx])[:, 1]
    fold_preds  = (fold_probas >= optimal_threshold).astype(int)
    fold_scores.append(balanced_accuracy_score(y_train[val_idx], fold_preds))

fold_scores     = np.array(fold_scores)
sigma_empirical = fold_scores.std() / np.sqrt(len(fold_scores))

# On prend le plus conservateur des deux sigmas.
sigma_used    = max(sigma_empirical, sigma_theoretical) #min ou max change rien pour le rapport negative/positive du test set classifié
alpha         = 0.5 if sigma_used <= 0.03 else 1.0
predicted_bcr = float(bcr_hat - alpha * sigma_used)

#print(f"\n  BCRhat OOF: {bcr_hat:.4f}")
#print(f"  sigma: {sigma_used:.4f}  (theorique={sigma_theoretical:.4f}, empirique={sigma_empirical:.4f})")
#print(f"  IC 95: [{bcr_hat - 1.96*sigma_used:.4f}, {bcr_hat + 1.96*sigma_used:.4f}]")
#print(f"  alpha: {alpha}")
#print(f"  Scores par fold: {np.round(fold_scores, 4)}")


# -----------------------------------------------------------------------
# 5. Entrainement final sur toutes les donnees d'entrainement
# -----------------------------------------------------------------------
fitted_pipeline = deepcopy(final_pipeline)
fitted_pipeline.fit(X_train, y_train)
#print("Entrainement terminé")


# -----------------------------------------------------------------------
# 6. Generation des predictions sur le test set
# -----------------------------------------------------------------------

test_probas      = fitted_pipeline.predict_proba(X_test)[:, 1]
test_preds_int   = (test_probas >= optimal_threshold).astype(int)
test_labels      = [REVERSE_LABEL_MAP[p] for p in test_preds_int]

output_df = pd.DataFrame(
    {"label": test_labels},
    index=range(len(test_labels)),
)

# Format requis par la competition : virgule avant "label" dans le header.
output_df.to_csv(PREDICTIONS_PATH, index=True, index_label="")

#print(f"  predictions.csv ecrit dans : {PREDICTIONS_PATH}")
#print(f"  Total predictions : {len(output_df)}")
#print(f"  Distribution :\n{output_df['label'].value_counts().to_string()}")