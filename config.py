# config.py
# Central configuration for all paths, constants, and flags.
# Edit this file to change file paths or global settings.

import os
import multiprocessing

# -------------------------------------------------------------------
# File paths (relative to the working directory where main.py is run)
# -------------------------------------------------------------------
TRAIN_FEATURES_PATH = "./A5_2026_train.csv"
TRAIN_LABELS_PATH   = "./A5_2026_train_labels.csv"
TEST_FEATURES_PATH  = "./A5_2026_test.csv"
PREDICTIONS_PATH    = "./predictions.csv"

# -------------------------------------------------------------------
# Output directory for plots
# -------------------------------------------------------------------
PLOTS_DIR = "./plots"

# -------------------------------------------------------------------
# Label encoding convention
# positive (toxic)     -> 1
# negative (not toxic) -> 0
# -------------------------------------------------------------------
LABEL_MAP         = {"positive": 1, "negative": 0}
REVERSE_LABEL_MAP = {1: "positive", 0: "negative"}

# -------------------------------------------------------------------
# Cross-validation settings
# -------------------------------------------------------------------
CV_N_SPLITS   = 5
CV_RANDOM_STATE = 42

# -------------------------------------------------------------------
# Parallelism -- use all available CPU cores on the Mac Studio
# -------------------------------------------------------------------
N_JOBS = -1  
_n_physical = multiprocessing.cpu_count()

# -------------------------------------------------------------------
# Imbalance threshold: flag if minority class is below this fraction
# -------------------------------------------------------------------
IMBALANCE_THRESHOLD = 0.30

# -------------------------------------------------------------------
# Variance filter threshold for near-constant feature detection
# -------------------------------------------------------------------
NEAR_CONSTANT_VARIANCE = 0.01

# -------------------------------------------------------------------
# Correlation threshold for flagging near-duplicate feature pairs
# -------------------------------------------------------------------
HIGH_CORR_THRESHOLD = 0.98

# -------------------------------------------------------------------
# Outlier detection: number of standard deviations from the mean
# -------------------------------------------------------------------
OUTLIER_STD_THRESHOLD = 5.0

# -------------------------------------------------------------------
# Distribution shift: flag if mean difference exceeds this many stds
# -------------------------------------------------------------------
DIST_SHIFT_THRESHOLD = 0.5

# -------------------------------------------------------------------
# RandomizedSearchCV iterations per model
# More iterations = better search coverage (Mac Studio handles it)
# -------------------------------------------------------------------
XGB_N_ITER = 80
SVM_N_ITER = 50
MLP_N_ITER = 40

# -------------------------------------------------------------------
# Random seed used everywhere for reproducibility
# -------------------------------------------------------------------
RANDOM_STATE = 42

# -------------------------------------------------------------------
# Feature selection k values to try with SelectKBest
# -------------------------------------------------------------------
SELECT_K_VALUES = [100, 200, 300, 500, "all"]

# -------------------------------------------------------------------
# PCA n_components values to try
# -------------------------------------------------------------------
PCA_N_VALUES = [50, 100, 200, 300]

# -------------------------------------------------------------------
# Phase 6 evaluation CV — deliberately different from CV_RANDOM_STATE
# to produce independent folds and reduce optimism bias (WCCI 2006).
# 10 folds gives a lower-variance estimate than 5.
# -------------------------------------------------------------------
EVAL_N_SPLITS     = 10
EVAL_RANDOM_STATE = 99   # != CV_RANDOM_STATE → independent from tuning folds

# -------------------------------------------------------------------
# BCR shrinkage: predicted_BCR = bcr_hat - SHRINKAGE_ALPHA * sigma
# Corrects the documented optimism bias of CV estimates.
# -------------------------------------------------------------------
SHRINKAGE_ALPHA = 0.5

# -------------------------------------------------------------------
# Seeds used for multi-seed robustness evaluation (Phase 6)
# -------------------------------------------------------------------
ROBUSTNESS_SEEDS = [42, 0, 7, 123, 2026]

# -------------------------------------------------------------------
# Root directory for per-run output (timestamped subfolders)
# -------------------------------------------------------------------
RESULTS_DIR = "./results"
