# run_manager.py
# Manages per-run output directories, structured logging, and metrics export.
#
# Each call to main.py produces one timestamped folder:
#
#   results/
#     run_20260413_143022/
#       logs.txt          ← full console output (via TeeLogger)
#       config.json       ← all config.py constants
#       metrics.json      ← BCRhat, BER_guess, sigma, model, threshold, …
#       predictions.csv   ← copy of the submission file
#       plots/            ← all plots (PLOTS_DIR redirected here)

import os
import sys
import json
import shutil
import time


# -------------------------------------------------------------------
# Run directory creation
# -------------------------------------------------------------------

def create_run_dir(results_root: str) -> str:
    """Create a timestamped output directory under *results_root*.

    Parameters
    ----------
    results_root : str  e.g. "./results"

    Returns
    -------
    run_dir : str  absolute path of the newly created directory
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir   = os.path.join(results_root, f"run_{timestamp}")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"  Run directory : {run_dir}")
    return run_dir


# -------------------------------------------------------------------
# Console logging (Tee to file)
# -------------------------------------------------------------------

class TeeLogger:
    """Duplicate all stdout writes to a log file.

    Usage
    -----
    sys.stdout = TeeLogger(run_dir)
    ...
    sys.stdout.close()          # restores original stdout
    """

    def __init__(self, run_dir: str, append: bool = False):
        self._log_path = os.path.join(run_dir, "logs.txt")
        mode           = "a" if append else "w"
        self._file     = open(self._log_path, mode, buffering=1, encoding="utf-8")
        self._original = sys.stdout
        if append:
            self._file.write(f"\n{'='*60}\n  RESUMED at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n")

    def write(self, text: str) -> None:
        self._original.write(text)
        self._file.write(text)

    def flush(self) -> None:
        self._original.flush()
        self._file.flush()

    def close(self) -> None:
        sys.stdout = self._original
        self._file.close()

    # Make the Tee act as the real stdout for anything that inspects it.
    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()


def setup_logging(run_dir: str, append: bool = False) -> TeeLogger:
    """Replace sys.stdout with a TeeLogger that writes to run_dir/logs.txt.

    Parameters
    ----------
    append : bool
        If True, appends to an existing logs.txt (resume mode) instead of
        overwriting it.  A separator line is written to mark the resume point.

    Returns the TeeLogger so the caller can call .close() at the end.
    """
    tee = TeeLogger(run_dir, append=append)
    sys.stdout = tee
    return tee


# -------------------------------------------------------------------
# Config serialisation
# -------------------------------------------------------------------

def save_config(run_dir: str) -> None:
    """Serialise all public constants from config.py to config.json."""
    import config as cfg_module

    serialisable = {}
    for name in dir(cfg_module):
        if name.startswith("_"):
            continue
        value = getattr(cfg_module, name)
        # Keep only JSON-serialisable scalar / list values.
        if isinstance(value, (int, float, str, bool, list, dict)):
            serialisable[name] = value

    path = os.path.join(run_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, default=str)
    print(f"  config.json   → {path}")


# -------------------------------------------------------------------
# Metrics export
# -------------------------------------------------------------------

def save_metrics(run_dir: str, metrics: dict) -> None:
    """Write a metrics dict to metrics.json in the run directory.

    Parameters
    ----------
    run_dir : str
    metrics : dict  — e.g. {
        "model"              : "SVM RBF (tuned)",
        "bcr_hat"            : 0.7560,
        "predicted_bcr"      : 0.7432,
        "BER_guess"          : 0.2568,
        "sigma_empirical"    : 0.0128,
        "sigma_theoretical"  : 0.0114,
        "sigma_used"         : 0.0128,
        "optimal_threshold"  : 0.27,
        "robustness_score"   : 0.0093,
        "shrinkage_alpha"    : 0.5,
    }
    """
    path = os.path.join(run_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  metrics.json  → {path}")


# -------------------------------------------------------------------
# Predictions copy
# -------------------------------------------------------------------

def copy_predictions(predictions_path: str, run_dir: str) -> None:
    """Copy the submission predictions.csv into the run directory."""
    if not os.path.exists(predictions_path):
        print(f"  [run_manager] WARNING: {predictions_path} not found — skipping copy.")
        return
    dest = os.path.join(run_dir, "predictions.csv")
    shutil.copy2(predictions_path, dest)
    print(f"  predictions   → {dest}")
