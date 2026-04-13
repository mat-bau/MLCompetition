# predict.py
# Phase 7 -- Prediction and Output.
# Uses the final fitted pipeline to predict on the test set and writes
# predictions.csv in the required competition format.

import pandas as pd

from config import PREDICTIONS_PATH, REVERSE_LABEL_MAP


def generate_predictions(fitted_pipeline, X_test):
    """Predict labels for the test set.

    Parameters
    ----------
    fitted_pipeline : sklearn Pipeline fitted on all training data
    X_test          : np.ndarray of shape (n_test, n_features)

    Returns
    -------
    test_labels : list of string labels ('positive' or 'negative')
    """
    test_predictions_int = fitted_pipeline.predict(X_test)
    test_labels = [REVERSE_LABEL_MAP[p] for p in test_predictions_int]
    return test_labels


def write_predictions(test_labels, output_path=PREDICTIONS_PATH):
    """Write predictions to a CSV file in the required competition format.

    The output format is:
        ,label
        0,positive
        1,negative
        ...
        999,negative

    Parameters
    ----------
    test_labels : list of string labels
    output_path : str, path where the CSV will be written

    Returns
    -------
    output_df : pd.DataFrame with the saved predictions
    """
    # Build the output DataFrame.
    # The index runs from 0 to len(test_labels) - 1.
    output_df = pd.DataFrame(
        {"label": test_labels},
        index=range(len(test_labels)),
    )

    # index_label='' writes an empty string before 'label' in the header,
    # which produces the leading comma required by the competition format.
    output_df.to_csv(output_path, index=True, index_label="")

    return output_df


def run_predictions(fitted_pipeline, X_test, output_path=PREDICTIONS_PATH):
    """Generate predictions, write the file, and run sanity checks.

    Parameters
    ----------
    fitted_pipeline : sklearn Pipeline fitted on all training data
    X_test          : np.ndarray of shape (n_test, n_features)
    output_path     : str, path where the CSV will be written
    """
    print("\n" + "#" * 60)
    print("# PHASE 7 -- GENERATING PREDICTIONS")
    print("#" * 60)

    test_labels = generate_predictions(fitted_pipeline, X_test)
    output_df   = write_predictions(test_labels, output_path)

    # Sanity checks before submission.
    assert len(output_df) == len(test_labels), "Prediction count mismatch."
    assert set(output_df["label"].unique()).issubset({"positive", "negative"}), \
        "Unexpected label values found in predictions."

    print(f"predictions.csv written to: {output_path}")
    print(f"Total predictions : {len(output_df)}")
    print("Label distribution:")
    print(output_df["label"].value_counts().to_string())

    return output_df