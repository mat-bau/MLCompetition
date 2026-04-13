# data_loader.py
# Loads all three CSV files and encodes labels to integers.
# Returns numpy arrays and a raw labels series for inspection.

import numpy as np
import pandas as pd

from config import (
    TRAIN_FEATURES_PATH,
    TRAIN_LABELS_PATH,
    TEST_FEATURES_PATH,
    LABEL_MAP,
)


def load_data():
    """Load train features, train labels, and test features.

    Returns
    -------
    X_train : np.ndarray, shape (n_train, 1024)
    y_train : np.ndarray of int, shape (n_train,)
    X_test  : np.ndarray, shape (n_test, 1024)
    train_df       : pd.DataFrame of raw train features (for EDA)
    test_df        : pd.DataFrame of raw test features (for EDA)
    labels_series  : pd.Series of raw string labels (for EDA)
    """

    # Load train features.
    # The CSV has no explicit index column in its header, so the default
    # integer index (0, 1, 2, ...) is used.
    train_df = pd.read_csv(TRAIN_FEATURES_PATH)

    # Load labels.
    # The labels CSV has a leading comma in the header, meaning the first
    # column is an unnamed index. index_col=0 uses it as the DataFrame index.
    labels_df = pd.read_csv(TRAIN_LABELS_PATH, index_col=0)
    labels_df.columns = ["label"]

    # Load test features using the same convention as the train file.
    test_df = pd.read_csv(TEST_FEATURES_PATH)

    # Verify index alignment between features and labels before joining.
    if not train_df.index.equals(labels_df.index):
        # Try resetting the label index to align by position.
        labels_df = labels_df.reset_index(drop=True)

    # Map string labels to integers.
    labels_series = labels_df["label"].copy()
    y_train = labels_series.map(LABEL_MAP).values.astype(int)

    # Extract feature arrays as numpy float arrays.
    X_train = train_df.values.astype(np.float64)
    X_test  = test_df.values.astype(np.float64)

    return X_train, y_train, X_test, train_df, test_df, labels_series


if __name__ == "__main__":
    X_train, y_train, X_test, train_df, test_df, labels_series = load_data()
    print(f"X_train shape : {X_train.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"X_test  shape : {X_test.shape}")
    print(f"Label counts  :\n{labels_series.value_counts()}")