"""
Data loading and preprocessing utilities for all model training scripts.
Handles both TF-IDF (sparse) and text-based datasets, with ADASYN support for imbalance correction.
All models read from the canonical `models/advanced/` directory.
"""

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from imblearn.over_sampling import ADASYN

# --- Project Utilities ---
from src.utils.paths import PROCESSED_DATA_DIR, FEATURES_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="data_loader.py")


# ============================================================
#  TF-IDF / Sparse Feature Loading
# ============================================================
def load_feature_data(validate_files: bool = True):
    """
    Load pre-engineered TF-IDF features and label encoder from the unified models/advanced directory.

    Args:
        validate_files (bool): If True, raises FileNotFoundError when any required file is missing.

    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, label_encoder)
    """
    feature_files = {
        "X_train": FEATURES_DIR / "X_train.npz",
        "X_val": FEATURES_DIR / "X_val.npz",
        "X_test": FEATURES_DIR / "X_test.npz",
        "y_train": FEATURES_DIR / "y_train.npy",
        "y_val": FEATURES_DIR / "y_val.npy",
        "y_test": FEATURES_DIR / "y_test.npy",
        "label_encoder": FEATURES_DIR / "label_encoder.pkl",
    }

    if validate_files:
        for name, path in feature_files.items():
            if not path.exists():
                logger.error(f"Missing feature file: {path}")
                raise FileNotFoundError(f"Missing feature file: {path}")

    logger.info("Loading pre-engineered TF-IDF features and labels...")

    X_train = load_npz(feature_files["X_train"]).tocsr()
    X_val = load_npz(feature_files["X_val"]).tocsr()
    X_test = load_npz(feature_files["X_test"]).tocsr()

    y_train = np.load(feature_files["y_train"])
    y_val = np.load(feature_files["y_val"])
    y_test = np.load(feature_files["y_test"])

    import pickle

    with open(feature_files["label_encoder"], "rb") as f:
        le = pickle.load(f)

    logger.info(
        f"Loaded features: "
        f"X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, le


# ============================================================
#  Text Data Loading (for Transformer Models)
# ============================================================
def load_text_data():
    """
    Load text data for transformer-based models (e.g., BERT) from data/processed/.
    Returns:
        tuple: (train_df, val_df)
    """
    train_path = PROCESSED_DATA_DIR / "train.parquet"
    val_path = PROCESSED_DATA_DIR / "val.parquet"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Processed text data not found in data/processed/.")

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    logger.info(f"Loaded text data: Train={len(train_df)}, Val={len(val_df)}")
    return train_df, val_df


# ============================================================
#  ADASYN Oversampling - Best method from imbalance_tuning (F1-positive: 0.7333)
# ============================================================
def apply_adasyn(X_train, y_train):
    """
    Apply ADASYN to the training data to mitigate class imbalance.

    Args:
        X_train (array or sparse matrix): Training features.
        y_train (array): Training labels.

    Returns:
        tuple: (X_resampled, y_resampled)
    """
    logger.info("Applying ADASYN oversampling for class imbalance correction...")
    adasyn = ADASYN(random_state=42, n_neighbors=5)
    X_res, y_res = adasyn.fit_resample(X_train, y_train)

    logger.info(
        f"Resampled dataset shapes — X: {X_res.shape}, y: {y_res.shape} | "
        f"Original: {len(y_train)}, Resampled: {len(y_res)}"
    )
    return X_res, y_res
