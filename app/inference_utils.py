"""
Inference utilities for sentiment analysis prediction services.

Provides:
- Model loading from MLflow Model Registry (with local fallback).
- Derived feature engineering (e.g., lengths, lexicon ratios) for consistent preprocessing.

Usage:
    from app.inference_utils import load_production_model, build_derived_features
"""

import joblib
import mlflow
import pandas as pd
import numpy as np
from typing import Any, Set

# --- Project Utilities ---
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri
from src.utils.paths import ADVANCED_DIR, PROJECT_ROOT

# --- Logging Setup ---
logger = get_logger(__name__, headline="inference_utils.py")


def load_production_model(
    model_name: str = "youtube_sentiment_lightgbm", alias_name: str = "Production"
) -> Any:
    """
    Loads the trained model object, prioritizing MLflow Model Registry with the
    '@Production' alias, and falling back to a local DVC-tracked PKL file.

    MLflow Priority:
        1. Try to load the model artifact from MLflow Model Registry using the
           '@Production' alias.
        2. Sets the tracking URI internally using the environment configuration.

    Local Fallback:
        If MLflow loading fails, fall back to loading the locally DVC-tracked
        'lightgbm_model.pkl'.

    Returns:
        Any: The loaded model instance (either an mlflow.pyfunc.PyFuncModel
             or a LightGBM Booster/Scikit-learn wrapper).

    Raises:
        RuntimeError: If model loading fails from both sources.
    """
    # 1. Attempt MLflow Model Registry Load
    try:
        mlflow_uri = get_mlflow_uri()
        mlflow.set_tracking_uri(mlflow_uri)

        model_uri = f"models:/{model_name}@{alias_name}"
        model = mlflow.pyfunc.load_model(model_uri=model_uri)

        logger.info(
            f"✅ Loaded model from MLflow registry → {model_name}@{alias_name} "
            f"(Tracking URI: {mlflow_uri})"
        )

        return model

    except Exception as e:
        # 2. Local Fallback Load
        model_path = ADVANCED_DIR / "lightgbm_model.pkl"

        try:
            # Load the local model artifact
            model = joblib.load(model_path)
            logger.warning(
                f"⚠️ MLflow registry unavailable or alias not found. "
                f"Loaded local LightGBM model from {model_path.relative_to(PROJECT_ROOT)}. "
                f"Original MLflow error: {e}"
            )
            return model

        except FileNotFoundError:
            logger.error(
                f"❌ Local model fallback failed. Model file not found at "
                f"{model_path.relative_to(PROJECT_ROOT)}."
            )
            raise RuntimeError(
                "Failed to load model from both MLflow and local filesystem. "
                "Ensure MLflow is running or 'lightgbm_model.pkl' is DVC-pulled."
            )


def build_derived_features(df: pd.DataFrame) -> np.ndarray:
    """
    Recreate simple derived features used during training.

    Features:
        - char_len: Character length of cleaned comment.
        - word_len: Word count of cleaned comment.
        - pos_ratio: Ratio of positive lexicon words.
        - neg_ratio: Ratio of negative lexicon words.

    Args:
        df: DataFrame with 'clean_comment' column (preprocessed text).

    Returns:
        np.ndarray: 2D array of shape (n_samples, 4) with derived features.
    """
    df = df.copy()
    df["char_len"] = df["clean_comment"].str.len()
    df["word_len"] = df["clean_comment"].str.split().str.len()

    pos_words: Set[str] = {"good", "great", "love", "like", "positive", "best"}
    neg_words: Set[str] = {"bad", "hate", "worst", "negative", "shit", "fuck"}

    def count_lexicon_ratio(text: str, lexicon: Set[str]) -> float:
        words = text.split()
        return len([w for w in words if w in lexicon]) / max(len(words), 1)

    df["pos_ratio"] = df["clean_comment"].apply(
        lambda x: count_lexicon_ratio(x, pos_words)
    )
    df["neg_ratio"] = df["clean_comment"].apply(
        lambda x: count_lexicon_ratio(x, neg_words)
    )

    return df[["char_len", "word_len", "pos_ratio", "neg_ratio"]].values
