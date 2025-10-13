"""
Prepare a processed dataset from raw Reddit data.

Loads raw CSV, cleans text, engineers labels, performs stratified train/val/test split,
and saves Parquet files to data/processed/.

Usage:
    uv run python -m src.data.make_dataset --test_size 0.15 --random_state 42

Requirements:
    - Raw data at data/raw/reddit_comments.csv (from download_dataset.py).
    - uv sync (for pandas, scikit-learn, nltk).

Design Considerations:
- Reliability: Input validation, NaN handling, post-split integrity checks.
- Scalability: Parquet for efficient I/O; vectorized pandas operations.
- Maintainability: Logging, type hints, modular functions; paths relative to script.
- Adaptability: Parameterized splits; extensible cleaning (e.g., add lemmatization).
"""

import argparse
from typing import Optional
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Project Utilities ---
from src.utils.paths import RAW_DATA_DIR, TRAIN_PATH, VALIDATION_PATH, TEST_PATH
from src.utils.logger import get_logger

# --- Logging Setup ---
logger = get_logger(__name__, headline="make_dataset.py")

# --- NLTK Setup ---
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# --- File Paths ---
RAW_PATH = RAW_DATA_DIR / "reddit_comments.csv"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str, stop_words: Optional[set] = None) -> str:
    """
    Enhanced text cleaning: lowercase, remove non-alphabetic except spaces,
    remove stopwords, strip whitespace. Retains sentiment signals.

    Args:
        text (str): Input text.
        stop_words (set, optional): NLTK stopwords to remove.

    Returns:
        str: Cleaned text.
    """
    if pd.isna(text):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    if stop_words:
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
        text = " ".join(tokens)
    return text


def prepare_reddit_dataset(test_size: float = 0.15, random_state: int = 42) -> None:
    """
    Orchestrate data preparation.

    Args:
        test_size (float): Fraction for test split.
        random_state (int): Seed for reproducibility.
    """

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Raw data missing: {RAW_PATH}. Run data ingestion first."
        )

    # Load and initial validation
    df = pd.read_csv(RAW_PATH)
    if df.empty or "clean_comment" not in df.columns or "category" not in df.columns:
        raise ValueError("Invalid raw data structure.")
    logger.info(f"Loaded {len(df)} rows from raw data with shape: {df.shape}.")

    # --- Label normalization for consistency ---
    # Many ML tools (e.g., np.bincount, SMOTE, StratifiedKFold) require non-negative labels.
    unique_labels = sorted(df["category"].unique())
    # This ensures your raw data always has the expected structure
    if unique_labels != [-1, 0, 1]:
        raise ValueError(f"Unexpected category labels: {unique_labels}")
    # Map {-1, 0, 1} → {0, 1, 2}
    df["category_encoded"] = df["category"].map({-1: 0, 0: 1, 1: 2})
    logger.info(
        f"Original label distribution: {dict(df['category'].value_counts().sort_index())}"
    )
    logger.info(
        f"Encoded label distribution: {dict(df['category_encoded'].value_counts().sort_index())}"
    )

    # Cleaning
    stop_words = set(stopwords.words("english"))
    df = df.dropna(subset=["clean_comment"])
    df["clean_comment"] = df["clean_comment"].apply(lambda x: clean_text(x, stop_words))
    df = df[df["clean_comment"].str.len() > 0]
    logger.info(f"Dataset after cleaning: {len(df)} rows.")

    # Label engineering
    label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
    df["sentiment_label"] = df["category"].map(label_map)

    # Stratified split
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["category"]
    )
    val_size = test_size / (1 - test_size)  # ~0.1765 for 15% val from 85%
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=train_val["category"],
    )

    # Save and validate shapes
    outputs = [
        (TRAIN_PATH, train),
        (VALIDATION_PATH, val),
        (TEST_PATH, test),
    ]
    for out_path, split_df in outputs:
        split_df.to_parquet(out_path, index=False)
        if split_df.empty:
            raise ValueError(f"Empty split for {out_path}.")

    # Log splits
    logger.info(
        f"Splits prepared: Train {train.shape[0]}, Val {val.shape[0]}, Test {test.shape[0]}"
    )
    logger.info(
        f"Train class distribution: {train['category'].value_counts().to_dict()}"
    )


def main() -> None:
    """Parse args and run preparation."""
    parser = argparse.ArgumentParser(description="Prepare processed dataset.")
    parser.add_argument(
        "--test_size", type=float, default=0.15, help="Test split fraction."
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    logger.info(
        f"--- Preparing dataset with test_size={args.test_size} and random_state={args.random_state} ---"
    )
    prepare_reddit_dataset(args.test_size, args.random_state)


if __name__ == "__main__":
    main()
