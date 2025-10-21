"""
Feature Engineering Module for Sentiment Analysis
=================================================

Generates and saves reusable feature matrices (X) and labels (y) as compressed
NumPy arrays for efficient downstream modeling. Supports both TF-IDF and BERT
representations, along with simple derived features. Integrates with MLOps via logging and paths.

Saves feature matrices as compressed NumPy sparse arrays (.npz) and labels
as NumPy arrays (.npy) to the models/features directory.

Usage (DVC pulls parameters from params.yaml):
    uv run python -m src.features.feature_engineering --use_bert False --max_features 1000 --ngram_range (1,1)

Requirements:
    - Processed data in data/processed/.
    - uv sync (for pandas, scikit-learn, torch, transformers, scipy).

Design Goals:
    - Reliability: Validations, consistent outputs, structured logging.
    - Scalability: Sparse TF-IDF, batched BERT embeddings, compressed storage.
    - Maintainability: Modular, documented, centralized configuration.
    - Adaptability: Easily switch vectorization techniques or feature sets.

Outputs:
    models/features
        ├── X_train.npz / y_train.npy
        ├── X_val.npz   / y_val.npy
        ├── X_test.npz  / y_test.npy
        ├── vectorizer.pkl (TF-IDF)
        ├── label_encoder.pkl
        └── tokenizer.pkl / bert_model.pth (if use_bert=True)
"""

import argparse
import pickle
from typing import Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import save_npz, hstack, csr_matrix, issparse
from scipy.sparse import spmatrix  # Added for improved type hinting

# --- Conditional Imports for BERT ---
# Define placeholders if imports fail to prevent NameError outside the try/except block
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    torch: Optional[Any] = None
    AutoTokenizer: Optional[Any] = None
    AutoModel: Optional[Any] = None

# --- Project Utilities ---
from src.utils.paths import TRAIN_PATH, VAL_PATH, TEST_PATH, FEATURES_DIR, PROJECT_ROOT
from src.utils.logger import get_logger
from src.features.helpers.feature_utils import parse_dvc_param

# --- Logging Setup ---
logger = get_logger(__name__, headline="feature_engineering.py")


def _get_bert_embeddings(
    texts: list, device: Optional[str] = None, batch_size: int = 32
) -> np.ndarray:
    """
    Generate mean-pooled BERT embeddings for a list of texts. Helper function.

    Args:
        texts (list): Cleaned text samples.
        device (str): "cuda" or "cpu". Auto-detected if None.
        batch_size (int): Batch size for batched inference.

    Returns:
        np.ndarray: Dense embeddings (n_samples, 768).
    """
    # NOTE: Checks rely on the conditional imports above
    if torch is None or AutoModel is None:
        raise ImportError("BERT mode requires torch and transformers to be installed.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"BERT Inference: Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased").to(device)
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512,
            ).to(device)
            outputs = model(**inputs)
            # Mean pool over non-special tokens
            pooled = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)
            embeddings.append(pooled.cpu().numpy())

    return np.vstack(embeddings)


def _add_derived_features(df: pd.DataFrame) -> np.ndarray:
    """Calculates dense, simple, derived features (length, word ratios).

    Returns:
        np.ndarray: Dense numeric features (char_len, word_len, pos_ratio, neg_ratio)
    """
    df_local = df.copy()
    df_local["char_len"] = df_local["clean_comment"].str.len()
    df_local["word_len"] = df_local["clean_comment"].str.split().str.len()

    # Simple lexicon-based features (can be expanded with VADER/other lexicon)
    pos_words = {"good", "great", "love", "like", "positive", "best"}
    neg_words = {"bad", "hate", "worst", "negative", "shit", "fuck"}

    def count_lexicon_ratio(text, lexicon):
        """Helper to count ratio of words in lexicon."""
        words = text.split()
        return len([w for w in words if w in lexicon]) / max(len(words), 1)

    df_local["pos_ratio"] = df_local["clean_comment"].apply(
        lambda x: count_lexicon_ratio(x, pos_words)
    )
    df_local["neg_ratio"] = df_local["clean_comment"].apply(
        lambda x: count_lexicon_ratio(x, neg_words)
    )

    return df_local[["char_len", "word_len", "pos_ratio", "neg_ratio"]].values


def engineer_features(
    use_bert: bool,
    max_features: int,
    ngram_range: Tuple[int, int],
    bert_batch_size: int,
) -> None:
    """
    Generates final feature matrices (X) and labels (y) based on selected parameters.

    Args:
        use_bert (bool): If True, use BERT embeddings.
        max_features (int): Max vocabulary for TF-IDF.
        ngram_range (Tuple[int, int]): N-gram range for TF-IDF.
        bert_batch_size (int): Batch size for BERT inference.
    """

    # 1. Load Data
    splits = {"train": TRAIN_PATH, "val": VAL_PATH, "test": TEST_PATH}
    dfs = {}
    for name, path in splits.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Processed data missing: {path}. Run data_preparation first."
            )
        dfs[name] = pd.read_parquet(path)
        # NOTE: Ensure 'clean_comment' exists and is not empty (Relying on make_dataset.py for cleaning)
        if dfs[name]["clean_comment"].empty:
            raise ValueError(f"Clean comments column is empty in {name} split.")
        logger.info(f"{name.capitalize()} set loaded: {dfs[name].shape[0]} samples.")

    # Extract texts and labels (using category_encoded for consistent non-negative labels)
    train_texts = dfs["train"]["clean_comment"].tolist()
    val_texts = dfs["val"]["clean_comment"].tolist()
    test_texts = dfs["test"]["clean_comment"].tolist()

    # We use the original category for the LabelEncoder fitting
    y_all = pd.concat([dfs[s]["category"] for s in splits])

    # 2. Label Encoding (fit on combined for consistency)
    le = LabelEncoder()
    le.fit(y_all)
    y_train = le.transform(dfs["train"]["category"])
    y_val = le.transform(dfs["val"]["category"])
    y_test = le.transform(dfs["test"]["category"])

    # 3. Text Feature Generation (TF-IDF or BERT)
    vectorizer: Optional[TfidfVectorizer] = None  # Initialize vectorizer placeholder
    X_train_text: Union[spmatrix, np.ndarray]
    X_val_text: Union[spmatrix, np.ndarray]
    X_test_text: Union[spmatrix, np.ndarray]

    if use_bert:
        logger.info("🚀 Generating BERT embeddings (768 dim)...")
        X_train_text = _get_bert_embeddings(train_texts, batch_size=bert_batch_size)
        X_val_text = _get_bert_embeddings(val_texts, batch_size=bert_batch_size)
        X_test_text = _get_bert_embeddings(test_texts, batch_size=bert_batch_size)

    else:
        logger.info(
            f"🚀 Generating TF-IDF features (max_features={max_features}, ngram={ngram_range})..."
        )
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            lowercase=False,  # Text is already lowercased and pre-cleaned
            min_df=2,
        )
        X_train_text = vectorizer.fit_transform(train_texts)
        X_val_text = vectorizer.transform(val_texts)
        X_test_text = vectorizer.transform(test_texts)

    # 4. Derived Feature Generation (Dense)
    derived_train = _add_derived_features(dfs["train"])
    derived_val = _add_derived_features(dfs["val"])
    derived_test = _add_derived_features(dfs["test"])

    # 5. Combine Features (Text + Derived)
    X_sets = []
    text_sets = [X_train_text, X_val_text, X_test_text]
    derived_sets = [derived_train, derived_val, derived_test]

    for X_text, X_derived in zip(text_sets, derived_sets):
        if issparse(X_text):
            # Sparse (TF-IDF) + Dense (hstack converts result to sparse)
            X_combined = hstack([X_text, X_derived])
        else:
            # Dense (BERT) + Dense (numpy.hstack)
            X_combined_dense = np.hstack([X_text, X_derived])
            # For consistent saving as .npz (sparse format), convert dense numpy array
            # This ensures X_combined is always a sparse matrix type
            X_combined = csr_matrix(X_combined_dense)
        X_sets.append(X_combined)

    # 6. Save Artifacts (Features, Labels, Encoder, Vectorizer)
    splits = ["train", "val", "test"]
    X_train, X_val, X_test = X_sets

    # Save Features (.npz for efficient sparse storage)
    for split, X in zip(splits, X_sets):
        save_npz(FEATURES_DIR / f"X_{split}.npz", X)

    # Save Labels (.npy)
    for split, y in zip(splits, [y_train, y_val, y_test]):
        np.save(FEATURES_DIR / f"y_{split}.npy", y)

    # Save Encoder
    label_encoder_name = "label_encoder.pkl"
    with open(FEATURES_DIR / label_encoder_name, "wb") as f:
        pickle.dump(le, f)
        logger.info(
            f"Saved Label Encoder to {FEATURES_DIR.relative_to(PROJECT_ROOT) / label_encoder_name}"
        )

    # Save Vectorizer (if TF-IDF was used)
    if vectorizer:
        vectorizer_name = f"tfidf_vectorizer_max_{max_features}.pkl"
        with open(FEATURES_DIR / vectorizer_name, "wb") as f:
            pickle.dump(vectorizer, f)
            logger.info(
                f"Saved TF-IDF Vectorizer to {FEATURES_DIR.relative_to(PROJECT_ROOT) / vectorizer_name}"
            )
    elif use_bert:
        logger.info("Using BERT: Tokenizer/Model are sourced from HuggingFace.")

    feature_type = "BERT" if use_bert else f"TF-IDF (max_features={max_features})"
    logger.info(
        f"✅ Features engineered successfully | Type: {feature_type} | "
        f"Shapes → Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
    )


def main() -> None:
    """Parse args and run feature engineering."""
    parser = argparse.ArgumentParser(
        description="Generate final feature set using best parameters."
    )
    # NOTE: DVC will pass the best parameters found in the previous stages here
    parser.add_argument(
        "--use_bert",
        type=lambda x: x.lower() == "true",
        default=False,
        help="If True, use BERT embeddings; otherwise use TF-IDF.",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=1000,  # Best default based on prior experiments (tfidf_max_features.py)
        help="Max vocabulary for TF-IDF (ignored if use_bert=True).",
    )
    parser.add_argument(
        "--ngram_range",
        type=str,
        default="(1,1)",  # Best default based on prior experiments (tfidf_vs_bert.py)
        help="N-gram range for TF-IDF as string tuple (ignored if use_bert=True).",
    )
    parser.add_argument(
        "--bert_batch_size", type=int, default=32, help="Batch size for BERT inference."
    )
    args = parser.parse_args()

    # --- Parameter Parsing ---
    # Replaced manual ast.literal_eval with the helper function
    ngram_range = parse_dvc_param(
        args.ngram_range, name="ngram_range", expected_type=tuple
    )

    if ngram_range is None:
        # The helper function logs the error and returns None if parsing fails
        return

    logger.info("--- Feature Engineering Parameters ---")
    engineer_features(
        use_bert=args.use_bert,
        max_features=args.max_features,
        ngram_range=ngram_range,
        bert_batch_size=args.bert_batch_size,
    )


if __name__ == "__main__":
    main()
