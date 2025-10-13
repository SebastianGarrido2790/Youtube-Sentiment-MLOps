import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import save_npz, hstack
from scipy import sparse
from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModel
import pickle

# Import clean_text (reuse from src/data/make_dataset.py)
from ..data.make_dataset import clean_text


def get_bert_embeddings(
    texts: list, device: str = None, batch_size: int = 32
) -> np.ndarray:
    """
    Generate mean-pooled BERT embeddings for a list of texts.

    Args:
        texts: List of cleaned texts.
        device: 'cuda' or 'cpu'.
        batch_size: Batch size for inference.

    Returns:
        np.ndarray of shape (n_samples, 768).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
            # Mean pool over tokens (exclude CLS/SEP)
            pooled = outputs.last_hidden_state[:, 1:-1, :].mean(dim=1)
            embeddings.append(pooled.cpu().numpy())

    return np.vstack(embeddings)


def engineer_features(
    use_bert: bool = False,
    max_features: int = 7000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> None:
    """
    Engineer features from prepared datasets, with optional BERT swap.

    Args:
        use_bert: If True, use BERT embeddings instead of TF-IDF.
        max_features: Max vocabulary for TF-IDF (ignored if use_bert=True).
        ngram_range: N-gram range for TF-IDF (ignored if use_bert=True).

    Saves:
        X_train.npz, y_train.npy, etc., in models/features/.
    """
    splits = ["train", "val", "test"]
    os.makedirs("models/features", exist_ok=True)

    # Load splits
    dfs = {}
    for split in splits:
        path = f"data/processed/{split}.parquet"
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run make_dataset.py first.")
        dfs[split] = pd.read_parquet(path)

    # Re-clean text (ensure consistency)
    for split in splits:
        dfs[split]["clean_comment"] = dfs[split]["clean_comment"].apply(clean_text)

    # Encode labels (numeric for modeling)
    le = LabelEncoder()
    y_all = pd.concat([dfs[s]["category"] for s in splits])  # Use numeric category
    le.fit(y_all)

    # Extract texts
    train_texts = dfs["train"]["clean_comment"].tolist()
    val_texts = dfs["val"]["clean_comment"].tolist()
    test_texts = dfs["test"]["clean_comment"].tolist()

    if use_bert:
        # BERT embeddings
        print("Generating BERT embeddings...")
        X_train_tfidf = get_bert_embeddings(train_texts)  # Shape: (n, 768)
        X_val_tfidf = get_bert_embeddings(val_texts)
        X_test_tfidf = get_bert_embeddings(test_texts)
        # Save tokenizer/model paths for inference
        tokenizer_path = "models/features/tokenizer.pkl"
        model_path = (
            "models/features/bert_model"  # Use torch.save for full model if needed
        )
        with open(tokenizer_path, "wb") as f:
            pickle.dump(AutoTokenizer.from_pretrained("bert-base-uncased"), f)
        torch.save(
            AutoModel.from_pretrained("bert-base-uncased").state_dict(),
            model_path + ".pth",
        )
    else:
        # TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            lowercase=True,
            min_df=2,
        )
        X_train_tfidf = vectorizer.fit_transform(train_texts)
        X_val_tfidf = vectorizer.transform(val_texts)
        X_test_tfidf = vectorizer.transform(test_texts)
        # Save vectorizer
        with open("models/features/vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)

    # Derived features: length-based (dense, for all splits)
    def add_derived_features(df: pd.DataFrame) -> np.ndarray:
        df_local = df.copy()
        df_local["char_len"] = df_local["clean_comment"].str.len()
        df_local["word_len"] = df_local["clean_comment"].str.split().str.len()
        # Simple positive/negative word ratios (expandable with VADER lexicon)
        pos_words = {"good", "great", "love", "like", "positive", "best"}
        neg_words = {"bad", "hate", "worst", "negative", "shit", "fuck"}
        df_local["pos_ratio"] = df_local["clean_comment"].apply(
            lambda x: len([w for w in x.split() if w in pos_words])
            / max(len(x.split()), 1)
        )
        df_local["neg_ratio"] = df_local["clean_comment"].apply(
            lambda x: len([w for w in x.split() if w in neg_words])
            / max(len(x.split()), 1)
        )
        return df_local[["char_len", "word_len", "pos_ratio", "neg_ratio"]].values

    derived_train = add_derived_features(dfs["train"])
    derived_val = add_derived_features(dfs["val"])
    derived_test = add_derived_features(dfs["test"])

    # Combine: Text features + dense derived
    if use_bert:
        # Dense + dense
        X_train = np.hstack([X_train_tfidf, derived_train])
        X_val = np.hstack([X_val_tfidf, derived_val])
        X_test = np.hstack([X_test_tfidf, derived_test])
    else:
        # Sparse + dense
        X_train = hstack([X_train_tfidf, derived_train])
        X_val = hstack([X_val_tfidf, derived_val])
        X_test = hstack([X_test_tfidf, derived_test])

    # Labels (encoded)
    y_train = le.transform(dfs["train"]["category"])
    y_val = le.transform(dfs["val"]["category"])
    y_test = le.transform(dfs["test"]["category"])

    # Save features (convert to sparse if dense for consistency)
    if use_bert and not sparse.issparse(X_train):
        X_train = sparse.csr_matrix(X_train)
        X_val = sparse.csr_matrix(X_val)
        X_test = sparse.csr_matrix(X_test)

    save_paths = {
        "X": ["X_train.npz", "X_val.npz", "X_test.npz"],
        "y": ["y_train.npy", "y_val.npy", "y_test.npy"],
    }
    for split, x_path in zip(splits, save_paths["X"]):
        save_npz(f"models/features/{x_path}", locals()[f"X_{split}"])
    for split, y_path in zip(splits, save_paths["y"]):
        np.save(f"models/features/{y_path}", locals()[f"y_{split}"])

    # Save encoder
    with open("models/features/label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    feature_type = "BERT" if use_bert else f"TF-IDF (max_features={max_features})"
    print(
        f"Features engineered ({feature_type}): Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}"
    )


if __name__ == "__main__":
    # Default: TF-IDF
    engineer_features()
    # BERT: engineer_features(use_bert=True)
