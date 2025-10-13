"""
Experiment with imbalance handling techniques for sentiment classification.

Applies techniques (SMOTE, ADASYN, etc.) to TF-IDF features on the train set only, trains RandomForest,
logs to MLflow, and saves artifacts/models.

Usage:
    uv run python -m src.features.imbalance_tuning --imbalance_methods "['class_weights','oversampling']" --max_features 1000

Requirements:
    - Processed data in data/processed/.
    - uv sync (for imblearn, scikit-learn, mlflow, seaborn).
    - MLflow server running (e.g., uv run mlflow server --host 127.0.0.1 --port 5000).

Design Considerations:
- Reliability: Train-only resampling; validation on untouched test.
- Scalability: Sparse matrices; efficient resampling.
- Maintainability: Logging, type hints, relative paths.
- Adaptability: Parameterized methods; extensible to other classifiers.
"""

import argparse
import ast  # For safe list parsing from params
from typing import Tuple
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pickle

# --- Project Utilities ---
from src.utils.paths import TRAIN_PATH, TEST_PATH, FIGURES_DIR, MODELS_DIR, PROJECT_ROOT
from src.utils.mlflow_config import get_mlflow_uri
from src.utils.logger import get_logger

# --- Logging Setup ---
logger = get_logger(__name__, headline="imbalance_tuning.py")

# --- Path setup ---
IMBALANCE_MODELS_DIR = MODELS_DIR / "features" / "imbalance_methods"
IMBALANCE_FIGURES_DIR = FIGURES_DIR / "imbalance_methods"
IMBALANCE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
IMBALANCE_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --- MLflow setup via utility function (for consistency) ---
mlflow_uri = get_mlflow_uri()
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("Exp - Imbalance Handling")


def run_imbalanced_experiment(
    imbalance_method: str,
    ngram_range: Tuple[int, int],
    max_features: int,
    n_estimators: int,
    max_depth: int,
) -> None:
    """
    Run experiment for specified imbalance handling method.

    Args:
        imbalance_method: Technique name ('class_weights', 'oversampling', 'adasyn', 'undersampling', 'smote_enn').
        ngram_range: N-gram range tuple.
        max_features: Maximum TF-IDF features.
        n_estimators: RF trees.
        max_depth: RF depth.
    """

    # --- Load data ---
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Processed data missing. Run data_preparation first.")

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    X_train_text = train_df["clean_comment"].tolist()
    y_train = train_df["category_encoded"].values
    X_test_text = test_df["clean_comment"].tolist()
    y_test = test_df["category_encoded"].values
    logger.info(
        f"Loaded train: {len(X_train_text)} samples, test: {len(X_test_text)} samples."
    )

    # --- Vectorization using TF-IDF (Fit on training, transform both) ---
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words="english",
        # tokenizer=lambda x: x.split(),  # Pre-cleaned tokens
        lowercase=False,  # Already lowercased
        min_df=2,
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)
    feature_dim = X_train_vec.shape[1]

    # --- Handle class imbalance (train set only) ---
    class_weight = None

    if imbalance_method == "class_weights":
        class_weight = "balanced"
        logger.info("Using 'balanced' class weights. No resampling applied.")
    else:
        sampler = None
        if imbalance_method == "oversampling":
            sampler = SMOTE(random_state=42)
        elif imbalance_method == "adasyn":
            sampler = ADASYN(random_state=42)
        elif imbalance_method == "undersampling":
            sampler = RandomUnderSampler(random_state=42)
        elif imbalance_method == "smote_enn":
            sampler = SMOTEENN(random_state=42)
        else:
            raise ValueError(f"Unsupported imbalance method: {imbalance_method}")

        if sampler:
            logger.info(f"Applying {imbalance_method.upper()} to training data...")
            X_train_vec, y_train = sampler.fit_resample(X_train_vec, y_train)
            logger.info(
                f"New training sample size: {X_train_vec.shape[0]} ({np.bincount(y_train)})"
            )  # np.bincount for sanity check: count of samples per class after resampling

    # --- MLflow Tracking ---
    with mlflow.start_run() as run:
        run_name = f"Imbalance_RF_{imbalance_method}_Feat_{feature_dim}"
        logger.info(f"🚀 Running experiment: {run_name}")
        # Tags
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.set_tag("experiment_type", "imbalance_tuning")
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag(
            "description",
            f"RF with TF-IDF, method={imbalance_method}, focus on positive F1",
        )

        # Params
        mlflow.log_param("vectorizer_type", "TF-IDF")
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("feature_dim", feature_dim)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("imbalance_method", imbalance_method)
        mlflow.log_param("class_weight", class_weight)

        # --- Model Training ---
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight=class_weight,
            n_jobs=-1,
        )
        model.fit(X_train_vec, y_train)

        # Predict and Metrics
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        logger.info(f"Model Accuracy: {accuracy:.4f}")

        # Classification report (log key metrics, focusing on class 1)
        report = classification_report(y_test, y_pred, output_dict=True)
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                mlflow.log_metric(f"{label}_f1-score", metrics.get("f1-score", 0))
                mlflow.log_metric(f"{label}_precision", metrics.get("precision", 0))
                mlflow.log_metric(f"{label}_recall", metrics.get("recall", 0))

        # Log key class 1 metrics to console
        logger.info(f"Recall: {report.get('1', {}).get('recall'):.4f}")
        logger.info(f"Precision: {report.get('1', {}).get('precision'):.4f}")
        logger.info(f"F1-score: {report.get('1', {}).get('f1-score'):.4f}")

        # Confusion matrix (artifact)
        conf_matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix: {run_name}")
        plot_path = IMBALANCE_FIGURES_DIR / f"confusion_matrix_{run_name}.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(plot_path))

        # Log model (for later deployment/registration)
        model_path = f"random_forest_model_imbalanced_{imbalance_method}"
        mlflow.sklearn.log_model(model, model_path)

        # Save the vectorizer and model locally for this specific method
        # model_path = IMBALANCE_MODELS_DIR / f"rf_model_{imbalance_method}.pkl"
        # with open(model_path, "wb") as f:
        #     pickle.dump(model, f)

        vectorizer_path = (
            IMBALANCE_MODELS_DIR / f"tfidf_vectorizer_{imbalance_method}.pkl"
        )
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)

        logger.info(
            f"✅ Experiment finished: {run_name} | MLflow Run ID: {run.info.run_id} | Artifacts saved locally to {IMBALANCE_MODELS_DIR.relative_to(PROJECT_ROOT)}"
        )


def main() -> None:
    """Parse args and run experiments."""
    parser = argparse.ArgumentParser(
        description="Handle class imbalance with MLflow tracking."
    )
    parser.add_argument(
        "--imbalance_methods",
        type=str,
        default="['class_weights','oversampling','adasyn','undersampling','smote_enn']",
        help='List of imbalance methods to test (e.g., \'["smote", "weights"]\').',
    )
    parser.add_argument(
        "--ngram_range", type=str, default="(1,1)", help="N-gram range as string tuple."
    )
    parser.add_argument(
        "--max_features", type=int, default=1000, help="Max TF-IDF features."
    )
    parser.add_argument(
        "--n_estimators", type=int, default=200, help="RF n_estimators."
    )
    parser.add_argument("--max_depth", type=int, default=15, help="RF max_depth.")
    args = parser.parse_args()

    # Parse lists/tuples
    imbalance_methods = ast.literal_eval(args.imbalance_methods.strip())
    ngram_range = ast.literal_eval(args.ngram_range.strip())
    if not isinstance(imbalance_methods, list) or not isinstance(ngram_range, tuple):
        raise ValueError(
            "imbalance_methods must be a list of strings and ngram_range must be a tuple of ints."
        )

    logger.info(
        f"--- Running imbalance experiments for methods: {imbalance_methods} ---"
    )
    for method in imbalance_methods:
        run_imbalanced_experiment(
            imbalance_method=method,
            ngram_range=ngram_range,
            max_features=args.max_features,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )
    logger.info(
        "--- Imbalance handling tuning complete. Analyze results in MLflow UI ---"
    )


if __name__ == "__main__":
    main()
