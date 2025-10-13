"""
Tune TF-IDF max_features for sentiment feature engineering (unigrams).

Loads processed data, varies max_features, trains RandomForest baselines, logs to MLflow,
and saves visualiartifactszations/models.

Usage:
    uv run python -m src.features.tfidf_max_features --max_features_values '[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]'

Requirements:
    - Processed data in data/processed/.
    - uv sync (for scikit-learn, mlflow, seaborn).
    - MLflow server running (e.g., uv run mlflow server --host 127.0.0.1 --port 5000).

Design Considerations:
- Reliability: Input validation, consistent splits.
- Scalability: Sparse TF-IDF matrices.
- Maintainability: Logging, type hints, relative paths.
- Adaptability: Parameterized via args/params.yaml; extensible to other n-grams.
"""

import argparse
import ast  # For safe list/tuple parsing from params
from typing import Tuple
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Project Utilities ---
from src.utils.mlflow_config import get_mlflow_uri
from src.utils.logger import get_logger
from src.utils.paths import FIGURES_DIR, MODELS_DIR, TRAIN_PATH, TEST_PATH, PROJECT_ROOT

# --- Logging Setup ---
logger = get_logger(__name__, headline="tfidf_max_features.py")

# --- Path setup ---
TFIDF_MODELS_DIR = MODELS_DIR / "features" / "tfidf_max_features"
TFIDF_FIGURES_DIR = FIGURES_DIR / "tfidf_max_features"
TFIDF_MODELS_DIR.mkdir(parents=True, exist_ok=True)
TFIDF_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --- MLflow setup via utility function (for consistency) ---
mlflow_uri = get_mlflow_uri()
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("Exp - TFIDF Max Features")


def run_max_features_experiment(
    max_features: int,
    ngram_range: Tuple[int, int],
    n_estimators: int,
    max_depth: int,
) -> None:
    """
    Run experiment for TF-IDF with specified max_features.

    Args:
        max_features: Maximum number of features for TF-IDF.
        ngram_range: N-gram range tuple.
        n_estimators: RF trees.
        max_depth: RF depth.
    """

    # --- Load data ---
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Processed data missing. Run data_preparation first.")

    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    X_train_text = train_df["clean_comment"].tolist()
    y_train = train_df["category"].values
    X_test_text = test_df["clean_comment"].tolist()
    y_test = test_df["category"].values
    logger.info(
        f"Loaded train: {len(X_train_text)} samples, test: {len(X_test_text)} samples."
    )

    # --- Vectorization using TF-IDF ---
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        stop_words="english",
        # tokenizer=lambda x: x.split(),  # Use pre-cleaned tokens
        lowercase=False,  # Already lowercased
        min_df=2,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    feature_dim = X_train.shape[1]

    # --- MLflow Tracking ---
    with mlflow.start_run() as run:
        run_name = f"TFIDF_max_features_{max_features}"
        logger.info(f"🚀 Running experiment: {run_name}")
        # Tags
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.set_tag("experiment_type", "feature_comparison")
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.set_tag(
            "description",
            f"RF with TF-IDF, max_features={max_features}, ngram={ngram_range}",
        )

        # Params
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("ngram_range", ngram_range)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("feature_dim", feature_dim)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # --- Model Training ---
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
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
        plot_path = TFIDF_FIGURES_DIR / f"confusion_matrix_{run_name}.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(plot_path))

        # Log model (for later deployment/registration)
        model_path = f"random_forest_model_tfidf_max_{max_features}"
        mlflow.sklearn.log_model(model, model_path)

        # Save the vectorizer and model locally for this specific method
        # model_path = TFIDF_MODELS_DIR / f"rftfidf_max_{max_features}.pkl"
        # with open(model_path, "wb") as f:
        #     pickle.dump(model, f)

        vectorizer_path = (
            TFIDF_MODELS_DIR / f"tfidf_vectorizer_max_features_{max_features}.pkl"
        )
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)

        logger.info(
            f"✅ Experiment finished: {run_name} | MLflow Run ID: {run.info.run_id} | Artifacts saved locally to {TFIDF_MODELS_DIR.relative_to(PROJECT_ROOT)}"
        )


def main() -> None:
    """Parse args and run experiments."""
    parser = argparse.ArgumentParser(
        description="Tune TF-IDF max_features using RandomForest baseline with MLflow tracking."
    )
    parser.add_argument(
        "--max_features_values",
        type=str,
        default="[1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]",
        help="Max features values as string list.",
    )
    parser.add_argument(
        "--ngram_range", type=str, default="(1,1)", help="N-gram range as string tuple."
    )
    parser.add_argument(
        "--n_estimators", type=int, default=200, help="RF n_estimators."
    )
    parser.add_argument("--max_depth", type=int, default=15, help="RF max_depth.")
    args = parser.parse_args()

    # Parse lists/tuples
    max_features_values = ast.literal_eval(args.max_features_values.strip())
    ngram_range = ast.literal_eval(args.ngram_range.strip())

    if not isinstance(max_features_values, list) or not isinstance(ngram_range, tuple):
        raise ValueError(
            "max_features_list must be a list of ints and ngram_range_str must be a tuple of ints."
        )

    logger.info(
        f"--- Running TF-IDF tuning for max_features: {max_features_values} ---"
    )

    for max_features in max_features_values:
        run_max_features_experiment(
            max_features=max_features,
            ngram_range=ngram_range,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
        )

    logger.info("--- Max features tuning complete. Analyze results in MLflow UI ---")


if __name__ == "__main__":
    main()
