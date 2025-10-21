"""
Model Evaluation Script for Selected Best Model (LightGBM).

Evaluates the best LightGBM model on the test set, logs metrics to MLflow,
saves artifacts (e.g., confusion matrix, JSON metrics), and infers model signature.

Usage:
    uv run python -m src.models.model_evaluation

Design Considerations:
- Reliability: Uses pre-loaded features/labels; validates inputs.
- Scalability: Sparse matrix support; batched predictions if needed.
- Maintainability: Leverages shared helpers (data_loader, train_utils); centralized logging/MLflow.
- Adaptability: Parameterized model selection; extensible to other models.
"""

import numpy as np
import pickle
import json
import mlflow
import mlflow.lightgbm  # Use LightGBM-specific logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# --- Project Utilities ---
from src.utils.paths import ADVANCED_DIR, EVAL_DIR, PROJECT_ROOT
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri
from src.models.helpers.data_loader import load_feature_data
from src.models.helpers.train_utils import setup_experiment, log_metrics_to_mlflow

# --- Configuration ---
MODEL_NAME = "lightgbm"
EXPERIMENT_NAME = "Model Evaluation - LightGBM Test Set"

# --- Logging Setup (using centralized utility) ---
logger = get_logger(__name__, headline="model_evaluation.py")


# =====================================================================
#  Core Helper Functions
# =====================================================================


def load_best_model_artifact(model_name: str):
    """Load the final trained model object from the local DVC-tracked artifact path."""
    # Assuming lightgbm_training.py saves the model object locally for DVC tracking
    model_path = ADVANCED_DIR / f"{model_name}_model.pkl"
    if not model_path.exists():
        logger.error(
            f"Model file not found at {model_path}. Please check the training stage outputs."
        )
        raise FileNotFoundError(f"Required model artifact {model_path} is missing.")
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        logger.info(f"Best model loaded from {model_path.relative_to(PROJECT_ROOT)}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and log classification metrics and confusion matrix."""
    try:
        # Predict on test data
        y_pred = model.predict(X_test)

        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        logger.info("Model evaluation completed on test set.")
        return report, cm
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


def log_confusion_matrix_as_artifact(cm: np.ndarray, model_name: str, labels: list):
    """Generate, save, and log confusion matrix plot to MLflow."""
    file_path = EVAL_DIR / f"{model_name}_confusion_matrix.png"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title(f"Confusion Matrix - {model_name} Test Data")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    plt.savefig(file_path)
    plt.close()

    mlflow.log_artifact(str(file_path))
    logger.info(
        f"Confusion Matrix saved to {file_path.relative_to(PROJECT_ROOT)} and logged to MLflow."
    )


def save_test_metrics_json(model_name: str, report: dict):
    """Save the primary test metric (macro F1) to a JSON file for DVC metrics tracking."""
    filepath = EVAL_DIR / f"{model_name}_test_metrics.json"

    macro_f1 = report["macro avg"]["f1-score"]

    # DVC expects a simple JSON structure with the metric name and value
    metrics_data = {"test_macro_f1": macro_f1}

    with open(filepath, "w") as f:
        json.dump(metrics_data, f, indent=4)

    logger.info(
        f"Test metrics saved for DVC tracking → {filepath.relative_to(PROJECT_ROOT)}"
    )


def save_best_model_run_info(run_id: str, model_name: str):
    """Save the evaluation run ID and model name to a JSON file for the next stage (registration)."""
    filepath = EVAL_DIR / f"{model_name}_evaluation_run.json"
    model_info = {"evaluation_run_id": run_id, "model_name": model_name}
    with open(filepath, "w") as file:
        json.dump(model_info, file, indent=4)
    logger.info(f"Evaluation run info saved to {filepath.relative_to(PROJECT_ROOT)}")


# =====================================================================
#  Main Execution
# =====================================================================


def main():
    # --- Setup MLflow ---
    mlflow_uri = get_mlflow_uri()
    setup_experiment(EXPERIMENT_NAME, mlflow_uri)

    with mlflow.start_run() as run:
        logger.info(f"🚀 Starting model evaluation for {MODEL_NAME} on Test Set...")

        try:
            # --- 1. Load Data, Model, and Label Encoder ---
            # load_feature_data returns: X_train, X_val, X_test, y_train, y_val, y_test, le
            _, _, X_test, _, _, y_test, le = load_feature_data(validate_files=True)

            # Load the best model artifact
            model = load_best_model_artifact(MODEL_NAME)

            # --- 2. Evaluate Model ---
            report, cm = evaluate_model(model, X_test, y_test)

            # --- 3. Log Metrics (MLflow & DVC) ---

            # Flatten classification report for MLflow logging
            flat_metrics = {}
            for label, metrics in report.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if metric_name not in ("support"):
                            # Rename f1-score to f1 for consistency
                            key = metric_name.replace("-score", "_f1")
                            flat_metrics[f"test_{label}_{key}"] = value

            log_metrics_to_mlflow(flat_metrics)

            # Save key metric to local JSON for DVC tracking
            save_test_metrics_json(MODEL_NAME, report)

            # Log confusion matrix artifact
            labels = (
                le.classes_.tolist()
            )  # Get human-readable labels: Negative, Neutral, Positive
            log_confusion_matrix_as_artifact(cm, MODEL_NAME, labels)

            # --- 4. Log Model Artifact to MLflow Run (for registration readiness) ---
            mlflow.lightgbm.log_model(
                lgb_model=model,
                artifact_path="model",
                registered_model_name=None,  # Defer registration to the next stage
            )
            logger.info(f"Model artifact logged to MLflow run: {run.info.run_id}")

            # --- 5. Save Run Info for Registration Stage ---
            save_best_model_run_info(run.info.run_id, MODEL_NAME)

            # --- 6. Set Final Tags ---
            mlflow.set_tag("model_type", MODEL_NAME)
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset", "YouTube Comments (Reddit Proxy)")

            logger.info(
                f"🏁 Model evaluation complete. Test Macro F1: {report['macro avg']['f1-score']:.4f}"
            )

        except Exception as e:
            logger.error(f"Failed to complete model evaluation: {e}")
            raise


if __name__ == "__main__":
    main()
