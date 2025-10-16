"""
Utility functions for standardized training and experiment management.
Used across baseline and advanced models.
"""

import pickle
import mlflow
import json
from datetime import datetime
from src.utils.logger import get_logger
from src.utils.paths import ADVANCED_DIR, PROJECT_ROOT

logger = get_logger(__name__, headline="train_utils.py")


# ---------------------------------------------------------------------
# 1. MLflow Setup
# ---------------------------------------------------------------------
def setup_experiment(experiment_name: str, mlflow_uri: str):
    """Initialize MLflow tracking with URI and experiment name.

    Args:
        experiment_name (str): Name of the MLflow experiment.
        mlflow_uri (str): Tracking URI for MLflow.

    Features:
        - Appends timestamp (YYYY-MM-DD) for uniqueness.
        - Auto-creates experiment if deleted or missing.
    """
    mlflow.set_tracking_uri(mlflow_uri)

    # Timestamp for uniqueness
    timestamp = datetime.now().strftime("%Y-%m-%d")
    full_name = f"{experiment_name} - {timestamp}"

    try:
        mlflow.set_experiment(full_name)
    except mlflow.exceptions.MlflowException as e:
        if "deleted" in str(e).lower() or "not found" in str(e).lower():
            logger.info(
                f"Experiment '{full_name}' not found/deleted. Creating new one."
            )
            mlflow.create_experiment(full_name)
            mlflow.set_experiment(full_name)
        else:
            raise  # Re-raise non-resolvable errors

    logger.info(f"MLflow experiment initialized → {full_name} | URI: {mlflow_uri}")


# ---------------------------------------------------------------------
# 2. Metric Logging
# ---------------------------------------------------------------------
def log_metrics_to_mlflow(metrics: dict):
    """Log a dictionary of metrics to MLflow."""
    for key, value in metrics.items():
        mlflow.log_metric(key, value)
    logger.info(f"Logged {len(metrics)} metrics to MLflow.")


# ---------------------------------------------------------------------
# 3. Model Bundle Saving
# ---------------------------------------------------------------------
def save_model_bundle(model_bundle: dict, save_path):
    """
    Save model and encoder bundle for DVC tracking and reproducibility.

    Args:
        model_bundle (dict): Contains model and encoder.
        save_path (Path): Destination file path.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model_bundle, f)
    logger.info(f"Model bundle saved to: {save_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------
# 4. Parameter Saving (Optuna or manual tuning)
# ---------------------------------------------------------------------
def save_best_params(model_name: str, params: dict, score: float):
    """Save best parameters and score to pickle file."""
    filepath = ADVANCED_DIR / f"{model_name}_best.pkl"
    with open(filepath, "wb") as f:
        pickle.dump({"params": params, "score": score}, f)
    logger.info(
        f"Best parameters saved for {model_name} → {filepath.relative_to(PROJECT_ROOT)}"
    )


# ---------------------------------------------------------------------
# 5. Metric Saving (JSON for DVC)
# ---------------------------------------------------------------------
def save_metrics_json(model_name: str, score: float):
    """
    Save the primary metric score to a JSON file for DVC metrics tracking.
    """
    filepath = ADVANCED_DIR / f"{model_name}_metrics.json"

    # DVC expects a simple JSON structure with the metric name and value
    metrics_data = {"val_macro_f1": score}

    with open(filepath, "w") as f:
        json.dump(metrics_data, f, indent=4)

    logger.info(
        f"Metrics saved for DVC tracking → {filepath.relative_to(PROJECT_ROOT)}"
    )
