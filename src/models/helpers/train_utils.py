"""
Utility functions for standardized local artifact management (DVC tracking)
and model persistence.
"""

import pickle
import json
from datetime import datetime

# --- Project Utilities ---
from src.utils.logger import get_logger
from src.utils.paths import ADVANCED_DIR, EVAL_DIR, PROJECT_ROOT, BASELINE_MODEL_DIR

# --- Logging Setup ---
logger = get_logger(__name__, headline="train_utils.py")


# ---------------------------------------------------------------------
# 1. Local Model Saving
# ---------------------------------------------------------------------
def save_hyperparams_bundle(model_name: str, params: dict, score: float):
    """Save the best hyperparameters and their corresponding score locally."""
    # Use ADVANCED_DIR (models/advanced/) as the DVC-tracked path
    filepath = ADVANCED_DIR / f"{model_name}_best_hyperparams.pkl"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        # Saving a dictionary containing params and score
        pickle.dump({"params": params, "score": score}, f)
    logger.info(
        f"Best parameters saved for {model_name} → {filepath.relative_to(PROJECT_ROOT)}"
    )


def save_model_object(model, model_name: str):
    """Save the final trained model object for DVC tracking."""
    filepath = ADVANCED_DIR / f"{model_name}_model.pkl"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    logger.info(
        f"Final model object saved for {model_name} → {filepath.relative_to(PROJECT_ROOT)}"
    )


def save_model_bundle(model_bundle: dict, save_path):
    """Save a model and encoder bundle (e.g., for baselines) for DVC tracking."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model_bundle, f)
    logger.info(f"Model bundle saved → {save_path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------
# 2. Metric Saving (JSON for DVC)
# ---------------------------------------------------------------------
def save_metrics_json(model_name: str, score: float):
    """
    Save the primary validation metric score to a JSON file for DVC metrics tracking.
    (Used by hyperparameter_tuning/bert_training)
    """
    filepath = ADVANCED_DIR / f"{model_name}_metrics.json"

    # DVC expects a simple JSON structure with the metric name and value
    metrics_data = {"val_macro_f1": score}

    with open(filepath, "w") as f:
        json.dump(metrics_data, f, indent=4)

    logger.info(
        f"Validation metrics saved for DVC tracking → {filepath.relative_to(PROJECT_ROOT)}"
    )


def save_test_metrics_json(model_name: str, report: dict):
    """
    Save key test set metrics from the classification report to a JSON file
    for DVC tracking and comparison. (Used by model_evaluation)
    """
    # Use EVAL_DIR (models/advanced/evaluation/) to distinguish from training metrics
    filepath = EVAL_DIR / f"{model_name}_test_metrics.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)

    test_metrics = {
        "test_macro_f1": report["macro avg"]["f1-score"],
        "test_weighted_f1": report["weighted avg"]["f1-score"],
    }

    with open(filepath, "w") as f:
        json.dump(test_metrics, f, indent=4)

    logger.info(
        f"Test metrics saved for DVC tracking → {filepath.relative_to(PROJECT_ROOT)}"
    )


def save_baseline_metrics_json(score: float):
    """
    Saves the baseline model's primary metric to a JSON file.
    """
    filepath = BASELINE_MODEL_DIR / "baseline_metrics.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)

    metrics_data = {"val_macro_f1": score}

    with open(filepath, "w") as f:
        json.dump(metrics_data, f, indent=4)

    logger.info(
        f"Baseline metrics saved for DVC tracking → {filepath.relative_to(PROJECT_ROOT)}"
    )


# ---------------------------------------------------------------------
# 3. Champion Model Tracking (for Model Registration)
# ---------------------------------------------------------------------
def save_best_model_run_info(run_id: str, model_name: str):
    """
    Saves the MLflow Run ID and model name of the champion model
    to a JSON file for the 'register_model' stage.
    """
    filepath = EVAL_DIR / "best_model_run_info.json"
    filepath.parent.mkdir(parents=True, exist_ok=True)

    info = {
        "model_name": model_name,
        "run_id": run_id,
        "metric_used_for_selection": "test_macro_auc",
        "timestamp": datetime.now().isoformat(),
    }

    with open(filepath, "w") as f:
        json.dump(info, f, indent=4)
    logger.info(f"Champion Run Info saved → {filepath.relative_to(PROJECT_ROOT)}")
