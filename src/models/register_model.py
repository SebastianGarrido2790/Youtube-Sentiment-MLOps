"""
Automated Model Registration Script.

This module:
- Reads evaluation metrics from the latest test results.
- Selects the best-performing model (if multiple are available).
- Registers the model in MLflow Model Registry.
- Handles both legacy and modern MLflow versions (pre/post 2.9.0).
- Supports fallback for local MLflow setups (no registry backend).

NOTE: The local MLflow tracking server doesn’t fully support stage transitions when running without a backend database (like SQLite/MySQL/PostgreSQL).
If you’re using the default file-based backend, this endpoint may return HTTP 500.

Dependencies:
    - MLflow tracking server running and accessible.
    - From model evaluation stage:
      models/advanced/evalution
        ├── lightgbm_evaluation_run.json
        └── lightgbm_test_metrics.json

Usage:
    uv run python -m src.models.register_model
"""

import json
from packaging import version
import mlflow
from mlflow.tracking import MlflowClient

from src.utils.paths import EVAL_DIR
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri

logger = get_logger(__name__, headline="register_model.py")


# =====================================================================
# Load Best Model from Evaluation
# =====================================================================


def load_best_model_info():
    """
    Loads the best evaluated model and its metrics from EVAL_DIR.

    Expected files:
        - <model>_test_metrics.json
        - <model>_evaluation_run.json

    Returns:
        tuple: (model_name, run_id, f1_score)
    """
    try:
        metric_files = list(EVAL_DIR.glob("*_test_metrics.json"))
        if not metric_files:
            raise FileNotFoundError("No test metric files found in EVAL_DIR.")

        best_model, best_f1 = None, -1.0
        for metric_file in metric_files:
            model_name = metric_file.stem.replace("_test_metrics", "")
            with open(metric_file, "r") as f:
                metrics = json.load(f)
                f1 = metrics.get("test_macro_f1", 0)
                logger.info(f"Detected {model_name} → F1={f1:.4f}")
                if f1 > best_f1:
                    best_model, best_f1 = model_name, f1

        if best_model is None:
            raise RuntimeError("No valid model metrics found for registration.")

        # Load run ID for the best model
        run_info_path = EVAL_DIR / f"{best_model}_evaluation_run.json"
        if not run_info_path.exists():
            raise FileNotFoundError(
                f"Missing run info for {best_model} → {run_info_path}"
            )

        with open(run_info_path, "r") as f:
            run_data = json.load(f)
            run_id = run_data["evaluation_run_id"]

        logger.info(f"🏆 Best model selected → {best_model.upper()} (F1={best_f1:.4f})")
        return best_model, run_id, best_f1

    except Exception as e:
        logger.error(f"Error loading best model info: {e}")
        raise


# =====================================================================
# Register Best Model
# =====================================================================


def register_best_model(
    model_name: str, run_id: str, f1: float, f1_threshold: float = 0.75
):
    """
    Register the best-performing model in the MLflow Model Registry.

    Handles version-aware MLflow behavior:
    - For MLflow < 2.9.0 → uses stage transitions (Production)
    - For MLflow ≥ 2.9.0 → uses tag-based stage management
    - Ensures graceful fallback when registry APIs are unavailable (e.g., local file store)
    """
    client = MlflowClient()
    logger.info("🚀 Starting model registration process...")

    if f1 < f1_threshold:
        logger.warning(
            f"❌ Model performance below threshold (F1={f1:.4f} < {f1_threshold:.2f}). Skipping registration."
        )
        return

    # --- Register model from run ---
    try:
        logger.info(f"Registering model from run {run_id}")
        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=f"youtube_sentiment_{model_name}",
        )
        logger.info(
            f"✅ Successfully registered model '{model_name}' version {model_version.version}"
        )

    except Exception as e:
        logger.error(f"❌ Failed to register model '{model_name}': {e}")
        raise

    # --- Handle MLflow version differences ---
    mlflow_version = version.parse(mlflow.__version__)
    supports_stages = mlflow_version < version.parse("2.9.0")

    try:
        if supports_stages:
            # ✅ Legacy MLflow (pre-2.9.0): use stage transitions
            client.transition_model_version_stage(
                name=f"youtube_sentiment_{model_name}",
                version=model_version.version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info(
                f"🟢 Model version {model_version.version} transitioned to 'Production' stage."
            )
        else:
            # ✅ Modern MLflow (2.9.0+): use tag-based deployment stage
            client.set_model_version_tag(
                name=f"youtube_sentiment_{model_name}",
                version=model_version.version,
                key="deployment_stage",
                value="Production",
            )
            logger.info(
                f"🟢 MLflow ≥ 2.9 detected — set tag 'deployment_stage=Production' "
                f"for model version {model_version.version}."
            )

    except Exception as e:
        logger.warning(
            f"⚠️ Stage/Tag assignment failed for model '{model_name}' (v{model_version.version}): {e}"
        )
        logger.info(
            "Continuing without stage assignment (likely unsupported in local MLflow file store)."
        )

    # --- Log registration summary ---
    logger.info(
        f"🧾 Registration Summary → "
        f"Model: youtube_sentiment_{model_name} | "
        f"Version: {model_version.version} | "
        f"F1: {f1:.4f}"
    )

    logger.info("🏁 Model registration workflow completed successfully.")


# =====================================================================
#  Main Execution
# =====================================================================


def main():
    logger.info("🚀 Starting automated model registration workflow...")

    try:
        # --- 1. Load best model info from evaluation directory ---
        model_name, run_id, f1 = load_best_model_info()
        logger.info(f"Detected model: {model_name} | F1={f1:.4f}")

        # --- 2. Connect to MLflow Tracking Server ---
        mlflow_uri = get_mlflow_uri()
        mlflow.set_tracking_uri(mlflow_uri)
        logger.info(f"Using MLflow Tracking URI: {mlflow_uri}")

        # --- 3. Register the best model ---
        register_best_model(model_name, run_id, f1, f1_threshold=0.75)

    except Exception as e:
        logger.error(f"❌ Model registration failed: {e}")
        raise


if __name__ == "__main__":
    main()
