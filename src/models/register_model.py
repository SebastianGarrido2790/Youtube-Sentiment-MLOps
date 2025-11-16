"""
Automated Model Registration Script.

This module reads the champion model information (selected during the
'model_evaluation' stage) from 'best_model_run_info.json'.

It then:
1.  Loads the champion's 'run_id' and 'model_name'.
2.  Loads the champion's corresponding test metrics (e.g., 'lightgbm_test_metrics.json').
3.  Loads the 'f1_threshold' from 'params.yaml'.
4.  Checks if the champion's F1 score meets the threshold.
5.  If it passes, registers the model in the MLflow Model Registry.
6.  Handles modern (tag-based) and legacy (stage-based) MLflow registry workflows.

Usage:
    uv run python -m src.models.register_model
"""

import json
import yaml
from packaging import version
import mlflow
from mlflow.tracking import MlflowClient

# --- Project Utilities ---
from src.utils.paths import EVAL_DIR, PROJECT_ROOT
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri

logger = get_logger(__name__, headline="register_model.py")


# =====================================================================
# Load Best Model from Evaluation
# =====================================================================
def load_champion_model_data():
    """
    Loads the champion model info and its corresponding F1 score.

    Returns:
        tuple: (model_name, run_id, f1_score)
    """
    try:
        # 1. Read the champion info file created by model_evaluation
        info_path = EVAL_DIR / "best_model_run_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Champion info file not found at {info_path}")

        with open(info_path, "r") as f:
            info = json.load(f)

        model_name = info["model_name"]
        run_id = info["run_id"]

        # 2. Read the champion's corresponding metrics file to get F1 score
        metric_path = EVAL_DIR / f"{model_name}_test_metrics.json"
        if not metric_path.exists():
            raise FileNotFoundError(
                f"Champion's metric file not found at {metric_path}"
            )

        with open(metric_path, "r") as f:
            metrics = json.load(f)

        f1_score = metrics.get("test_macro_f1")
        if f1_score is None:
            raise KeyError(f"'test_macro_f1' not found in {metric_path}")

        logger.info(
            f"🏆 Champion identified: {model_name.upper()} (Run ID: {run_id}) | Test Macro F1: {f1_score:.4f}"
        )
        return model_name, run_id, f1_score

    except Exception as e:
        logger.error(f"Error loading champion model data: {e}")
        raise


# =====================================================================
# Register Best Model
# =====================================================================
def register_best_model(model_name: str, run_id: str, f1: float, f1_threshold: float):
    """
    Register the best-performing model in the MLflow Model Registry,
    if it meets the F1 threshold.
    """
    client = MlflowClient()
    logger.info("🚀 Starting model registration process...")
    model_registry_name = f"youtube_sentiment_{model_name}"

    if f1 < f1_threshold:
        logger.warning(
            f"❌ Model performance below threshold (F1={f1:.4f} < {f1_threshold:.2f}). "
            f"Skipping registration for model '{model_registry_name}'."
        )
        return

    logger.info(
        f"✅ Model performance meets threshold (F1={f1:.4f} >= {f1_threshold:.2f})."
    )

    # --- Register model from run ---
    try:
        logger.info(f"Registering model '{model_registry_name}' from run {run_id}")
        model_uri = f"runs:/{run_id}/model"
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_registry_name,
        )
        logger.info(
            f"✅ Successfully registered model '{model_registry_name}' version {model_version.version}"
        )

    except Exception as e:
        logger.error(f"❌ Failed to register model '{model_registry_name}': {e}")
        raise

    # --- Handle MLflow version differences for stage/tag promotion ---
    mlflow_version = version.parse(mlflow.__version__)
    supports_stages = mlflow_version < version.parse("2.9.0")

    try:
        if supports_stages:
            # ✅ Legacy MLflow (pre-2.9.0): use stage transitions
            client.transition_model_version_stage(
                name=model_registry_name,
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
                name=model_registry_name,
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
            f"⚠️ Stage/Tag assignment failed for model '{model_registry_name}' (v{model_version.version}): {e}"
        )
        logger.info(
            "Continuing without stage assignment (likely unsupported in local MLflow file store)."
        )

    logger.info(
        f"🧾 Registration Summary → "
        f"Model: {model_registry_name} | "
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
        # --- 1. Load params to get F1 threshold ---
        params_path = PROJECT_ROOT / "params.yaml"
        if not params_path.exists():
            raise FileNotFoundError(f"params.yaml not found at {params_path}")

        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        f1_threshold = params.get("register", {}).get("f1_threshold", 0.75)
        logger.info(f"Using F1 threshold for registration: {f1_threshold}")

        # --- 2. Load champion model info from evaluation directory ---
        model_name, run_id, f1 = load_champion_model_data()

        # --- 3. Connect to MLflow Tracking Server ---
        mlflow_uri = get_mlflow_uri()
        mlflow.set_tracking_uri(mlflow_uri)
        logger.info(f"Using MLflow Tracking URI: {mlflow_uri}")

        # --- 4. Register the best model ---
        register_best_model(model_name, run_id, f1, f1_threshold)

    except Exception as e:
        logger.error(f"❌ Model registration failed: {e}")
        # Re-raise to ensure DVC stage fails if registration fails
        raise


if __name__ == "__main__":
    main()
