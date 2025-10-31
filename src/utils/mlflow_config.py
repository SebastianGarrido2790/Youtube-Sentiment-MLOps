"""
Utility functions for MLflow configuration across modules.
Fully environment-aware, using ENV loaded from src.utils.paths.

================================================================

How to connect your inference pipeline correctly (app/predict_model.py)?

In the case that your logs show:
⚠️ MLflow registry unavailable or no Production model found
even though your MLflow UI clearly lists a Production model version at http://127.0.0.1:5000/
means that your FastAPI inference process isn’t successfully connecting to the same tracking server instance that MLflow UI is serving.

To correctly run with a registry-enabled backend, you can use an SQLite database (`sqlite:///mlflow.db`).

Start MLflow with the correct parameters:

mlflow server `
    --backend-store-uri sqlite:///mlflow.db `
    --default-artifact-root ./mlruns `
    --host 127.0.0.1 `
    --port 5000

This command:

* Initializes a SQL database (`mlflow.db`) containing all tracking and registry metadata.
* Creates ./mlruns for storing artifacts (model files, metrics, etc.).
* Exposes a REST server at http://127.0.0.1:5000, which now supports:
  * /api/2.0/mlflow/... endpoints (for tracking)
  * /api/2.0/mlflow/registered-models/... endpoints (for model registry)

Hence, the "⚠️ MLflow registry unavailable" warning will now disappear, as soon as your FastAPI inference service connects to the same server.
"""

import os
import yaml
from pathlib import Path
from src.utils.paths import ENV  # Centralized environment detection
from src.utils.logger import get_logger  # Centralized logging

logger = get_logger(__name__)


def get_mlflow_uri(params_path: str = "params.yaml") -> str:
    """
    Returns the MLflow Tracking URI with clear priority and automatic environment handling.
    Detects the appropriate MLflow URI based on the current environment (ENV), checks environment variables, and falls back to params.yaml.
    This function is a pure utility that does not rely on or call the mlflow library itself.
    This isolation is crucial for testing and adaptability.

    Priority:
        1. Environment variable MLFLOW_TRACKING_URI (highest priority)
        2. Environment-based defaults (production/staging/local)
        3. 'feature_comparison.mlflow_uri' in params.yaml (fallback for local mode)

    ENV modes:
        - production  → Use remote MLflow server (must be defined in env vars)
        - staging     → Use test/staging tracking server (optional fallback)
        - local       → Use params.yaml fallback or local ./mlruns directory

    Args:
        params_path (str): Path to params.yaml (default: project root).

    Returns:
        str: MLflow Tracking URI.

    Raises:
        RuntimeError: If no valid URI is found.
    """

    # --- Priority 1: Environment variable (always takes precedence) ---
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        logger.info(f"[ENV={ENV}] Using MLflow from environment variable)")
        return mlflow_uri

    # --- Priority 2: Environment-based defaults ---
    if ENV == "production":
        raise RuntimeError(
            "Production mode requires MLFLOW_TRACKING_URI to be set in environment variables."
        )

    elif ENV == "staging":
        default_staging_uri = "http://staging-mlflow-server:5000"
        logger.info(
            f"[ENV={ENV}] Using default staging MLflow URI: {default_staging_uri}"
        )
        return default_staging_uri

    # --- Priority 3: YAML fallback (local mode only) ---
    elif ENV == "local":
        params_file = Path(params_path)
        if not params_file.exists():
            logger.warning(
                f"params.yaml not found at {params_path}. Using local ./mlruns directory."
            )
            local_uri = "file:./mlruns"
            logger.info(f"[ENV={ENV}] Using local MLflow URI: {local_uri}")
            return local_uri

        try:
            with open(params_file, "r") as f:
                params = yaml.safe_load(f)
                mlflow_uri = params["feature_comparison"]["mlflow_uri"]
                logger.info(
                    f"[ENV={ENV}] Using MLflow URI from params.yaml: {mlflow_uri}"
                )
                return mlflow_uri
        except KeyError:
            logger.warning(
                "[ENV=local] Missing 'feature_comparison.mlflow_uri' in params.yaml."
            )
            local_uri = "file:./mlruns"
            logger.info(f"[ENV={ENV}] Using fallback local MLflow URI: {local_uri}")
            return local_uri

    # --- No valid URI found ---
    raise RuntimeError(
        f"MLflow Tracking URI not found for ENV={ENV}. "
        "Define MLFLOW_TRACKING_URI in your .env or system environment."
    )
