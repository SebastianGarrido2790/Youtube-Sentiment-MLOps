"""
Utility functions for MLflow configuration across modules.
Fully environment-aware, using ENV loaded from src.utils.paths.
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
        logger.info(f"[ENV={ENV}] Using MLflow from environment variable")
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
