"""
Centralized Hyperparameter Optimization Script using Optuna.

This script provides a framework for running hyperparameter tuning for different models.
It is designed to be adaptable and can be extended to support various models like LightGBM, XGBoost, etc.

- Reads model and search space configuration from params.yaml.
- Logs experiments to MLflow.
- Saves best hyperparameters and metrics for DVC tracking.

Usage:
    uv run python -m src.models.hyperparameter_tuning --model [lightgbm|xgboost]

Design Considerations:
- Adaptability: Can be configured to tune different models by specifying the model name.
- Maintainability: Centralizes the tuning logic, reducing code duplication.
- Reproducibility: Logs all trials to MLflow and saves the best results.
"""

import argparse
import optuna
import mlflow
import yaml
from sklearn.metrics import f1_score

# --- Project Utilities ---
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri
from src.utils.paths import PROJECT_ROOT
from src.models.helpers.data_loader import load_feature_data, apply_adasyn
from src.models.helpers.train_utils import (
    save_hyperparams_bundle,
    save_model_object,
    save_metrics_json,
)
from src.models.helpers.mlflow_tracking_utils import setup_experiment

# --- Model Specific Imports ---
import lightgbm as lgb
import xgboost as xgb
import numpy as np

logger = get_logger(__name__, headline="hyperparameter_tuning.py")


def load_params():
    """Load project configuration parameters."""
    params_path = PROJECT_ROOT / "params.yaml"
    if not params_path.exists():
        raise FileNotFoundError(f"params.yaml not found at {params_path}")
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def get_objective(model_name: str):
    """
    Returns the objective function for the specified model.
    """
    if model_name == "lightgbm":
        return lightgbm_objective
    elif model_name == "xgboost":
        return xgboost_objective
    # Add other models here
    else:
        raise ValueError(
            f"Model '{model_name}' is not supported for hyperparameter tuning."
        )


def lightgbm_objective(trial):
    """Objective function for LightGBM."""
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
        "verbose": -1,
    }

    # Load engineered features and apply ADASYN
    # Placeholders to unpack 7 values returned by load_feature_data()
    X_train, X_val, _, y_train, y_val, _, _ = load_feature_data()
    X_res, y_res = apply_adasyn(X_train, y_train)

    model = lgb.LGBMClassifier(**params)
    model.fit(X_res, y_res)

    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="macro")

    with mlflow.start_run(run_name=f"LightGBM_Trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("val_macro_f1", f1)

    return f1


def xgboost_objective(trial):
    """Objective function for XGBoost."""
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
    }

    # Load engineered features and apply ADASYN
    # Placeholders to unpack 7 values returned by load_feature_data()
    X_train, X_val, _, y_train, y_val, _, _ = load_feature_data()
    X_res, y_res = apply_adasyn(X_train, y_train)

    dtrain = xgb.DMatrix(X_res, label=y_res)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(params, dtrain, num_boost_round=params["n_estimators"])

    y_pred_proba = model.predict(dval)
    y_pred = np.argmax(y_pred_proba, axis=1)
    f1 = f1_score(y_val, y_pred, average="macro")

    with mlflow.start_run(run_name=f"XGBoost_Trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("val_macro_f1", f1)

    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning Script")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lightgbm", "xgboost"],
        help="The model to tune.",
    )
    args = parser.parse_args()

    model_name = args.model
    logger.info(f"🚀 Starting hyperparameter tuning for {model_name.upper()}...")

    params = load_params()
    tuning_params = params.get("hyperparameter_tuning", {}).get(model_name, {})
    n_trials = tuning_params.get("n_trials", 20)

    # --- Setup MLflow Experiment ---
    mlflow_uri = get_mlflow_uri()
    setup_experiment(f"Hyperparameter Tuning - {model_name.upper()}", mlflow_uri)

    # --- Parent MLflow run for the Optuna study ---
    with mlflow.start_run(run_name=f"{model_name.upper()}_Optuna_Study") as parent_run:
        objective_func = get_objective(model_name)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_func, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value

        mlflow.set_tag("best_trial_id", study.best_trial.number)
        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_macro_f1", best_score)

        # Retrain model on the best parameters (for artifact logging)
        X_train, _, _, y_train, _, _, _ = load_feature_data()
        X_res, y_res = apply_adasyn(X_train, y_train)

        if model_name == "lightgbm":
            best_model = lgb.LGBMClassifier(**best_params)
            best_model.fit(X_res, y_res)
            mlflow.lightgbm.log_model(
                best_model, artifact_path=f"best_{model_name}_model"
            )
        elif model_name == "xgboost":
            # --- FIX: Merge static params with best_params ---
            # best_params only contains tuned values. We must re-add
            # the objective and other static params for correct retraining.
            static_params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "random_state": 42,
            }
            final_params = {**static_params, **best_params}

            # Pop 'n_estimators' as it's passed as 'num_boost_round'
            # This also fixes the "Parameters: { n_estimators } are not used" warning.
            num_boost_round = final_params.pop("n_estimators", 100)

            dtrain = xgb.DMatrix(X_res, label=y_res)
            best_model = xgb.train(
                final_params, dtrain, num_boost_round=num_boost_round
            )
            mlflow.xgboost.log_model(
                best_model, artifact_path=f"best_{model_name}_model"
            )

        logger.info(
            f"✅ Optuna tuning complete for {model_name} | Best F1: {best_score:.4f}"
        )

        # Save the actual model object, not just params/score.
        save_model_object(best_model, model_name)
        # Save params to a separate file (optional, for logging purposes)
        save_hyperparams_bundle(model_name, best_params, best_score)
        # Save score for DVC metrics tracking
        save_metrics_json(model_name, best_score)

        logger.info(
            f"🎯 Best {model_name.upper()} trial ({study.best_trial.number}) logged to MLflow | "
            f"Run ID: {mlflow.active_run().info.run_id}"
        )
