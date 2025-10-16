"""
Train XGBoost model using Optuna with ADASYN balancing and MLflow logging.

- Maintainability: Modular functions, centralized logging and path utilities.
- Decoupling: Loads features from .npz/.npy files, independent of feature generation script.
- Reproducibility: Saves best params and metrics for DVC tracking.

Usage:
    uv run python -m src.models.xgboost_training
"""

import optuna
import xgboost as xgb
import mlflow
import mlflow.xgboost
import numpy as np
from sklearn.metrics import f1_score
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri
from src.models.helpers.data_loader import load_feature_data, apply_adasyn
from src.models.helpers.train_utils import (
    setup_experiment,
    save_best_params,
    save_metrics_json,
)

logger = get_logger(__name__, headline="xgboost_training.py")


def objective(trial):
    """Objective function for Optuna hyperparameter tuning with MLflow logging."""
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

    # Load data and apply ADASYN
    # Placeholders to unpack 7 values returned by load_feature_data()
    X_train, X_val, _, y_train, y_val, _, _ = load_feature_data()
    X_res, y_res = apply_adasyn(X_train, y_train)

    dtrain = xgb.DMatrix(X_res, label=y_res)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Train model
    model = xgb.train(params, dtrain, num_boost_round=params["n_estimators"])

    # Evaluate
    y_pred_proba = model.predict(dval)
    y_pred = np.argmax(y_pred_proba, axis=1)
    f1 = f1_score(y_val, y_pred, average="macro")

    # Nested MLflow logging for this trial
    with mlflow.start_run(run_name=f"XGBoost_Trial_{trial.number}", nested=True):
        mlflow.set_tag("model_type", "XGBoost")
        mlflow.set_tag("imbalance_method", "ADASYN")
        mlflow.set_tag("experiment_type", "optuna_trial")
        mlflow.log_params(params)
        mlflow.log_metric("val_macro_f1", f1)
        mlflow.xgboost.log_model(model, artifact_path="xgboost_model")

    return f1


if __name__ == "__main__":
    logger.info("🚀 Starting XGBoost training with Optuna hyperparameter tuning...")

    # --- MLflow Setup ---
    mlflow_uri = get_mlflow_uri()
    setup_experiment("Model Training - XGBoost Advanced Tuning", mlflow_uri)

    # --- Parent MLflow run for the entire Optuna study ---
    with mlflow.start_run(run_name="XGBoost_Optuna_Study") as parent_run:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30, show_progress_bar=True)

        best_params = study.best_params
        best_score = study.best_value
        best_f1 = best_score  # For clarity in saving metrics

        # Log best trial to MLflow
        mlflow.set_tag("best_trial_id", study.best_trial.number)
        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_macro_f1", best_score)

        # Retrain model on the best parameters (for artifact logging)
        X_train, X_val, _, y_train, y_val, _, _ = load_feature_data()
        X_res, y_res = apply_adasyn(X_train, y_train)
        dtrain = xgb.DMatrix(X_res, label=y_res)
        best_model = xgb.train(
            best_params, dtrain, num_boost_round=best_params["n_estimators"]
        )

        logger.info(f"✅ Optuna tuning complete | Best F1: {best_score:.4f}")

        mlflow.xgboost.log_model(best_model, artifact_path="best_xgboost_model")

        # Save best parameters AND metrics to disk for DVC tracking
        save_best_params("xgboost", best_params, best_f1)
        save_metrics_json("xgboost", best_f1)

        logger.info(
            f"🎯 Best XGBoost trial ({study.best_trial.number}) logged to MLflow | "
            f"Run ID: {mlflow.active_run().info.run_id}"
        )
