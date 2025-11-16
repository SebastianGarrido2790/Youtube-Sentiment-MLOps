"""
Comparative Model Evaluation Script.

Evaluates a list of "champion" models (e.g., LightGBM, XGBoost) on the
independent test set using their best hyperparameters (as saved model artifacts).

This script is designed for scalability and can evaluate any number of models
passed via the command line.

It logs to MLflow in a structured way:
- A single Parent Run for the comparison.
- A Child Run for each model, containing its specific metrics and artifacts.
- A comparative ROC curve artifact logged to the Parent Run.

Usage:
    uv run python -m src.models.model_evaluation --models lightgbm xgboost logistic_baseline
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

# --- ML/Scikit-learn Imports ---
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer

# --- Project Utilities ---
from src.utils.paths import ADVANCED_DIR, EVAL_FIG_DIR, PROJECT_ROOT, BASELINE_MODEL_DIR
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri
from src.models.helpers.data_loader import load_feature_data
from src.models.helpers.train_utils import (
    save_test_metrics_json,
    save_best_model_run_info,
)
from src.models.helpers.mlflow_tracking_utils import (
    setup_experiment,
    log_metrics_to_mlflow,
    log_confusion_matrix_as_artifact,
)

# --- Configuration ---
EXPERIMENT_NAME = "Final Model Evaluation - Test Set"

# --- Logging Setup (using centralized utility) ---
logger = get_logger(__name__, headline="model_evaluation.py")


# =====================================================================
#  Core Helper Functions
# =====================================================================
def load_model_artifact(model_name: str):
    """
    Loads the trained model artifact from the correct directory.

    The Logistic Regression baseline is saved as a model bundle (dict)
    in models/baseline/. Other models are saved directly in models/advanced/.

    Args:
        model_name (str): The name of the model to load ('lightgbm', 'xgboost',
                          or 'logistic_baseline').

    Returns:
        object: The trained model object (LogisticRegression, LGBMClassifier, etc.).

    Raises:
        ValueError: If an unknown model name is provided.
        FileNotFoundError: If the model artifact cannot be found.
    """

    if model_name == "logistic_baseline":
        # 1. Baseline model is saved as a 'bundle' (dict with 'model' and 'encoder')
        filepath = BASELINE_MODEL_DIR / "logistic_baseline.pkl"
        logger.info(
            f"Loading baseline model bundle from: {filepath.relative_to(PROJECT_ROOT)}"
        )

        try:
            with open(filepath, "rb") as f:
                model_bundle = pickle.load(f)

            # Extract the actual model object from the dictionary
            model = model_bundle.get("model")

            if model is None:
                raise ValueError("Model object not found in the baseline bundle.")
            return model

        except FileNotFoundError:
            logger.error(f"Baseline model file not found at: {filepath}")
            raise

    elif model_name in ["lightgbm", "xgboost"]:
        # 2. Advanced models are saved directly as the model object
        filepath = ADVANCED_DIR / f"{model_name}_model.pkl"
        logger.info(
            f"Loading advanced model artifact from: {filepath.relative_to(PROJECT_ROOT)}"
        )

        try:
            with open(filepath, "rb") as f:
                model = pickle.load(f)
            return model

        except FileNotFoundError:
            logger.error(f"Advanced model file not found at: {filepath}")
            raise

    else:
        raise ValueError(
            f"Unknown model name: {model_name}. Must be 'lightgbm', 'xgboost', or 'logistic_baseline'."
        )


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model, handling different model types (LGBM vs XGB).
    Returns classification report, confusion matrix, and prediction probabilities.
    """
    try:
        # Standardize prediction logic
        if "LGBMClassifier" in str(type(model)):
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
        elif "xgboost.core.Booster" in str(type(model)):
            # XGBoost DMatrix is required for native Booster object
            dtest = xgb.DMatrix(X_test, label=y_test)
            y_pred_proba = model.predict(dtest)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            # Fallback for standard sklearn-compatible APIs
            logger.warning(f"Unknown model type: {type(model)}. Assuming sklearn API.")
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        logger.info("Model evaluation completed on test set.")
        return report, cm, y_pred_proba
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise


def plot_comparative_roc_curve(y_test_bin, roc_results: list, labels: list):
    """
    Plots a comparative (Macro-Average) One-vs-Rest ROC curve for all models.
    Logs the final plot to the parent MLflow run.
    """
    n_classes = len(labels)
    file_path = EVAL_FIG_DIR / "comparative_roc_curve.png"

    plt.figure(figsize=(12, 10))
    colors = cycle(["blue", "green", "red", "cyan", "magenta", "yellow"])

    for result, color in zip(roc_results, colors):
        model_name = result["name"]
        y_pred_proba = result["proba"]

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot macro-average ROC curve for the current model
        plt.plot(
            fpr["macro"],
            tpr["macro"],
            color=color,
            lw=2,
            label=f"{model_name} (Macro-Avg AUC = {roc_auc['macro']:.3f})",
        )

    # Plot final "chance" line
    plt.plot([0, 1], [0, 1], "k--", lw=2, label="Chance (AUC = 0.50)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Comparative Macro-Average OvR ROC Curves (Test Set)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(file_path)
    plt.close()

    # Log to the current (parent) MLflow run
    mlflow.log_artifact(str(file_path))
    logger.info(
        f"Comparative ROC Curve saved to {file_path.relative_to(PROJECT_ROOT)} and logged to MLflow."
    )


def log_model_to_mlflow(model, model_name: str):
    """Logs the model artifact using the correct MLflow flavor."""
    if "LGBMClassifier" in str(type(model)):
        mlflow.lightgbm.log_model(
            lgb_model=model,
            artifact_path="model",
            registered_model_name=None,
        )
    elif "xgboost.core.Booster" in str(type(model)):
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=None,
        )
    else:
        logger.warning(f"No specific MLflow flavor found for {type(model)}.")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=None,
        )
    logger.info(f"Model artifact for {model_name} logged to MLflow run.")


# =====================================================================
#  Main Execution
# =====================================================================
def main(model_list: list):
    # --- Setup MLflow ---
    mlflow_uri = get_mlflow_uri()
    setup_experiment(EXPERIMENT_NAME, mlflow_uri)

    # --- 1. Load Data & Binarize Labels (Done Once) ---
    try:
        # load_feature_data: X_train, X_val, X_test, y_train, y_val, y_test, le
        _, _, X_test, _, _, y_test, le = load_feature_data(validate_files=True)
        labels = le.classes_.tolist()  # [Negative, Neutral, Positive]

        # Binarize labels for multiclass ROC calculation
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test)

    except FileNotFoundError as e:
        logger.error(f"Failed to load data. Ensure 'dvc repro' is complete. Error: {e}")
        return
    except Exception as e:
        logger.error(f"An error occurred during data loading: {e}")
        raise

    # --- 2. Start Parent MLflow Run for Comparison ---
    with mlflow.start_run(run_name="Model_Comparison_Test_Set") as parent_run:
        logger.info(f"🚀 Starting model comparison for: {model_list}")
        mlflow.set_tag("task", "Comparative Evaluation")
        mlflow.log_param("models_evaluated", ", ".join(model_list))

        roc_results = []  # Store probas for final comparative plot

        champion_metrics = []  # List to track champion model metrics

        # --- 3. Loop and Evaluate Each Model in a Child Run ---
        for model_name in model_list:
            logger.info(f"--- Evaluating model: {model_name} ---")
            with mlflow.start_run(
                run_name=f"Evaluation_{model_name}", nested=True
            ) as child_run:
                try:
                    mlflow.set_tag("model_name", model_name)
                    mlflow.set_tag("parent_run_id", parent_run.info.run_id)

                    # Load model
                    model = load_model_artifact(model_name)

                    # Evaluate (get report, cm, probas)
                    report, cm, y_pred_proba = evaluate_model(model, X_test, y_test)

                    # Calculate Macro AUC Score
                    test_macro_auc = roc_auc_score(
                        y_test, y_pred_proba, multi_class="ovr", average="macro"
                    )

                    # Flatten report for MLflow logging
                    flat_metrics = {
                        f"test_{label}_{metric_name.replace('-score', '_f1')}": value
                        for label, metrics in report.items()
                        if isinstance(metrics, dict)
                        for metric_name, value in metrics.items()
                        if metric_name not in ("support")
                    }
                    flat_metrics["test_macro_auc"] = test_macro_auc
                    log_metrics_to_mlflow(flat_metrics)

                    # Save key metrics to local JSON for DVC tracking
                    save_test_metrics_json(model_name, report)

                    # Log CM artifact to child run
                    log_confusion_matrix_as_artifact(cm, model_name, labels)

                    # Log model artifact (LGBM, XGB, etc.) to child run
                    log_model_to_mlflow(model, model_name)

                    # Store results for comparative ROC plot
                    roc_results.append({"name": model_name, "proba": y_pred_proba})

                    # Add metrics to champion list
                    champion_metrics.append(
                        {
                            "model_name": model_name,
                            "run_id": child_run.info.run_id,
                            "test_macro_auc": test_macro_auc,
                            "test_macro_f1": report["macro avg"]["f1-score"],
                        }
                    )

                    logger.info(
                        f"✅ Evaluation complete for {model_name}. "
                        f"Test Macro F1: {report['macro avg']['f1-score']:.4f} | "
                        f"Test Macro AUC: {test_macro_auc:.4f}"
                    )

                except Exception as e:
                    logger.error(f"Failed to evaluate model {model_name}: {e}")
                    mlflow.set_tag("status", "FAILED")
                    mlflow.log_param("error", str(e))
                    continue  # Continue to the next model

        # --- 4. After Loop: Generate Comparative Artifacts (in Parent Run) ---
        if roc_results:
            logger.info("Generating comparative ROC curve...")
            plot_comparative_roc_curve(y_test_bin, roc_results, labels)

            # Select Champion and Save Info for DVC Registration
            if champion_metrics:
                logger.info("Selecting champion model...")
                # Select champion based on the highest Test Macro AUC
                champion = max(
                    champion_metrics, key=lambda item: item["test_macro_auc"]
                )

                logger.info(
                    f"🏆 Champion selected: {champion['model_name']} "
                    f"(AUC: {champion['test_macro_auc']:.4f}). "
                    f"Saving run info for registration."
                )

                # Save the champion's info to the DVC output file
                save_best_model_run_info(
                    run_id=champion["run_id"], model_name=champion["model_name"]
                )
            else:
                logger.error(
                    "No models were successfully evaluated. Cannot select champion."
                )

        else:
            logger.warning(
                "No models were successfully evaluated. Skipping ROC curve and champion selection."
            )

        logger.info(
            f"🏁 Model comparison complete. View Parent Run: {parent_run.info.run_id}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparative Model Evaluation Script")
    parser.add_argument(
        "--models",
        nargs="+",  # Accepts one or more model names
        required=True,
        help="List of model names to evaluate (e.g., lightgbm xgboost).",
    )
    args = parser.parse_args()

    main(model_list=args.models)
