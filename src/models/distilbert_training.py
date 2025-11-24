"""
Fine-tune DistilBERT with Optuna, ADASYN balancing, and MLflow logging.
Saves best model and metrics for DVC tracking.

Features:
    - Controlled via params.yaml → feature_engineering.use_distilbert
    - Logs metrics and hyperparameters to MLflow
    - Skips entirely if DistilBERT is disabled (for CPU setups)

Usage:
    uv run python -m src.models.distilbert_training

Design Considerations:
- Reliability: Uses pre-loaded features/labels; validates inputs.
- Scalability: Leverages Hugging Face Trainer for efficient training.
- Maintainability: Leverages shared helpers (data_loader, train_utils); centralized logging/MLflow.
- Adaptability: Parameterized hyperparameters via Optuna; easily switchable models.
"""

import optuna
import yaml
import numpy as np
from sklearn.metrics import f1_score
import mlflow
import mlflow.transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# --- Project Utilities ---
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri
from src.utils.paths import ADVANCED_DIR, PROJECT_ROOT
from src.models.helpers.data_loader import load_text_data
from src.models.helpers.train_utils import (
    save_hyperparams_bundle,
    save_metrics_json,
)
from src.models.helpers.mlflow_tracking_utils import setup_experiment

logger = get_logger(__name__, headline="bert_training.py")


# ============================================================
#  Load configuration
# ============================================================
def load_params():
    """Load project configuration parameters."""
    params_path = PROJECT_ROOT / "params.yaml"
    if not params_path.exists():
        raise FileNotFoundError(
            f"params.yaml not found at {params_path.relative_to(PROJECT_ROOT)}"
        )
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
#  Objective function for Optuna optimization
# ============================================================
def objective(trial: optuna.trial.Trial) -> float:
    """Define Optuna optimization logic for DistilBERT fine-tuning."""
    train_df, val_df = load_text_data()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        tokenized = tokenizer(
            batch["clean_comment"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        tokenized["labels"] = batch["category"] + 1  # Shift labels to 0–2 range
        return tokenized

    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    )

    training_args = TrainingArguments(
        output_dir=str(ADVANCED_DIR / "distilbert_results"),
        num_train_epochs=trial.suggest_int("num_epochs", 2, 5),
        per_device_train_batch_size=trial.suggest_categorical(
            "batch_size", [8, 16, 32]
        ),
        learning_rate=trial.suggest_float("lr", 1e-5, 5e-5, log=True),
        weight_decay=trial.suggest_float("weight_decay", 0.001, 0.1),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        logging_dir=str(ADVANCED_DIR / "distilbert_logs"),
    )

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        preds = np.argmax(preds, axis=1)
        return {"macro_f1": f1_score(labels, preds, average="macro")}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()
    f1 = results["eval_macro_f1"]

    mlflow.log_metric("val_macro_f1", f1)
    mlflow.transformers.log_model(
        trainer.model, "distilbert_model", task="text-classification"
    )

    return f1


# ============================================================
#  Main entrypoint
# ============================================================
if __name__ == "__main__":
    params = load_params()
    distilbert_config = params.get("train", {}).get("distilbert", {})
    enable_distilbert = str(distilbert_config.get("enable", "false")).lower() == "true"

    # --- Conditional Execution Logic ---
    run_training = False
    # Check if DistilBERT training is enabled and CUDA is available
    if enable_distilbert:
        try:
            import torch

            if torch.cuda.is_available():
                logger.info(
                    "✅ CUDA is available. Proceeding with DistilBERT training."
                )
                run_training = True
            else:
                logger.warning(
                    "⚠️ DistilBERT training skipped: 'enable_distilbert' is true, but CUDA is not available."
                )
        except ImportError:
            logger.error(
                "❌ DistilBERT training skipped: PyTorch is not installed. Please run 'uv add torch'."
            )
    else:
        logger.info(
            "ℹ️ DistilBERT training is disabled in params.yaml (train.distilbert.enable: false)."
        )

    if not run_training:
        logger.warning(
            "Skipping DistilBERT training. Creating placeholder artifacts for DVC continuity."
        )
        # --- Ensure expected DVC outputs exist ---
        distilbert_model_path = ADVANCED_DIR / "distilbert_model.pkl"
        distilbert_results_dir = ADVANCED_DIR / "distilbert_results"
        metrics_path = ADVANCED_DIR / "distilbert_metrics.json"
        hyperparams_path = ADVANCED_DIR / "distilbert_hyperparams.pkl"

        distilbert_model_path.parent.mkdir(parents=True, exist_ok=True)
        distilbert_results_dir.mkdir(parents=True, exist_ok=True)

        # Create lightweight placeholder files so DVC doesn't fail
        with open(distilbert_model_path, "wb") as f:
            f.write(b"")  # empty placeholder file
        with open(metrics_path, "w") as f:
            f.write('{"val_macro_f1": null}')
        with open(hyperparams_path, "wb") as f:
            f.write(b"")

        logger.info(
            "Created placeholder artifacts for skipped DistilBERT stage → DVC continuity ensured."
        )
        exit(0)

    # --- Proceed with training if all checks passed ---
    logger.info("🚀 Starting DistilBERT training with Optuna hyperparameter tuning...")

    mlflow_uri = get_mlflow_uri()
    setup_experiment("DistilBERT - Advanced Tuning", mlflow_uri)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=distilbert_config.get("n_trials", 20))

    best_params = study.best_params
    best_f1 = study.best_value

    save_hyperparams_bundle("distilbert", best_params, best_f1)
    save_metrics_json("distilbert", best_f1)

    logger.info(f"🏁 DistilBERT training complete | Best Macro-F1: {best_f1:.4f}")
