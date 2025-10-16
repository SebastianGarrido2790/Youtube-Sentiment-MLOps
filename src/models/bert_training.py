"""
Fine-tune BERT with Optuna, ADASYN balancing, and MLflow logging.
Saves best model and metrics for DVC tracking.
Usage:
    uv run python -m src.models.bert_training
"""

import optuna
import numpy as np
from sklearn.metrics import f1_score
import mlflow
import mlflow.transformers
from src.utils.logger import get_logger
from src.utils.mlflow_config import get_mlflow_uri
from src.utils.paths import ADVANCED_DIR
from src.models.helpers.data_loader import load_text_data
from src.models.helpers.train_utils import (
    setup_experiment,
    save_best_params,
    save_metrics_json,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

logger = get_logger(__name__, headline="bert_training.py")


def objective(trial):
    train_df, val_df = load_text_data()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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
        "bert-base-uncased", num_labels=3
    )

    training_args = TrainingArguments(
        output_dir=ADVANCED_DIR / "bert_results",
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
        trainer.model, "bert_model", task="text-classification"
    )

    return f1


if __name__ == "__main__":
    mlflow_uri = get_mlflow_uri()
    setup_experiment("BERT - Advanced Tuning", mlflow_uri)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    save_best_params("bert", study.best_params, study.best_value)
    save_metrics_json("bert", study.best_value)
    logger.info(f"BERT Best F1: {study.best_value:.4f}")
