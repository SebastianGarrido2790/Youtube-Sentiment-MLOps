### Baseline Model Training Script: src/models/train_model.py

Based on the MLflow metrics from `handling_imbalanced_data.py`, the best imbalance handling method is **RandomUnderSampler** (undersampling), which achieved the highest F1-score for the minority class (negative: 0.605) while maintaining strong performance on neutral (F1: 0.743). This method balances classes by reducing majority samples, improving minority detection without synthetic data artifacts, though it discards information—suitable for this baseline given the dataset size. Alternatives like ADASYN (F1 negative: 0.599) offer higher recall (0.597) but slightly lower precision; iterate in advanced runs.

This script loads pre-engineered TF-IDF features (default from `feature_engineering.py`), applies undersampling to the train set, trains a Logistic Regression baseline, and logs to MLflow (accuracy, macro F1, per-class F1). It evaluates on the test set for final metrics. Add `"imbalanced-learn"` and `"scikit-learn"` to `pyproject.toml` if needed (`uv sync`).

#### Usage and Best Practices
- **Run**: Executes training, logs to MLflow (view UI for comparisons), saves model.
- **Reliability**: Undersampling on train only; stratified splits preserved.
- **Scalability**: Sparse matrices efficient; extend to GPU for LSTM.
- **Maintainability**: Modular; DVC add `models/` post-run.
- **Adaptability**: Swap model (e.g., `from sklearn.ensemble import RandomForestClassifier`); add Optuna for tuning in v2.

#### Prototype Inference Endpoint
For quick testing, add this function to `src/models/predict_model.py` (create if needed). It loads the model and preprocesses new YouTube comments via saved vectorizer.

This endpoint is callable from the Chrome extension (e.g., via AWS Lambda later). Test locally; deploy next.

### Advanced Model Training Script: src/models/advanced_training.py

This script extends the pipeline with advanced models: XGBoost and LightGBM on TF-IDF features (tuned for gradient boosting), and BERT fine-tuning via Hugging Face on raw text (for semantic nuance). Optuna optimizes hyperparameters (e.g., learning rate, depth) over 50 trials, using validation F1 (macro) as objective. Imbalance handled via RandomUnderSampler on train. Results logged to MLflow for comparison; best model saved in `models/advanced/`.

Add to `pyproject.toml`:
```
optuna>=3.6
xgboost>=2.0
lightgbm>=4.3
transformers>=4.40
datasets>=2.20  # For BERT data loading
accelerate>=0.30  # For distributed training
```
Run `uv sync`. For BERT, GPU recommended (set `device='cuda'`).

#### Usage and Best Practices
- **Execution**: Runs tuning sequentially (~30-60 min total; BERT longest). Outputs best params/models.
- **Reliability**: Nested runs in Optuna log trials; undersampling applied consistently.
- **Scalability**: Optuna parallelizable (`n_jobs=-1`); BERT uses accelerate for multi-GPU.
- **Maintainability**: Pruning via `Pruner` in Optuna for efficiency; DVC track `models/advanced/`.
- **Adaptability**: Extend objectives for custom metrics; integrate YouTube data for domain tuning.

For innovation, chain Optuna with Bayesian optimization or ensemble best models. Next: Inference prototype or deployment?

### Troubleshooting BERT training process

The process is failing because the **labels in the dataset are not mapped to the expected range** for the PyTorch cross-entropy loss function.

#### 🐛 Root Cause: Label Encoding Mismatch

The sentiment labels are $\{-1, 0, 1\}$. The **PyTorch cross-entropy loss function** (which is used internally by the Hugging Face `Trainer`) requires classification targets to be non-negative integers starting from **zero** (i.e., $0, 1, 2, \dots, N-1$).

Since The labels include **$-1$**, the loss function attempts to index into its output distribution at position $-1$, which results in the `IndexError: Target -1 is out of bounds.`

#### ✅ Solution: Shift the Labels to $\{0, 1, 2\}$

Shift all the labels so the minimum value is $0$. Since the original labels are $\{-1, 0, 1\}$, adding $1$ to each will correctly map them to $\{0, 1, 2\}$.

**Action:** Modify the `bert_objective` function to shift the labels immediately after loading and renaming.

By shifting the labels, the data will be compatible with the standard PyTorch classification loss, allowing the BERT tuning to proceed. The model will now learn to predict $\{0, 1, 2\}$, corresponding to $\{-1, 0, 1\}$.

---

### Rationale for Focusing on F1-Score in Sentiment Analysis

In this pipeline, the F1-score (harmonic mean of precision and recall) is prioritized as the primary evaluation metric due to the dataset's inherent class imbalance (Negative: 22.22%, Neutral: 35.28%, Positive: 42.50%) and the task's practical demands. Below, I outline the key reasons, structured for clarity.

#### 1. **Handling Imbalance Robustly**
   - **Accuracy Pitfalls**: Simple accuracy favors the majority class (e.g., predicting all samples as Positive yields ~42.5% accuracy, masking poor minority performance). This is unreliable for real-world YouTube sentiment, where negatives (e.g., toxic comments) are underrepresented but critical.
   - **F1's Balance**: F1 penalizes imbalances in precision (TP / (TP + FP)) and recall (TP / (TP + FN)), ensuring models detect rare classes without excessive false alarms. Macro-F1 (unweighted average across classes) further equalizes treatment, amplifying minority class contributions—essential here for equitable evaluation.

#### 2. **Alignment with Task Requirements**
   - **Sentiment Nuances**: In video comment analysis, false negatives (missing a negative comment) could overlook harmful content, while false positives dilute trust in positive signals. F1 directly optimizes this trade-off, unlike precision (ignores missed detections) or recall (ignores false alarms).
   - **Multi-Class Suitability**: For three classes, macro-F1 provides a holistic score, while per-class F1 (logged in MLflow) enables granular insights (e.g., boosting Negative F1 from ~0.37 in baselines to ~0.60 with undersampling).

#### 3. **MLOps and Optimization Fit**
   - **Tuning and Selection**: Optuna uses macro-F1 as the objective for hyperparameter search, as it correlates with deployment KPIs (e.g., Chrome extension reliability). Cross-validation on F1 ensures generalizability.
   - **Comparability**: It standardizes A/B testing across models (Logistic Regression, XGBoost, BERT), facilitating selection of the best (e.g., via MLflow UI).

#### Practical Recommendations
- **Thresholding**: In production, adjust decision thresholds per class (e.g., lower for negatives) to fine-tune F1 components.
- **Innovation Opportunity**: Experiment with weighted F1 (emphasizing negatives) or custom metrics (e.g., incorporating latency for real-time inference). Track via MLflow to iterate empirically.

This focus ensures reliable, balanced performance, directly supporting the pipeline's reliability and adaptability goals. If needed, pivot to AUC-PR for probabilistic outputs in advanced iterations.

---
 
### Models Performance

The **macro F1 score** is the primary metric for comparing the performance of these models, as it handles class imbalance better than simple accuracy.

| Model | Best Macro F1 Score |
| :--- | :--- |
| **LightGBM** | **0.79986** (from `LightGBM_Trial_18`) |
| **Logistic Regression** | **0.78679** (from `macro_f1`) |
| **XGBoost** | **0.78317** (from `XGBoost_Trial_22`) |

***

## Performance Summary

| Model | Best Macro F1 Score | Notes |
| :--- | :--- | :--- |
| **LightGBM** | **0.79986** | Achieved the highest performance during Optuna tuning. |
| **Logistic Regression** | 0.78679 | A strong baseline model, performing better than XGBoost. |
| **XGBoost** | 0.78317 | Achieved a high score, but was slightly outperformed by both LightGBM and the Logistic Regression baseline. |

**LightGBM's best trial achieved a Macro F1 score of 0.79986, making it the top-performing model.**

---

## Imbalance Technique
### Choosing Between `class_weight="balanced"` and ADASYN for Baseline Logistic Regression

Excellent question — this is precisely the kind of trade-off thinking that separates **experimentation design** from **production MLOps**.

Let’s analyze both options systematically across **four key criteria** relevant to our current stage:
simplicity, reliability, reproducibility, and signal fidelity (how well the model captures patterns in imbalanced data).

---

## ⚖️ 1. **Purpose of a Baseline Model**

A **baseline model** serves to:

* Establish a *minimal viable benchmark* for downstream models.
* Be *simple, deterministic, and fast to train*.
* Represent the **“expected floor”** of performance before applying complex methods.

Hence, your baseline should emphasize **simplicity and reliability**, not raw performance.

---

## 🔹 Option A — `class_weight="balanced"`

**Mechanism:**
The model adjusts the contribution of each class’s loss term inversely proportional to its frequency.
Mathematically:
[
w_i = \frac{n_{\text{samples}}}{n_{\text{classes}} \times n_i}
]
No resampling, just weighted learning.

**Pros**

* ✅ *Built-in and stable*: native to scikit-learn; minimal risk of data leakage.
* ✅ *Lightweight*: no memory overhead, no synthetic data generation.
* ✅ *Deterministic*: consistent across runs; no random neighbor synthesis.
* ✅ *Ideal for baselines*: interpretable and fast to compute.

**Cons**

* ❌ May underperform in extreme imbalance when minority class signals are very weak.
* ❌ Does not modify class distributions (model still sees the same imbalance in data).

---

## 🔹 Option B — **ADASYN**

**Mechanism:**
Adaptive Synthetic Sampling (He et al., 2008) generates new samples in feature space for underrepresented classes, prioritizing difficult-to-learn regions.

**Pros**

* ✅ Often yields **higher recall and F1**, especially for non-linear models.
* ✅ Can reveal potential upper bounds on what resampling can achieve.

**Cons**

* ❌ Adds synthetic data, increasing memory and CPU cost.
* ❌ Introduces stochasticity — even with fixed random seeds, results can vary slightly.
* ❌ Not ideal for baseline reproducibility (extra data transformations).
* ❌ Risk of minor overfitting or distorted class boundaries with linear models like Logistic Regression.

---

## 📊 Empirical Context — Your Logs

| Method            | Accuracy | Recall | Precision | F1         |
| :---------------- | :------- | :----- | :-------- | :--------- |
| **Class weights** | 0.6758   | 0.9560 | 0.5886    | **0.7286** |
| **ADASYN**        | 0.6814   | 0.9245 | 0.6076    | **0.7333** |

ADASYN slightly outperforms class weights (+0.0047 F1), but both are close — and that’s crucial.

---

## 🧠 3. **Strategic Recommendation**

| Goal                                     | Recommended Approach                                             | Rationale                                                                        |
| ---------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **For Baseline (Current Stage)**         | ✅ `class_weight="balanced"`                                      | Simpler, native, fully deterministic, and strong enough to establish a baseline. |
| **For Advanced Model Training / Tuning** | ✅ ADASYN (or SMOTE variants)                                     | Use in experimentation stages once the pipeline baseline is fixed.               |
| **For Deployment / Production**          | ⚙️ Use whatever imbalance method generalizes best on unseen data | After formal model evaluation and registry comparison.                           |

---

## 🧩 4. Practical Implementation Choice

In our `baseline_logistic.py`, replace ADASYN with the native weight balancing:

```python
model = LogisticRegression(
    C=1.0,
    max_iter=2000,
    solver="liblinear",
    class_weight="balanced",
    random_state=42
)
```

That way:

* We keep the **baseline concept pure** (no data resampling).
* Future stages (e.g., `model_experiments.py`) can explicitly explore ADASYN and SMOTE variants for improved recall.

---

## ✅ Final Answer

> For a **baseline model**, we should use `class_weight="balanced"`.
> It’s simpler, more reliable, fully reproducible, and perfectly suited for establishing our project’s initial benchmark.
>
> Reserve **ADASYN** and other resampling techniques for subsequent **model improvement experiments**, not the baseline stage.

---

| Aspect                   | Benefit                                                                          |
| ------------------------ | -------------------------------------------------------------------------------- |
| **Maintainability**      | Easier to modify or debug one model’s code without affecting others.             |
| **Performance**          | Only imports required libraries (e.g., BERT’s heavy dependencies stay isolated). |
| **Pipeline Integration** | Each script can be a separate DVC/MLflow job.                                    |
| **Testing**              | Each model’s training and data loader can be unit tested independently.          |
| **CI/CD**                | CI can run tests only for changed components.                                    |

| Feature             | Implementation                                             |
| ------------------- | ---------------------------------------------------------- |
| **Reliability**     | Centralized logging, controlled imports, error handling    |
| **Scalability**     | Independent DVC/CI pipeline stages per model               |
| **Maintainability** | Shared helper modules (`data_loader.py`, `train_utils.py`) |
| **Adaptability**    | Easy to add new models or retrain specific ones            |
| **Reproducibility** | MLflow + Optuna integration with saved parameters          |

---

## Data Handling Consistency AcrossPpipeline Stages

Let’s clarify the design difference between the **baseline model** and the **advanced models**, and why the data loading logic diverges.

---

### 🔹 1. Purpose Difference: Benchmark vs. Optimization

| Model Type                                    | Objective                                                                                  | Data Handling                                                                                                        |
| --------------------------------------------- | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| **Baseline (Logistic Regression)**            | Establish a *reliable benchmark* that fully represents the data distribution and encoding. | Uses **all splits (train/val/test)** and the **label encoder** for clean benchmarking and reporting across datasets. |
| **Advanced Models (XGBoost, LightGBM, BERT)** | Optimize for *validation performance* through hyperparameter tuning and resampling.        | Focuses on **train/val** only (test kept untouched). ADASYN is applied on `train` to balance classes dynamically.    |

The **baseline** aims for **end-to-end evaluation**, while **advanced models** are still in the **experimentation phase** — they are not yet “final models,” so we don’t test them against the held-out test set at each trial.

---

### 🔹 2. Why Baseline Uses the Label Encoder (`le`)

The baseline model:

* Operates directly on **encoded labels** (e.g., `[-1, 0, 1] → [0, 1, 2]`),
* Then decodes them back for **readable reporting** (`Negative`, `Neutral`, `Positive`),
* Logs per-class F1 metrics to MLflow.

Advanced models don’t need that yet because:

* During tuning, they only need numeric labels for optimization.
* Label decoding and human-readable metrics come later in the **model evaluation and registration** stage.

---

### 🔹 3. Why Advanced Models Skip the Test Set

Hyperparameter optimization (Optuna) is a **search process**, not a final evaluation:

* Each trial should be scored on a **validation set**, not the test set.
* The **test set** is reserved for *final model evaluation* after choosing the best parameters.
* This prevents **information leakage** — using test data during tuning would bias results.

After tuning completes:

1. The best params (`xgboost_best.pkl`, `lightgbm_best.pkl`) are stored.
2. A **final evaluation script** (e.g., `model_evaluation.py`) loads those parameters,
   retrains on `train + val`, and evaluates on the **test set** once.

---

### 🔹 4. Architectural Summary

| Model Stage             | Splits Used        | Encoder | Balancing                 | Purpose                     |
| ----------------------- | ------------------ | ------- | ------------------------- | --------------------------- |
| **Baseline**            | Train + Val + Test | ✅ Yes   | `class_weight="balanced"` | Reliable benchmark          |
| **Advanced (XGB/LGBM)** | Train + Val        | ❌ No    | ✅ ADASYN                  | Hyperparameter optimization |
| **Final Evaluation**    | Train + Val + Test | ✅ Yes   | ✅ (best method)           | Final model registration    |

---

### ✅ Next Step

When you implement the **final evaluation and registration stage** (`src/models/model_evaluation.py`), that script will:

* Reuse the stored best parameters (`*_best.pkl`),
* Load the **label encoder**,
* Train on the combined train + val set,
* Evaluate and log performance on the **test set** for fair comparison with the baseline.

---

### Explanation of MLflow Logging Structure in `xgboost_training.py` and `lightgbm_training.py`

The logging pattern in your script creates a single parent run named "XGBoost_Optuna_Study" and "LightGBM_Optuna_Study" for the entire Optuna hyperparameter search (50 trials), with each individual trial logged as a **nested child run** (e.g., "XGBoost_Trial_0", "XGBoost_Trial_1", etc.). This is a deliberate design choice aligned with MLflow best practices for hyperparameter optimization workflows. Here's a breakdown:

#### Why a Single Parent Run?
- **Organization and Grouping**: The parent run encapsulates the full study, serving as a high-level container for all trials. This prevents the MLflow UI from being cluttered with 50+ independent runs, making it easier to:
  - Track the overall experiment (e.g., start/end time, aggregated metrics like "best_val_macro_f1").
  - Compare studies across models (e.g., XGBoost vs. LightGBM) at the parent level.
  - Use MLflow's hierarchy: Parent runs provide context (e.g., tags like "experiment_type: advanced_tuning"), while children detail per-trial params/metrics.
- **Efficiency in Nested Mode**: By setting `nested=True` in the trial's `mlflow.start_run()`, MLflow automatically associates child runs with the active parent. This leverages MLflow's run nesting feature, avoiding manual parent-child linking and ensuring traceability without redundant setup.
- **Reproducibility and Auditing**: The parent run logs study-level artifacts (e.g., best params via `mlflow.log_params(best_params)`), while trials log granular details (e.g., per-trial F1). This mirrors CRISP-DM's evaluation phase, where the "study" is the meta-experiment.

#### What Gets Logged Where?
| Level          | Run Name Example          | Contents                                                                 | Purpose |
|----------------|---------------------------|--------------------------------------------------------------------------|---------|
| **Parent**    | XGBoost_Optuna_Study      | - Best params/metrics (e.g., `best_val_macro_f1: 0.7508`).<br>- Tags (e.g., "model_type: XGBoost").<br>- Best model artifact (`best_xgboost_model`). | Summarizes the study; enables cross-study comparisons. |
| **Child (Trials)** | XGBoost_Trial_{n}        | - Trial-specific params (e.g., `max_depth: 8`).<br>- Per-trial metric (`val_macro_f1`).<br>- Trial model artifact (`xgboost_model`). | Details hyperparameter exploration; supports drill-down analysis. |

In the MLflow UI (as shown in your screenshot), the parent "XGBoost_Optuna_Study" appears as the primary run, with child trials expandable under it (via the "Runs" view or search filters like "Run Name contains 'Trial'"). If only the parent is visible at top-level, expand the run tree or filter by tags/experiment.

#### Potential Drawbacks and Alternatives
- **Visibility**: If trials feel "hidden," switch to flat runs by removing `nested=True`—each trial becomes a top-level sibling under the experiment. However, this increases clutter (50+ runs) and loses hierarchy.
- **Customization**: For more granularity, add trial artifacts (e.g., Optuna's `plot_param_importances(study)` as a PNG logged via `mlflow.log_artifact` in the parent).

#### Recommendations
- **Immediate Check**: In MLflow UI, filter runs by "Parent Run ID" matching the study's run ID to view all 50 trials.
- **Enhancement for Innovation**: Extend `train_utils.py` with a `log_study_summary` function to auto-generate a JSON/CSV of all trial params/metrics, logged as a parent artifact. This enables external analysis (e.g., via Pandas in a notebook) without UI reliance.
- **DVC Integration**: Since metrics are now in JSON (via `save_metrics_json`), run `dvc metrics diff` post-repro to compare F1 across pipeline versions—practical for model selection.

This structure balances simplicity with depth, prioritizing a clean audit trail. If trials aren't nesting correctly (e.g., due to `mlflow.end_run()` calls), share console output for troubleshooting.