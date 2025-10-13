## Feature Engineering Script: src/features/feature_engineering.py

This script loads the prepared Parquet splits from `data/processed/`, engineers features for sentiment analysis, and saves feature matrices (X) and labels (y) as compressed NumPy arrays in `data/processed/features/`. Features include:
- **Text-based**: TF-IDF vectors (unigrams/bigrams, max features=5000 for scalability).
- **Derived**: Text length (chars/words), sentiment-specific ratios (e.g., positive word proportion).
- **Preprocessing**: Uses the existing `clean_text` from `make_dataset.py`; vectorizes consistently across splits.

This ensures reproducibility (fit on train only) and adaptability (configurable via params).

To incorporate BERT embeddings as a swappable option, for the script to accept a `use_bert` flag (default: False). When `True`, it replaces TF-IDF with mean-pooled embeddings from a pre-trained BERT model (`bert-base-uncased`), yielding 768-dimensional vectors per comment. This enhances semantic capture for nuanced sentiments but increases compute (recommend GPU for large datasets). Derived features (lengths, ratios) remain appended for hybrid utility.

Add dependencies to `pyproject.toml`:
```
transformers>=4.30
torch>=2.0
accelerate>=0.20  # For efficient inference
```
Then run `uv sync`. For innovation, this enables easy A/B testing (e.g., BERT vs. TF-IDF in MLflow); extend to domain-specific fine-tuning later.

### Usage and Best Practices
- Run TF-IDF: `uv run python src/features/feature_engineering.py`.
- Run BERT: Edit `__main__` to `engineer_features(use_bert=True)` and rerun.
- Outputs: Updated `.npz`/`.pkl` files; BERT adds tokenizer/model saves for inference.
- **Reliability**: Batch processing mitigates OOM; test on subsets first.
- **Scalability**: BERT is ~10x slower—use AWS SageMaker for production.
- **Maintainability**: Flag enables branching in CI/CD; DVC tracks changes.
- **Adaptability**: Innovate by fine-tuning BERT on Reddit data for politics-specific lift.

---

### Necessity of Saving Feature Matrices and Labels as Compressed NumPy Arrays

Saving feature matrices (X) and labels (y) as compressed NumPy arrays in `../../models/features/` is a core MLOps practice for ensuring reproducibility, efficiency, and modularity in the pipeline. It decouples data preparation from modeling, allowing independent iteration without redundant computations.

#### Why Necessary?
- **Reproducibility**: Features (e.g., TF-IDF vectors or BERT embeddings) are deterministic once fitted on the train set. Saving them prevents recomputation on every run, reducing errors from environmental variations (e.g., random seeds in embeddings). This aligns with DVC versioning, where changes in raw data trigger re-engineering without manual intervention.
- **Efficiency and Scalability**: High-dimensional features (e.g., 5,000+ TF-IDF dims) are sparse, consuming gigabytes if stored densely. Compression minimizes storage (e.g., from GBs to MBs) and accelerates loading in training loops, critical for AWS deployments or CI/CD.
- **Pipeline Modularity**: Models (`src/models/train_model.py`) can directly load these artifacts for training/evaluation, while inference (`predict_model.py`) reuses the same format for new YouTube comments. This supports A/B testing (e.g., TF-IDF vs. BERT) without upstream changes.
- **Practical Innovation**: Enables experimentation, like augmenting with YouTube data—load, append, retrain—fostering creative extensions (e.g., federated learning across datasets).

Without this, pipelines risk data leakage, high compute costs, and debugging overhead, undermining reliability and maintainability.

#### How It Works?
- **Format Choice**:
  - **y (.npy)**: Dense 1D NumPy arrays for labels (e.g., shape: (n_samples,)). Simple, uncompressed serialization via `np.save()`. Loading: `y = np.load('y_train.npy')`.
  - **X (.npz)**: Compressed archive for sparse matrices (SciPy CSR format), via `save_npz()`. Handles TF-IDF sparsity efficiently; BERT (dense) is sparsified post-hstack for consistency. Loading: `X = load_npz('X_train.npz')`.
- **Process in Script**:
  1. Engineer X (sparse/dense text features + dense derived) and y (encoded labels).
  2. Save with `save_npz()`/`np.save()`—NumPy's binary format is platform-independent and fast.
  3. Include preprocessors (vectorizer/tokenizer) as .pkl for inference reconstruction.
- **Integration**: In modeling, `load_npz`/`np.load` feeds directly into scikit-learn/PyTorch; DVC tracks `.npz`/`.npy` as outputs for git-like versioning.

This approach is lightweight yet robust—test by loading a file post-run to verify shapes match originals. For innovation, consider HDF5 for larger-scale datasets.

---

### Error Analysis and Fixes

The `TypeError: 'coo_matrix' object is not subscriptable` arises because `load_npz` returns a COO sparse matrix, which lacks support for slicing (e.g., `[:, -4:]`). Solution: Convert to CSR format post-load with `.tocsr()`, enabling efficient indexing. This is standard for SciPy sparse operations.

---

Based on the MLflow metrics provided, the best choice for the **TF-IDF `max_features`** is **$7000$**. This value yields the highest overall accuracy.

The relevant runs and their final overall `accuracy` metric are extracted from the provided data:

| Run ID | `vectorizer_max_features` (Inferred) | `accuracy` |
| :--- | :--- | :--- |
| `c4bfb747eaa341ef93be37f767e80a30` | 10000 | 0.6345634563456346 |
| `c5d92a5cd68941cfa9c27a833351a0d8` | 9000 | 0.6304230423042304 |
| `633b1c20ef5548b48877f44eaa915678` | 8000 | 0.6334833483348334 |
| **`d9ab47065c7447e896471aaa5c859a15`** | **7000** | **0.6477047704770477** |
| `1cebe189798f4d8682f1953ea8058f88` | 6000 | 0.6343834383438344 |
| `87bee845e6434409bfa5edecec23e4d5` | 5000 | 0.6387038703870387 |
| `dcdde7eac604456f8a2fe63d753fba37` | 4000 | 0.6336633663366337 |
| `6ad69553ca6d45a2a95526e1ad91d0c5` | 3000 | 0.6345634563456346 |
| `ec0159be51284e6f8659d5f918f4c0a9` | 2000 | 0.6441044104410441 |
| (Not fully logged) | 1000 | (Incomplete) |

The run with an **accuracy of $0.6477$** (or $64.77\%$) belongs to the run with `run_id` `6ad69553ca6d45a2a95526e1ad91d0c5`.

### Best `max_features`

The **highest accuracy score** of **$0.6477$** was achieved with the run corresponding to `max_features = 7000`.

| Metric | Value ($\text{max\_features} = 7000$) |
| :--- | :--- |
| **Accuracy** | **$0.6477$** |
| Weighted Avg F1-Score | $0.5861$ |
| Macro Avg F1-Score | $0.5219$ |

This suggests that using $7000$ features (trigrams) strikes the best balance between providing enough information for the model and avoiding noise or overfitting compared to the other tested values.

---

## **three TF-IDF vectorizers**. Let’s interpret these results systematically.

---

### 🔹 1. Context Recap

You tested three TF-IDF configurations:

| Run ID                               | Vectorizer       | Accuracy   |
| ------------------------------------ | ---------------- | ---------- |
| **2c2a8f962f0b437bbe2ee8310f880699** | **TF-IDF (1,1)** | **0.6448** |
| e6accc6f0842412fa818252fed6008c7     | TF-IDF (1,2)     | 0.6385     |
| addba9debd9e4aee8dffe432be3b96d6     | TF-IDF (1,3)     | 0.6417     |

Your dataset is moderately imbalanced:

| Sentiment | Count  |    % |
| --------- | ------ | ---: |
| Positive  | 15,830 | 42.5 |
| Neutral   | 13,142 | 35.3 |
| Negative  | 8,277  | 22.2 |

---

### 🔹 2. Aggregate (Weighted) Metrics

| Metric                 | (1,1)      | (1,2)  | (1,3)     |
| ---------------------- | ---------- | ------ | --------- |
| **Accuracy**           | **0.6448** | 0.6385 | 0.6417    |
| **Weighted Precision** | 0.710      | 0.707  | **0.711** |
| **Weighted Recall**    | **0.645**  | 0.638  | 0.642     |
| **Weighted F1-Score**  | **0.582**  | 0.567  | 0.577     |

✅ **Best configuration:** **TF-IDF (1,1)**
Highest accuracy and F1-score, simplest model, and least overfitting risk.

---

### 🔹 3. Class-Level Insights (TF-IDF (1,1))

| Sentiment         | Precision | Recall | F1-Score | Support |
| ----------------- | --------- | ------ | -------- | ------: |
| **Negative (-1)** | 0.94      | 0.06   | 0.12     |    1241 |
| **Neutral (0)**   | 0.68      | 0.77   | 0.72     |    1908 |
| **Positive (1)**  | 0.62      | 0.85   | 0.71     |    2375 |

### 🧩 Interpretation:

* **Positive comments:** Detected best (recall = 0.85).
* **Neutral comments:** Reasonably balanced precision/recall.
* **Negative comments:** High precision but **very low recall** — the model rarely identifies negatives.

The low recall for negatives is expected in imbalanced datasets, where fewer negative samples lead to weak feature learning.

---

### 🔹 4. Why TF-IDF (1,1) Wins

| Factor              | TF-IDF (1,1) | TF-IDF (1,2) | TF-IDF (1,3)   |
| ------------------- | ------------ | ------------ | -------------- |
| Vocabulary size     | Small (fast) | Larger       | Very large     |
| Risk of overfitting | Low          | Medium       | High           |
| Generalization      | ✅ Best       | Decent       | Slightly worse |
| Runtime             | ⚡ Fastest    | Slower       | Slowest        |

Adding bigrams/trigrams slightly hurts generalization because rare n-grams dominate sparse features.

---

### 🔹 5. Next Step: Improving Negative Recall

You can address the poor recall for negative samples using one or several of these strategies:

| Strategy                | Description                                                                                        |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| **Class weighting**     | Give more importance to negative samples. E.g., `RandomForestClassifier(class_weight='balanced')`. |
| **Resampling**          | Oversample negatives (`SMOTE`) or undersample positives.                                           |
| **Feature engineering** | Include sentiment lexicons or emoji tokens (many negatives use “not”, “never”, etc.).              |
| **Model upgrade**       | Try linear models (`LogisticRegression`, `LinearSVC`) that handle sparse text better than trees.   |

---

### 🔹 6. Summary

| Observation         | Conclusion                                           |
| ------------------- | ---------------------------------------------------- |
| Highest accuracy/F1 | TF-IDF (1,1)                                         |
| Model weakness      | Detecting negatives                                  |
| Likely cause        | Class imbalance and sparse negative signals          |
| Next improvement    | Use `class_weight='balanced'` or a linear classifier |
| MLflow setup        | Correct — metrics are well-logged for comparison     |

---

## src\features\imbalance_tuning.py
We have successfully run the imbalance tuning experiments using various techniques to handle class imbalance in our sentiment analysis task. The results have been logged to MLflow for easy comparison. Our DVC + MLflow pipeline is working perfectly, and now we have the full experimental comparison across all imbalance methods.

Let’s interpret the results objectively and select the best approach based on our metrics.

---

## ⚙️ Summary of Results

| Method            | Accuracy   | F1 (Neutral class, 1) | Comments                                                               |
| :---------------- | :--------- | :-------------------- | :--------------------------------------------------------------------- |
| **Class Weights** | 0.6758     | **0.7286**            | Baseline RandomForest with internal balancing                          |
| **Oversampling**  | 0.6723     | 0.7239                | Balanced dataset, but risk of overfitting                              |
| **ADASYN**        | **0.6814** | **0.7333**            | Slightly better overall — adaptive sampling focuses on difficult cases |
| **Undersampling** | 0.6749     | 0.7303                | Simple, efficient, but information loss                                |
| **SMOTE+ENN**     | 0.2696     | 0.1482                | Severe degradation — noisy data or poor overlap between classes        |

---

## 🧠 Interpretation

### 🔹 1. **SMOTE+ENN failed badly**

SMOTE+ENN underperformed (Accuracy ≈ 0.27), which means the combined synthetic/cleaning process destroyed class structure — common when:

* Text embeddings are sparse (TF-IDF).
* Boundary samples overlap heavily.
* You have a large class imbalance and few minority examples.

→ **Drop this method** for text classification unless you switch to dense embeddings (e.g., BERT).

---

### 🔹 2. **ADASYN outperformed slightly**

ADASYN achieved:

* Best **accuracy (0.6814)**
* Best **F1 score (0.7333)**

Because it focuses on generating synthetic samples *only around hard-to-learn regions*, it adapts better to nonlinear class boundaries typical in sentiment data.

→ **Best overall trade-off** between balance and generalization.

---

### 🔹 3. **Class Weights** remain a strong baseline

* No data alteration.
* Stable performance (F1 ≈ 0.7286).
* Ideal for reproducibility and quick training.

→ Recommended if you prioritize **simplicity and speed**.

---

## 🏆 Final Recommendation

| Goal                                               | Best Method       | Reason                                      |
| -------------------------------------------------- | ----------------- | ------------------------------------------- |
| **Best performance (balanced recall + precision)** | **ADASYN**        | Improves minority class learning adaptively |
| **Most stable & production-ready**                 | **Class Weights** | No synthetic noise, simpler reproducibility |
| **Fast & low-resource**                            | **Undersampling** | Useful for quick prototyping                |

---

## 🔍 Next step

You can confirm this choice in MLflow visually:

1. Open your local MLflow UI (`http://127.0.0.1:5000`).
2. Compare all runs under `imbalance_tuning`.
3. Sort by **F1 score** or **accuracy**.
4. Export the ADASYN model as your baseline for the next stage (e.g., hyperparameter tuning or embedding-level model).

---
