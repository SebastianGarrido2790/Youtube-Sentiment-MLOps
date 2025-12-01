# DVC and MLflow for Reproducible MLOps

This document provides a comprehensive overview of how DVC (Data Version Control) and MLflow are integrated into this project to create a reproducible, automated, and maintainable MLOps workflow.

## 1. Overview: The "Why"

In MLOps, the goal is to bridge the gap between model development and operations. This requires two key capabilities that Git alone cannot provide:
1.  **Versioning large files**: Git is not designed to handle large datasets, models, or other binary artifacts.
2.  **Tracking experiments**: A typical project involves dozzn if not hundreds of experiments with different parameters and code versions. Tracking which parameters produced which model is critical.

This project solves these challenges by combining DVC and MLflow:
-   **DVC** acts as the pipeline orchestrator and data/model version control system.
-   **MLflow** acts as the experiment tracker and model registry.

Together, they provide end-to-end lineage, from data ingestion to model deployment.

## 2. DVC: The Pipeline and Data Versioning Layer

DVC extends Git by adding capabilities to version large files and define a reproducible pipeline.

### How it Works in This Project

#### `dvc.yaml`: The MLOps Pipeline Definition
The `dvc.yaml` file defines the entire ML pipeline as a Directed Acyclic Graph (DAG). Each `stage` is a step in the pipeline with defined inputs and outputs.

**Example: The `feature_engineering` stage**
```yaml
feature_engineering:
  cmd: uv run python -m src.features.feature_engineering
  deps:
    - data/processed/train.parquet
    - src/features/feature_engineering.py
  params:
    - feature_engineering.use_distilbert
    - imbalance_tuning.best_max_features
    - imbalance_tuning.best_ngram_range
  outs:
    - models/features/
```
-   **`cmd`**: The command to execute. Notice it is clean and simple; the script loads its own parameters directly.
-   **`deps`**: Dependencies. If any of these files change (e.g., `feature_engineering.py`), DVC knows to re-run this stage.
-   **`params`**: Parameters from `params.yaml` that this stage depends on. Even though they aren't passed in the command, listing them here ensures DVC detects changes in `params.yaml` and triggers a re-run.
-   **`outs`**: Outputs. DVC tracks the directory `models/features/`. The actual data is stored in DVC's cache and tracked via `.dvc` files in Git.

#### `params.yaml`: Centralized Configuration
The `params.yaml` file centralizes all pipeline parameters, from data split ratios to model hyperparameters. This is a core MLOps principle for **Adaptability** and **Reproducibility**.

**Example:**
```yaml
imbalance_tuning:
  imbalance_methods: "['class_weights','oversampling','adasyn','undersampling','smote_enn']"
  best_max_features: 1000  # Best result from feature_tuning
  best_ngram_range: (1,1)  # Best result from feature_comparison
```
When `dvc repro feature_engineering` is run, the Python script internally calls `dvc.api.params_show()` to read `1000` for `best_max_features`.

### MLOps Advantages of DVC
-   **Reproducibility**: DVC tracks the exact version of code, data, and parameters used to produce a result. Anyone can check out a Git commit and run `dvc repro` to reproduce an experiment perfectly.
-   **Efficiency**: DVC only re-runs stages where dependencies or parameters have changed, saving significant computation time.
-   **Collaboration**: Team members can easily share and access large datasets and models without bloating the Git repository.

### How to Use DVC in This Project
1.  **Reproduce the Pipeline**: To run the entire pipeline from start to finish, simply execute:
    ```bash
    dvc repro
    ```
    DVC will automatically determine which stages need to be run.

2.  **Compare Metrics**: To see how metrics have changed between your current workspace and the last Git commit, run:
    ```bash
    dvc metrics diff
    ```
    This command reads the `.json` files defined in the `metrics` section of `dvc.yaml` stages.

## 3. MLflow: The Experiment Tracking and Model Registry Layer

MLflow is used to log the details of every experiment and manage the lifecycle of trained models.

### How it Works in This Project

#### Experiment Tracking (`src/features/**` & `src/models/**`)
The Python scripts in `src/` are instrumented with MLflow logging.

**Example: `src/models/hyperparameter_tuning.py`**
This script uses a nested MLflow run structure for clarity:
1.  A **Parent Run** is created for the entire Optuna study (e.g., "XGBoost_Optuna_Study").
2.  Each Optuna trial is logged as a **Child Run** within the parent.

```python
# From src/models/hyperparameter_tuning.py

# Parent run for the whole study
with mlflow.start_run(run_name=f"{model_name.upper()}_Optuna_Study") as parent_run:
    # ... Optuna study creation ...

    # Inside the objective function, a nested run is created for each trial
    with mlflow.start_run(run_name=f"LightGBM_Trial_{trial.number}", nested=True):
        mlflow.log_params(params)
        mlflow.log_metric("val_macro_f1", f1)
```
This logs all parameters (`mlflow.log_params`), metrics (`mlflow.log_metric`), and even artifacts like plots (`mlflow.log_artifact`) for every single experiment, providing complete traceability.

#### Model Registry (`src/models/register_model.py`)
After evaluation, the best-performing model (the "champion") is registered with the MLflow Model Registry.

The `register_model.py` script automates this process:
1.  It reads the `run_id` of the champion model, which was saved by the `model_evaluation` stage.
2.  It checks if the model's performance (`test_macro_f1`) exceeds a quality threshold defined in `params.yaml`.
    ```yaml
    # From params.yaml
    register:
      f1_threshold: 0.75
    ```
3.  If the threshold is met, it registers the model from its run URI (`runs:/<run_id>/model`) and promotes it to the "Production" stage (or applies a "Production" tag).

### MLOps Advantages of MLflow
-   **Traceability**: Provides a central UI to see every experiment ever run, including its code version, parameters, metrics, and output artifacts.
-   **Comparability**: The MLflow UI makes it easy to compare dozens of runs to identify the best-performing models.
-   **Model Lifecycle Management**: The Model Registry provides a structured way to manage model versions, transition them between stages (e.g., Staging to Production), and load them for inference.

### How to Use MLflow in This Project
1.  **Start the MLflow UI**:
    ```bash
    # Ensure the MLflow server is running (as defined in docker-compose.yml or run manually)
    mlflow ui --host 127.0.0.1 --port 5000
    ```
2.  **View Experiments**: Open your browser to `http://127.0.0.1:5000`. You can browse experiments, compare runs, and inspect artifacts.
3.  **View Registered Models**: Navigate to the "Models" tab to see registered models and their versions.

## 4. The Combined DVC + MLflow Workflow

The true power comes from using DVC and MLflow together.

Here is the end-to-end developer workflow:
1.  **Change a Parameter**: A developer modifies a hyperparameter in `params.yaml`, for example, increasing the number of `n_trials` for LightGBM.
2.  **Run the Pipeline**: The developer runs `dvc repro`.
3.  **DVC Executes Stages**: DVC detects the change in `params.yaml` and re-runs the `hyperparameter_tuning_lightgbm` stage.
4.  **Scripts Log to MLflow**: The `hyperparameter_tuning.py` script executes, and as it runs, it logs all its Optuna trials as nested runs in MLflow.
5.  **DVC Tracks Outputs**: Once the script finishes, DVC versions the output model (`lightgbm_model.pkl`) and metrics file (`lightgbm_metrics.json`).
6.  **Analyze Results**:
    -   The developer can run `dvc metrics diff` to see the change in the primary metric directly in the terminal.
    -   For a deeper analysis, they can open the MLflow UI to compare the full learning curves and parameters of all the new trials.
7.  **Commit and Share**: The developer commits the changes to `dvc.lock` and `params.yaml`. When another team member pulls the commit and runs `dvc pull`, they retrieve the exact model produced by that experiment.

This combined workflow ensures a fully automated, reproducible, and traceable system, which is the foundation of a production-grade MLOps environment.

-----

## ⚙️ MLOps Best Practice: Configuration-Driven Pipeline

Using `params.yaml` over hardcoded values is essential because it externalizes configuration, making the entire pipeline reproducible, traceable, and flexible.

| Practice | `params.yaml` | Hardcoded Values | Rationale |
| :--- | :--- | :--- | :--- |
| **Reproducibility** | **Excellent.** DVC automatically tracks the specific parameter values used for every run in `dvc.lock`. | **Poor.** To reproduce a result, you must manually check the script's source code from a specific Git commit. | Ensures model results are linked to specific configuration values. |
| **Adaptability** | **Excellent.** Parameters can be changed in a single file. For example, the `test_size` can be adjusted without touching Python code. | **Poor.** Requires modifying and re-committing the Python source code for every change, which is error-prone. | Allows flexible retraining and easy swapping of configurations (e.g., changing the data URL). |
| **Maintainability** | **Excellent.** All hyper-parameters and file paths are centralized in one file. | **Poor.** Configuration values are scattered across multiple script files. | Centralizes governance and keeps Python scripts clean and focused on logic. |
| **Reliability** | **Excellent.** DVC can detect if a parameter has changed and automatically triggers a re-run of the affected stage. | **Poor.** Changes may not automatically trigger necessary re-runs in the pipeline. | Ensures the pipeline computes the correct lineage after any configuration change. |

-----

## ✅ How The Code Follows the Best Practice

The current setup correctly implements this configuration-driven approach using DVC and `params.yaml`.

### 1\. **`dvc.yaml` Stage Definition**

The `dvc.yaml` relies on DVC's parameter tracking to trigger re-runs, but the command itself remains clean:

```yaml
  data_preparation:
    cmd: uv run python -m src.data.make_dataset
    # ...
    params:
      - data_preparation.test_size  # DVC monitors this for changes
      - data_preparation.random_state # DVC monitors this for changes
    # ...
```

DVC watches `params.yaml` for changes to these keys. If they change, it invalidates the stage cache.

### 2\. **`src` Scripts**

The Python scripts themselves (e.g., `src/data/make_dataset.py`) are responsible for reading the configuration:

```python
# Inside src/data/make_dataset.py
import dvc.api

def load_params():
    # Loads directly from params.yaml
    params = dvc.api.params_show()
    return params["data_preparation"]
```

This ensures that `params.yaml` is the **Single Source of Truth**. The CLI arguments are reserved strictly for local debugging or temporary overrides.