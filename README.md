# YouTube Sentiment Analysis MLOps Pipeline

## 1. Project Overview

The **YouTube Sentiment Analysis** project builds an end-to-end MLOps pipeline for real-time sentiment analysis of YouTube comments. The system automatically processes comments, predicts sentiment, and provides rich, actionable insights for content creators through two bespoke Chrome Extensions.

The core goal is to master and demonstrate modern ML engineering practices, leveraging tools like **DVC for data/pipeline versioning** and **MLflow for experiment tracking/model registry**, alongside CI/CD, containerization, and automated deployment. This creates a production-grade, scalable, and maintainable system.

## 2. Table of Contents

- [Project Overview](#1-project-overview)
- [Table of Contents](#2-table-of-contents)
- [Features](#3-features)
- [System Architecture](#4-system-architecture)
- [Technology Stack](#5-technology-stack)
- [MLOps Pipeline](#6-mlops-pipeline)
- [Installation](#7-installation)
- [Chrome Extensions](#8-chrome-extensions)
  - [Standard Sentiment Insights](#standard-sentiment-insights)
  - [Aspect-Based Sentiment Analysis (ABSA)](#aspect-based-sentiment-analysis-absa)
- [Deployment](#9-deployment)
- [License](#10-license)
- [Contact](#11-contact)

## 3. Features

- **End-to-End MLOps:** Full pipeline automation from data ingestion to model deployment, with `DVC` managing **data and pipeline versioning** and `MLflow` providing **experiment tracking and a model registry**.
- **Dual Model Approach:**
  - **Tree-based models (LightGBM/XGBoost)** for fast and accurate general sentiment prediction (Positive, Neutral, Negative).
  - **Transformer-based Model (BERT)** for nuanced Aspect-Based Sentiment Analysis (ABSA).
- **Dual Chrome Extensions:**
  - **Sentiment Insights:** Provides aggregated sentiment metrics, time-series trend graphs, and word clouds.
  - **ABSA Insights:** Identifies sentiment towards specific, user-defined topics (e.g., "video quality," "presenter").
- **Containerized Deployment:** `Docker` and `Docker Compose` for reproducible, isolated environments for both the API and MLflow services.
- **CI/CD Automation:** `GitHub Actions` for automated testing, linting, and build validation on every push.
- **Reproducibility:** `uv` for fast, lockfile-based dependency management.

## 4. System Architecture

The architecture is composed of three main layers:

1.  **Data & Modeling Layer:**
    - **DVC:** Manages the **data pipeline (`dvc.yaml`)**, **versioning raw, processed, and interim datasets**.
    - **MLflow:** **Tracks experiments**, **logs metrics**, and **manages the model lifecycle** in the Model Registry.
2.  **Inference & API Layer:**
    - **FastAPI:** Serves the trained models via a high-performance REST API. It exposes two main endpoints:
      - `/predict`: For the general sentiment model.
      - `/predict_absa`: For the aspect-based sentiment model.
    - **Docker:** Containerizes the FastAPI application and the MLflow server for consistent deployment.
3.  **Presentation Layer (Frontend):**
    - **Chrome Extensions:** Two vanilla JS extensions that interact with the FastAPI backend to provide real-time insights directly on YouTube video pages.

## 5. Technology Stack

| Layer | Tool | Purpose |
| :--- | :--- | :--- |
| **Methodology** | CRISP-DM + MLOps | Project Lifecycle & Structure |
| **Python** | Python 3.11 | Core Programming Language |
| **Dependencies** | `uv` + `pyproject.toml` | Environment & Dependency Management |
| **Data Versioning** | DVC | Data & Pipeline Version Control |
| **Experiment Tracking** | MLflow | Experiment Logging & Model Registry |
| **Model Serving** | FastAPI + Docker | Real-time Inference API & Containerization |
| **CI/CD** | GitHub Actions | Continuous Integration & Delivery |
| **ML Models** | LightGBM, XGBoost, Transformers | Sentiment & ABSA Modeling |
| **Frontend** | JavaScript (Vanilla) | Chrome Extensions |

## 6. MLOps Pipeline

The project's MLOps core is orchestrated using a **DVC pipeline** defined in `dvc.yaml` for **data and pipeline versioning**, strictly following a **configuration-driven** philosophy. This ensures every stage of the ML lifecycle is reproducible and its artifacts are tracked. **MLflow** is integrated throughout to provide **robust experiment tracking**, allowing for comprehensive logging of parameters, metrics, and models, and managing the **model lifecycle within its Model Registry**.

### Configuration as Code (Single Source of Truth)

Unlike traditional scripts that rely on long, brittle command-line arguments, this project uses `params.yaml` as the single source of truth.

-   **`params.yaml`**: Centralizes all configuration (hyperparameters, file paths, thresholds).
-   **`dvc.yaml`**: Defines the pipeline stages (DAG) but keeps commands clean and simple (e.g., `uv run python -m src.features.feature_engineering`).
-   **Python Scripts**: Directly load their specific configuration from `params.yaml` using `dvc.api.params_show()`.

This approach guarantees that **changing a parameter in `params.yaml` automatically triggers only the necessary pipeline stages** when `dvc repro` is run, ensuring total reproducibility without command-line clutter.

**Key Pipeline Stages:**

1.  **`data_ingestion`**: Downloads raw data using the URL defined in `params.yaml`.
2.  **`data_preparation`**: Cleans and splits data based on `test_size` and `random_state` from `params.yaml`.
3.  **`feature_engineering`**: Generates features (TF-IDF/DistilBERT) using parameters tuned in previous steps.
4.  **`train_advanced_models`**: Trains multiple models (LightGBM, XGBoost) using hyperparameters defined in `params.yaml`.
5.  **`evaluate_models`**: Compares models and selects the "champion" based on metrics.
6.  **`register_model`**: Promotes the champion model to the "Production" stage in MLflow if it meets the `f1_threshold`.

To run the full pipeline:

```bash
# Ensure DVC is initialized and MLflow tracking server is running
dvc repro
```

## 7. Installation

```bash
# 1. Create a fresh virtual environment
uv venv .venv

# 2. Activate the environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. Install all dependencies from the lockfile
uv sync

# 4. (Optional) For local development, pull DVC-tracked data
dvc pull
```

## 8. Chrome Extensions

The project includes two Chrome Extensions for real-time analysis directly on YouTube.

### Standard Sentiment Insights

This extension provides a high-level overview of the sentiment landscape for a video's comment section. It's ideal for quickly gauging the overall community reaction.

-   **Features:** Aggregated sentiment breakdown (Positive, Neutral, Negative), trend graphs, word clouds, and key comment statistics.
-   **Backend Endpoint:** Communicates with `/predict_with_timestamps` and other endpoints from the `insights_api.py` service running on port `8001`.
-   **Location:** `chrome-extension/`

![YouTube Sentiment Insights](reports/figures/YouTube_API/sentiment_prediction/YouTube_API_1.png)

*Figure 1: The Sentiment Insights extension showing an overall analysis of comments.*

### Aspect-Based Sentiment Analysis (ABSA)

This advanced extension identifies sentiment towards specific topics (aspects) within the comments, offering a much more granular analysis.

-   **Features:** Analyzes comments for pre-defined aspects (`video quality`, `audio`, `presenter`, etc.) and displays the sentiment for each.
-   **Backend Endpoint:** Communicates with the `/predict_absa` endpoint.
-   **Location:** `chrome-extension-absa/`

![YouTube ABSA Insights](reports/figures/YouTube_API/aspect_based_sentiment/YouTube_API_4.png)

*Figure 2: The ABSA extension breaking down sentiment by specific aspects.*

### Setup for Both Extensions

1.  **Load the Extensions:**
    -   Open Chrome and navigate to `chrome://extensions/`.
    -   Enable **Developer mode**.
    -   Click **Load unpacked** and select the `chrome-extension/` directory.
    -   Click **Load unpacked** again and select the `chrome-extension-absa/` directory.
2.  **Ensure Backend is Running:** The FastAPI service must be running. See the [Deployment](#9-deployment) section.

## 9. Deployment

The entire application is containerized using Docker and orchestrated with Docker Compose for easy, reproducible deployment.

### Launching the Environment

1.  **Navigate** to the `docker/` directory:
    ```bash
    cd docker/
    ```
2.  **Build and run** all services in detached mode:
    ```bash
    docker-compose up --build -d
    ```

### Services

-   **`youtube_sentiment_api`**: The core FastAPI service running on `http://localhost:8000`. It automatically loads the "Production" model from the MLflow server.
-   **`mlflow_server`**: The MLflow Tracking Server, accessible at `http://localhost:5000`. It uses a mounted volume for persistent storage of experiments and registered models.
-   **`insights_api`**: A separate FastAPI service on `http://localhost:8001` dedicated to generating visualizations (charts, word clouds) for the main Chrome Extension.

### Cleanup

To stop and remove all containers, networks, and volumes:

```bash
docker-compose down -v
```

## 10. License

This project is licensed under the MIT License – see the [LICENSE.txt](LICENSE.txt) file.
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.txt)

## 11. Contact

-   **Project Lead:** Sebastian Garrido – sebastiangarrido2790@gmail.com
-   **GitHub Repository:** [https://github.com/SebastianGarrido2790/Youtube-Sentiment-MLOPS](https://github.com/SebastianGarrido2790/Youtube-Sentiment-MLOPS)
