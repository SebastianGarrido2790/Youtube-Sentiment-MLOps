# CI/CD Pipeline for MLOps

This document provides an overview of the Continuous Integration and Continuous Deployment (CI/CD) pipeline for the YouTube Sentiment Analysis project. The pipeline is designed to automate testing, building, securing, and deploying the application, ensuring it aligns with modern MLOps principles of Reliability, Scalability, and Maintainability.

## 1. Overview

The primary purpose of this pipeline is to provide a fast, reliable, and automated way to validate code changes and deploy production-ready artifacts. It integrates several key MLOps practices:

-   **Automated Testing**: Ensures that every change is automatically tested for correctness and quality.
-   **Data and Model Versioning**: Uses DVC to pull versioned data for reproducible tests.
-   **Containerization**: Builds a Docker image as a versioned, deployable artifact.
-   **Security Scanning**: Proactively scans the application and its dependencies for vulnerabilities.
-   **Continuous Deployment**: Automatically deploys the application to a production-like environment after successful testing and building.

## 2. Pipeline Triggers

The pipeline is automatically triggered by the following GitHub events:

-   **Push to `main`**: When code is pushed directly to the `main` branch.
-   **Pull Request**: When a pull request is opened or updated that targets the `main` branch.

The pipeline also uses a `concurrency` setting to automatically cancel any in-progress runs for the same branch or pull request, ensuring that only the latest commit is processed.

```yaml
on:
  push:
    branches: [ main ]
  pull_request:

concurrency:
  group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true
```

## 3. Pipeline Jobs

The pipeline is composed of three main jobs that run sequentially: `test`, `build`, and `deploy`.

### `test` Job

The `test` job is responsible for ensuring code quality, correctness, and reproducibility. It runs on every push and pull request.

**Key Steps:**

1.  **Checkout Code**: Checks out the repository's source code.
2.  **Set up Python**: Configures the Python 3.11 environment.
3.  **Install Dependencies**: Installs project dependencies using `uv`. This step is accelerated by caching `uv` dependencies, which significantly speeds up subsequent runs.
4.  **Linting and Formatting**: Runs `Ruff` to check for linting errors and ensure consistent code formatting.
5.  **Pull DVC Data**: Authenticates with AWS and pulls versioned data, models, and the DVC run-cache using `dvc pull --run-cache`. This MLOps step ensures tests are reproducible and can reuse results from previous pipeline stages.
6.  **Run Tests**: Executes the entire test suite using `pytest`.

**MLOps Highlights:**

-   **Reproducibility**: By using `dvc pull`, the pipeline guarantees that tests are always performed with the same version of the data and models.
-   **Reliability**: Comprehensive linting and testing ensure that code quality remains high and regressions are caught early.
-   **Efficiency**: Caching Python dependencies and the DVC run-cache minimizes setup time, providing faster feedback to developers.

### `build` Job

The `build` job creates a production-ready Docker image and pushes it to a container registry. This job only runs after the `test` job has succeeded on a push to the `main` branch.

**Key Steps:**

1.  **Checkout Code**: Checks out the repository's source code.
2.  **Log in to AWS ECR**: Authenticates with Amazon Elastic Container Registry (ECR).
3.  **Build and Push Docker Image**: Builds the Docker image, tags it with the Git commit hash and other metadata, and pushes it to the configured ECR repository.
4.  **Security Scan**: Scans the newly pushed image for known vulnerabilities using `Trivy`. The pipeline will fail if any `CRITICAL` or `HIGH` severity vulnerabilities are found.
5.  **Health Check**: Runs the container and performs a health check by sending a sample request to the `/predict` endpoint to ensure the application and model are working correctly inside the container.

**MLOps Highlights:**

-   **Automation**: The entire process of building, tagging, and pushing the application container is fully automated.
-   **Security**: Integrated vulnerability scanning is a crucial step in securing the software supply chain.
-   **Deployment-Ready Artifact**: The pipeline produces a versioned, immutable Docker image that is ready for deployment to any containerized environment (e.g., AWS ECS, Kubernetes).

### `deploy` Job

The `deploy` job is responsible for automatically deploying the application to a production environment. This job runs on a **self-hosted runner** (e.g., an AWS EC2 instance) after the `build` job succeeds on a push to `main`.

**Key Steps:**

1.  **Configure AWS Credentials**: Authenticates with AWS using OIDC.
2.  **Login to Amazon ECR**: Connects to the container registry to pull the image.
3.  **Stop and Remove Existing Container**: Checks for a running instance of the application container and, if found, stops and removes it to prepare for the update.
4.  **Pull Latest Image**: Pulls the newly built Docker image from ECR.
5.  **Run New Container**: Starts a new container with the updated image.
6.  **Clean Up Old Images**: Removes unused Docker images to free up disk space on the runner.

**MLOps Highlights:**

-   **Continuous Deployment**: Achieves true CI/CD by automatically deploying every validated change from `main` to a live environment, reducing manual intervention and speeding up delivery.
-   **Zero-Downtime Strategy (Basic)**: While simple, the stop-and-replace mechanism ensures that the old version is removed before the new one starts, providing a clean state transition. More advanced strategies like blue-green could be implemented in the future.
-   **Infrastructure as Code Principle**: The entire deployment logic is defined as code within the GitHub Actions workflow, making the process transparent and reproducible.

## 4. How to Run the Pipeline

The pipeline is designed to run automatically based on Git events. There is no manual trigger configured by default. This section outlines the required setup and the standard developer workflow to run the pipeline.

### 4.1. Prerequisites

For the pipeline to run successfully, you need to configure the following secrets in your GitHub repository settings under **Settings > Secrets and variables > Actions**:

-   `AWS_ACCOUNT_ID`: Your AWS account ID.
-   `AWS_IAM_ROLE_NAME`: The name of the IAM role that GitHub Actions will assume. This role should have permissions to access the DVC S3 bucket and the ECR repository.
-   `AWS_REGION`: The AWS region where your resources are located (e.g., `us-east-1`).
-   `ECR_REPOSITORY`: The name of your ECR repository.

The pipeline uses OpenID Connect (OIDC) to securely authenticate with AWS without needing to store long-lived access keys as secrets. For more information on setting this up, see the [GitHub documentation on configuring OIDC with AWS](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services).

### 4.2. Developer Workflow

The pipeline is triggered by making contributions to the repository through a standard Git workflow:

1.  **Create a Feature Branch**: Create a new branch from `main` for your changes.
    ```bash
    git checkout main
    git pull origin main
    git checkout -b my-new-feature
    ```
2.  **Make Changes**: Implement your new feature or bug fix.
3.  **Push and Create a Pull Request**: Push your branch to GitHub and open a pull request targeting the `main` branch.
    ```bash
    git push origin my-new-feature
    ```
4.  **Review Pipeline Results**: Opening the pull request automatically triggers the `test` job. You can monitor its progress and review the results in the "Actions" tab of the pull request. If the job fails, inspect the logs to diagnose the issue.
5.  **Merge to `main`**: Once the pull request is reviewed, approved, and the `test` job is passing, merge it into the `main` branch.
6.  **Trigger Full CI/CD Pipeline**: The merge to `main` triggers the full `test`, `build`, and `deploy` sequence. If all jobs succeed, the updated application will be automatically deployed to the production environment.
