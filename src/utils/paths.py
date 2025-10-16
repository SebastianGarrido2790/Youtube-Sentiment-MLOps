"""
Utility module for defining and managing file paths used throughout the project.
Includes automatic environment detection from `.env`.
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# --- Load environment variables (once, globally) ---
load_dotenv()
ENV = os.getenv("ENV", "local").lower()  # e.g., "local", "staging", "production"

# --- Project Root ---
# Automatically finds the top-level directory (the one containing 'src/')
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Data Directories ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INTERIM_DATA_DIR = DATA_DIR / "interim"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# --- Train and Test Data Paths ---
TRAIN_PATH = PROCESSED_DATA_DIR / "train.parquet"
TEST_PATH = PROCESSED_DATA_DIR / "test.parquet"
VAL_PATH = PROCESSED_DATA_DIR / "val.parquet"

# --- Model, Reports, and Artifacts ---
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FEATURES_DIR = MODELS_DIR / "features"
ADVANCED_DIR = MODELS_DIR / "advanced"

# --- Logs and MLflow ---
# Use system-specific log directory if running in production
if ENV == "production":
    LOGS_DIR = Path("/var/log/my_project")
else:
    LOGS_DIR = PROJECT_ROOT / "logs"

MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# --- Ensure directories exist (for reproducibility) ---
for path in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    LOGS_DIR,
    FEATURES_DIR,
    ADVANCED_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)
