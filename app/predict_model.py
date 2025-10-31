"""
FastAPI Inference Service for YouTube Sentiment Analysis.

Loads the latest Production model from MLflow Model Registry (via alias-based loading)
or falls back to a local LightGBM model.

Usage (local):
    uv run python -m app.predict_model
Or via Uvicorn:
    uv run uvicorn app.predict_model:app --reload --port 8000

Test with:
    curl -X POST "http://127.0.0.1:8000/predict" `
     -H "Content-Type: application/json" `
     -d '{"texts": ["I love this video! It was super helpful and well explained."]}'

Response Example:
    {
      "predictions": ["Positive"],
      "encoded_labels": [2],
      "feature_shape": [1, 1004]
    }

NOTE:
    - To use the registry, ensure you set a model alias:
        client.set_registered_model_alias("youtube_sentiment_lightgbm", "Production", 2)
    - For URI handling, see: src/utils/mlflow_config.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from scipy.sparse import hstack
import joblib
import numpy as np

# --- Project Utilities ---
from src.utils.logger import get_logger
from src.utils.paths import FEATURES_DIR
from app.inference_utils import load_production_model, build_derived_features

logger = get_logger(__name__, headline="predict_model.py")

app = FastAPI(title="YouTube Sentiment Prediction API", version="1.0")

# ============================================================
# Allow CORS from browser extension
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Artifact Loading (Run on startup only)
# ============================================================

# The model object (MLflow pyfunc or local LightGBM)
try:
    model = load_production_model()

except Exception as e:
    logger.error(
        f"❌ FATAL: Service cannot start. Model loading failed from all sources. Error: {e}"
    )
    # Re-raise the exception to prevent the application from starting
    raise


# ============================================================
# Load TF-IDF Vectorizer and Label Encoder
# ============================================================
try:
    vectorizer = joblib.load(FEATURES_DIR / "tfidf_vectorizer_max_1000.pkl")
    label_encoder = joblib.load(FEATURES_DIR / "label_encoder.pkl")
    logger.info("✅ Loaded TF-IDF vectorizer and label encoder successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load feature artifacts: {e}")
    # If feature artifacts fail to load, the service cannot run.
    raise RuntimeError(
        f"Failed to initialize service due to missing feature artifacts: {e}"
    )


# ============================================================
# Request Schema
# ============================================================
class PredictRequest(BaseModel):
    texts: list[str]


# ============================================================
# Prediction Endpoint
# ============================================================
@app.post("/predict")
def predict(data: PredictRequest):
    try:
        df_input = pd.DataFrame({"clean_comment": data.texts})

        # Vectorize text
        X_tfidf = vectorizer.transform(df_input["clean_comment"])

        # Derived features
        X_derived = build_derived_features(df_input)

        # Combine TF-IDF + Derived
        X = hstack([X_tfidf, X_derived])

        # Predict
        preds = model.predict(X)
        decoded_preds = label_encoder.inverse_transform(preds)
        # probs = (
        #     model.predict_proba(X).tolist() if hasattr(model, "predict_proba") else None
        # )

        logger.info(f"✅ Prediction completed for {len(data.texts)} samples.")

        def _safe_to_list(x):
            """Convert numpy/scipy objects to Python lists safely."""
            from scipy.sparse import issparse

            if isinstance(x, list):
                return x
            if issparse(x):
                return x.toarray().tolist()
            if isinstance(x, np.ndarray):
                return x.tolist()
            return [x]

        return {
            "predictions": _safe_to_list(decoded_preds),  # Human-readable labels
            "encoded_labels": _safe_to_list(preds),
            "feature_shape": list(X.shape),
            # "probabilities": _safe_to_list(probs),  # Not returning to reduce payload size
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Main Launcher
# ============================================================
if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting FastAPI inference server...")
    logger.info("➡️  Access API docs at: http://127.0.0.1:8000/docs")
    logger.info("➡️  Access ReDoc at: http://127.0.0.1:8000/redoc")

    uvicorn.run(
        "app.predict_model:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
    )
