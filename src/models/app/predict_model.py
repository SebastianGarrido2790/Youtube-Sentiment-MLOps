"""
FastAPI Inference Service for YouTube Sentiment Analysis.

Loads the latest Production model from MLflow Model Registry (or local DVC fallback)
and exposes a REST API for sentiment prediction.

Usage (local):
if __name__ == "__main__" allows the script to be run directly:
    uv run python -m src.models.app.predict_model
Or via Uvicorn:
    uv run uvicorn src.models.app.predict_model:app --reload --port 8000

Test with cURL or HTTP client:
    curl -X POST "http://127.0.0.1:8000/predict" `
     -H "Content-Type: application/json" `
     -d '{"texts": ["I love this video! It was super helpful and well explained."]}'

Response:
    {
    "predictions": ["Positive"],
    "encoded_labels": [2],
    "probabilities": [[0.0150, 0.0120, 0.9728]],
    "feature_shape": [1, 1004]
    }

- predictions: the human-readable sentiment label ("Positive", "Neutral", "Negative")
- encoded_labels: the encoded numeric value (2, 1, or 0)
- probabilities: class probabilities
- feature_shape: confirms feature dimensions (e.g., `[1, 1004]` for TF-IDF + derived features)

Verify the API is Live:
    Swagger UI: 👉 http://127.0.0.1:8000/docs
    → You can test predictions interactively here.
    ReDoc (read-only): 👉 http://127.0.0.1:8000/redoc

If everything is configured properly, you should see your /predict endpoint documented.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from scipy.sparse import hstack
import joblib
import mlflow
import numpy as np
from src.utils.logger import get_logger
from src.utils.paths import FEATURES_DIR, ADVANCED_DIR

logger = get_logger(__name__, headline="predict_model.py")

app = FastAPI(title="YouTube Sentiment Prediction API", version="1.0")

# ============================================================
# Load artifacts (vectorizer, label encoder, model)
# ============================================================
try:
    vectorizer = joblib.load(FEATURES_DIR / "tfidf_vectorizer_max_1000.pkl")
    label_encoder = joblib.load(FEATURES_DIR / "label_encoder.pkl")
    logger.info("✅ Loaded TF-IDF vectorizer and label encoder successfully.")
except Exception as e:
    logger.error(f"Failed to load vectorizer or encoder: {e}")
    raise e

# Try MLflow model registry first
try:
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    model_name = "youtube_sentiment_lightgbm"
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/Production")
    logger.info(f"✅ Loaded model from MLflow registry → {model_name} (Production)")
except Exception:
    import joblib

    model_path = ADVANCED_DIR / "lightgbm_model.pkl"
    model = joblib.load(model_path)
    logger.warning(
        f"⚠️ MLflow registry unavailable or no Production model found. "
        f"Loaded local LightGBM model from {model_path}"
    )


# ============================================================
# Request Schema
# ============================================================
class PredictRequest(BaseModel):
    texts: list[str]


# ============================================================
# Derived Feature Builder (must match training logic)
# ============================================================
def build_derived_features(df: pd.DataFrame) -> np.ndarray:
    """Recreate simple derived features used during training."""
    df["char_len"] = df["clean_comment"].str.len()
    df["word_len"] = df["clean_comment"].str.split().str.len()

    pos_words = {"good", "great", "love", "like", "positive", "best"}
    neg_words = {"bad", "hate", "worst", "negative", "shit", "fuck"}

    def count_lexicon_ratio(text, lexicon):
        words = text.split()
        return len([w for w in words if w in lexicon]) / max(len(words), 1)

    df["pos_ratio"] = df["clean_comment"].apply(
        lambda x: count_lexicon_ratio(x, pos_words)
    )
    df["neg_ratio"] = df["clean_comment"].apply(
        lambda x: count_lexicon_ratio(x, neg_words)
    )

    return df[["char_len", "word_len", "pos_ratio", "neg_ratio"]].values


# ============================================================
# Prediction Endpoint
# ============================================================
@app.post("/predict")
def predict(data: PredictRequest):
    try:
        df_input = pd.DataFrame({"clean_comment": data.texts})

        # Vectorize text
        X_tfidf = vectorizer.transform(df_input["clean_comment"])

        # Add derived features
        X_derived = build_derived_features(df_input)

        # Combine TF-IDF + Derived
        X = hstack([X_tfidf, X_derived])

        # Predict
        preds = model.predict(X)
        probs = (
            model.predict_proba(X).tolist() if hasattr(model, "predict_proba") else None
        )
        decoded_preds = label_encoder.inverse_transform(preds)

        logger.info(f"✅ Prediction completed for {len(data.texts)} samples.")

        # --- Safe return handling ---
        def _safe_to_list(x):
            """Convert numpy/scipy objects to Python lists safely."""
            import numpy as np
            from scipy.sparse import issparse

            if isinstance(x, list):
                return x
            if issparse(x):
                return x.toarray().tolist()
            if isinstance(x, np.ndarray):
                return x.tolist()
            return [x]  # fallback for scalars

        # --- Decode numeric predictions into human-readable labels ---
        try:
            decoded_preds = label_encoder.inverse_transform(preds)
        except Exception as e:
            logger.warning(f"Label decoding failed: {e}")
            decoded_preds = preds

        # --- Construct final JSON response ---
        return {
            "predictions": _safe_to_list(decoded_preds),  # Human-readable labels
            "encoded_labels": _safe_to_list(preds),  # Numeric labels (for debugging)
            "probabilities": _safe_to_list(probs),
            "feature_shape": list(X.shape),
        }

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------
# Main Launcher
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from src.utils.logger import get_logger

    logger = get_logger(__name__, headline="FastAPI Launcher")

    logger.info("🚀 Starting FastAPI inference server...")
    logger.info("➡️  Access the API documentation at http://127.0.0.1:8000/docs")
    logger.info("➡️  Access the ReDoc UI at http://127.0.0.1:8000/redoc")

    uvicorn.run(
        "src.models.app.predict_model:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Auto-reload during development
        log_level="info",
    )
