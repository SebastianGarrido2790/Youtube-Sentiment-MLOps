"""
FastAPI Insights API for YouTube Sentiment Analysis.

Extends the core prediction service with visualization endpoints:
- /predict: Basic sentiment predictions on comments.
- /predict_with_timestamps: Predictions including timestamps for trend analysis.
- /generate_chart: Generates pie chart from sentiment counts.
- /generate_wordcloud: Generates wordcloud from comments.
- /generate_trend_graph: Generates monthly sentiment trend line graph.

Usage:
    uv run python -m app.insights_api
Or via Uvicorn:
    uv run uvicorn app.insights_api:app --reload --port 8001

Integrates project utilities for modularity:
- Logging via src.utils.logger
- Paths via src.utils.paths
- MLflow config via src.utils.mlflow_config
- Inference helpers via app.inference_utils

Dependencies: Assumes NLTK data (stopwords, WordNet) is downloaded.

API Endpoint Differentiation:
- predict_model.py (extension-focused): Prioritizes strings for UI display.
- insights_api.py (dashboard-focused): Prioritizes numerics for charts/trends (e.g., "Sentiment: 0" in top comments, numeric trends).
"""

import io
import re
import joblib
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List, Dict, Any

matplotlib.use("Agg")  # Non-interactive backend

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Project Utilities
from src.utils.logger import get_logger
from src.utils.paths import FEATURES_DIR

# Inference Utilities
from app.inference_utils import load_production_model, build_derived_features

# ML/DL Imports
from scipy.sparse import hstack
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

logger = get_logger(__name__, headline="insights_api.py")

app = FastAPI(title="YouTube Sentiment Insights API", version="1.0")

# ============================================================
# CORS Middleware (matches predict_model.py)
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Global NLTK Setup
# ============================================================
_stop_words = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}
_lemmatizer = WordNetLemmatizer()

# ============================================================
# Load Production Model Object (MLflow pyfunc or local LightGBM)
# ============================================================
try:
    model = load_production_model()

except Exception as e:
    logger.error(
        f"❌ FATAL: Service cannot start. Model loading failed from all sources. Error: {e}"
    )
    # Re-raise the exception to prevent the application from starting
    raise

# ============================================================
# Load TF-IDF Vectorizer
# ============================================================
try:
    vectorizer = joblib.load(FEATURES_DIR / "tfidf_vectorizer_max_1000.pkl")
    logger.info("✅ Loaded TF-IDF vectorizer successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load vectorizer: {e}")
    raise e


# Sentiment Mapping (assuming model outputs 0=Negative, 1=Neutral, 2=Positive)
_SENTIMENT_MAP = {
    0: -1,
    1: 0,
    2: 1,
}  # Map encoded to original numeric labels (-1, 0, 1)


# ============================================================
# Pydantic Schemas
# ============================================================
class PredictRequest(BaseModel):
    comments: List[str]


class PredictWithTimestampsRequest(BaseModel):
    comments: List[Dict[str, Any]]  # Each: {"text": str, "timestamp": str (ISO)}


class SentimentCountsRequest(BaseModel):
    sentiment_counts: Dict[str, int]  # Keys: "-1", "0", "1"


class WordCloudRequest(BaseModel):
    comments: List[str]


class TrendGraphRequest(BaseModel):
    sentiment_data: List[
        Dict[str, Any]
    ]  # Each: {"sentiment": int (-1/0/1), "timestamp": str (ISO)}


# ============================================================
# Preprocessing (applied before vectorization)
# ============================================================
def preprocess_comment(comment: str) -> str:
    """Apply consistent preprocessing to match training pipeline."""
    try:
        # Lowercase and strip
        comment = comment.lower().strip()
        # Replace newlines
        comment = re.sub(r"\n", " ", comment)
        # Remove non-alphanumeric except basic punctuation
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)
        # Remove stopwords (retain negation/context words)
        words = [word for word in comment.split() if word not in _stop_words]
        comment = " ".join(words)
        # Lemmatize
        words = [_lemmatizer.lemmatize(word) for word in comment.split()]
        return " ".join(words)
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return comment


def preprocess_comments(comments: List[str]) -> List[str]:
    """Vectorized preprocess for batch."""
    return [preprocess_comment(c) for c in comments]


# ============================================================
# Core Prediction Logic
# ============================================================
def predict_internal(comments: List[str]) -> List[int]:
    """Internal prediction: returns numeric sentiments (-1, 0, 1)."""
    if not comments:
        raise ValueError("No comments provided")

    # Preprocess
    preprocessed = preprocess_comments(comments)
    df = pd.DataFrame({"clean_comment": preprocessed})

    # Vectorize + Derived Features
    X_tfidf = vectorizer.transform(df["clean_comment"])
    X_derived = build_derived_features(df)
    X = hstack([X_tfidf, X_derived])

    # Predict (encoded: 0,1,2)
    preds_encoded = model.predict(X)

    # Map to original numeric (-1,0,1)
    preds_numeric = [_SENTIMENT_MAP.get(int(p), 0) for p in preds_encoded]
    return preds_numeric


# ============================================================
# Endpoints
# ============================================================
@app.get("/")
def home():
    return {"message": "Welcome to YouTube Sentiment Insights API"}


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        sentiments = predict_internal(request.comments)
        response = [
            {"comment": comment, "sentiment": sentiment}
            for comment, sentiment in zip(request.comments, sentiments)
        ]
        logger.info(f"✅ Predictions for {len(request.comments)} comments.")
        return {"results": response}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_with_timestamps")
def predict_with_timestamps(request: PredictWithTimestampsRequest):
    try:
        comments = [item["text"] for item in request.comments]
        timestamps = [item["timestamp"] for item in request.comments]
        sentiments = predict_internal(comments)
        response = [
            {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
            for comment, sentiment, timestamp in zip(comments, sentiments, timestamps)
        ]
        logger.info(f"✅ Timestamped predictions for {len(comments)} comments.")
        return {"results": response}
    except Exception as e:
        logger.error(f"Timestamped prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_chart")
def generate_chart(request: SentimentCountsRequest):
    try:
        counts = request.sentiment_counts
        labels = ["Negative", "Neutral", "Positive"]
        sizes = [
            int(counts.get("-1", 0)),
            int(counts.get("0", 0)),
            int(counts.get("1", 0)),
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ["#ef4444", "#9ca3af", "#10b981"]  # Red (neg), Gray (neu), Green (pos)

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"color": "w"},
        )
        plt.axis("equal")

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True, bbox_inches="tight")
        img_io.seek(0)
        plt.close()

        logger.info("✅ Pie chart generated.")
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        logger.error(f"Chart generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_wordcloud")
def generate_wordcloud(request: WordCloudRequest):
    try:
        if not request.comments:
            raise ValueError("No comments provided")

        # Preprocess for wordcloud
        preprocessed = preprocess_comments(request.comments)
        text = " ".join(preprocessed)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=_stop_words,
            collocations=False,
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format="PNG")
        img_io.seek(0)

        logger.info("✅ Wordcloud generated.")
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        logger.error(f"Wordcloud generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_trend_graph")
def generate_trend_graph(request: TrendGraphRequest):
    try:
        if not request.sentiment_data:
            raise ValueError("No sentiment data provided")

        df = pd.DataFrame(request.sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df["sentiment"] = df["sentiment"].astype(int)

        # Resample monthly counts
        monthly_counts = (
            df.resample("M")["sentiment"].value_counts().unstack(fill_value=0)
        )
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure columns
        for val in [-1, 0, 1]:
            if val not in monthly_percentages.columns:
                monthly_percentages[val] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        # Plot
        plt.figure(figsize=(12, 6))
        colors = {-1: "red", 0: "gray", 1: "green"}
        labels = {-1: "Negative", 0: "Neutral", 1: "Positive"}

        for val in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[val],
                marker="o",
                linestyle="-",
                label=labels[val],
                color=colors[val],
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments (%)")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", bbox_inches="tight")
        img_io.seek(0)
        plt.close()

        logger.info("✅ Trend graph generated.")
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        logger.error(f"Trend graph generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Main Launcher
# ============================================================
if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Starting FastAPI Insights server...")
    logger.info("➡️  Access API docs at: http://127.0.0.1:8001/docs")
    uvicorn.run(
        "app.insights_api:app",
        host="127.0.0.1",
        port=8001,
        reload=True,
        log_level="info",
    )
