"""
Stock Prediction API — FastAPI Backend

Exposes the Hybrid SARIMAX-LSTM Univariate model as a REST API.
Predictions are cached in Redis with a TTL that expires at midnight.
"""

import logging
import os
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from cache import get_cached_prediction, set_cached_prediction
from model import train_and_predict

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Stock Prediction API",
    description="Predicts next-day closing prices using a Hybrid SARIMAX-LSTM model.",
    version="0.1.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Health check for Docker / monitoring."""
    return {"status": "ok"}


@app.get("/metrics")
def get_metrics():
    """Return evaluation metrics (RMSE, MAE, MAPE) for the models."""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "evaluation_metrics.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "stock-prediction-ml", "results", "evaluation_metrics.csv")
    if not os.path.exists(csv_path):
        csv_path = "evaluation_metrics.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Metrics dataset not found.")
    
    df = pd.read_csv(csv_path)
    metrics = []
    for _, row in df.iterrows():
        metrics.append({
            "model": str(row["Model"]),
            "rmse": round(float(row["RMSE"]), 2),
            "mae": round(float(row["MAE"]), 2),
            "mape": round(float(row["MAPE (%)"]), 4),
        })
    return metrics


@app.get("/history")
def get_history():
    """Return historical actual vs predicted stock prices."""
    csv_path = os.path.join(os.path.dirname(__file__), "..", "predictions_results.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", "stock-prediction-ml", "results", "predictions_results.csv")
    if not os.path.exists(csv_path):
        csv_path = "predictions_results.csv"
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="History dataset not found.")
    
    df = pd.read_csv(csv_path)
    # Return all history rows formatted cleanly
    history = []
    for _, row in df.iterrows():
        history.append({
            "date": str(row["Date"]),
            "actual": round(float(row["Actual_IHSG"]), 2),
            "hybrid_multivariat": round(float(row["Hybrid_Multivariat"]), 2),
            "hybrid_univariat": round(float(row["Hybrid_Univariat"]), 2),
            "lstm_standalone": round(float(row["LSTM_Tunggal"]), 2),
        })
    return history


@app.get("/predict/{ticker}")
def predict(ticker: str):
    """
    Predict the next trading day's closing price for the given ticker.

    1. Check Redis cache — if a prediction for today exists, return it instantly.
    2. If cache miss — train the model and cache the result.
    """
    ticker = ticker.strip().upper()

    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker symbol cannot be empty.")

    # 1. Check cache
    cached = get_cached_prediction(ticker)
    if cached is not None:
        cached["source"] = "cache"
        return cached

    # 2. Train and predict
    try:
        logger.info(f"Cache miss for {ticker}. Starting model training...")
        result = train_and_predict(ticker)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error predicting {ticker}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # 3. Cache the result
    set_cached_prediction(ticker, result)

    result["source"] = "model"
    return result