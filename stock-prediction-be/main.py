"""
Stock Prediction API — FastAPI Backend

Exposes the Hybrid SARIMAX-LSTM Univariate model as a REST API.
Predictions are cached in Redis with a TTL that expires at midnight.
"""

import logging

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