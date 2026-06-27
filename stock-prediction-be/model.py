"""
Hybrid SARIMAX-LSTM Univariate Model for Stock Price Prediction.

Refactored from stock_prediction.ipynb (Scenario 2: Univariate).
Predicts the next trading day's closing price for any given ticker
available on Yahoo Finance.

Pipeline:
1. Download closing price data from yfinance
2. Preprocess (forward-fill, business-day frequency)
3. SARIMAX grid search for best (p,d,q) order
4. Fit SARIMAX on full data
5. Compute residuals, scale with MinMaxScaler
6. Train LSTM on residual sequences (LOOK_BACK=30)
7. Predict: sarimax.forecast(1) + lstm_residual_correction
"""

import itertools
import datetime
import logging
import os

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TF warnings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

logger = logging.getLogger(__name__)

# ─── Hyperparameters (from notebook) ─────────────────────────────────────────
LOOK_BACK = 30
LSTM_UNITS = 50
EPOCHS = 100
BATCH_SIZE = 32
DROPOUT_RATE = 0.2
LEARNING_RATE = 0.001

# SARIMAX grid search ranges (from notebook)
P_RANGE = range(0, 3)
D_RANGE = range(0, 2)
Q_RANGE = range(0, 3)


def _download_data(ticker: str) -> pd.DataFrame:
    """Download closing price data from yfinance."""
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=15 * 365)  # ~15 years

    logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}")

    df = yf.download(
        ticker,
        start=str(start_date),
        end=str(end_date),
        progress=False,
        multi_level_index=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data found for ticker '{ticker}'. Please check the ticker symbol.")

    # Extract closing price only (univariate)
    close = df[["Close"]].copy()
    close.columns = ["Close"]

    return close


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing values and set business-day frequency."""
    df = df.ffill().bfill()
    df = df.dropna()
    df = df.asfreq("B")
    df = df.ffill()
    return df


def _sarimax_grid_search(endog: pd.Series) -> tuple:
    """
    Grid search for the best SARIMAX (p,d,q) order based on AIC.
    Univariate — no exogenous variables.
    """
    best_aic = np.inf
    best_order = (0, 1, 0)  # fallback default

    logger.info("Starting SARIMAX grid search...")

    for p, d, q in itertools.product(P_RANGE, D_RANGE, Q_RANGE):
        try:
            model = SARIMAX(
                endog,
                order=(p, d, q),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False, maxiter=200)
            aic = fitted.aic

            if aic < best_aic:
                best_aic = aic
                best_order = (p, d, q)
                logger.info(f"  New best: order={best_order}, AIC={best_aic:.2f}")
        except Exception:
            continue

    logger.info(f"Best SARIMAX order: {best_order}, AIC: {best_aic:.2f}")
    return best_order


def _create_lstm_sequences(data: np.ndarray, look_back: int = LOOK_BACK):
    """
    Create sequences for LSTM input.

    Parameters
    ----------
    data : array of shape (n_samples, 1) — scaled values
    look_back : number of previous time steps as input features

    Returns
    -------
    X : array of shape (samples, look_back, 1)
    y : array of shape (samples,)
    """
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back : i, 0])
        y.append(data[i, 0])
    X = np.array(X)
    y = np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y


def _build_lstm_model(
    look_back: int = LOOK_BACK,
    units: int = LSTM_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE,
):
    """
    Build the LSTM model.

    Architecture (from notebook):
    - LSTM(units, return_sequences=True)
    - Dropout
    - LSTM(units, return_sequences=False)
    - Dropout
    - Dense(25, relu)
    - Dense(1)
    """
    model = Sequential(
        [
            LSTM(units, return_sequences=True, input_shape=(look_back, 1)),
            Dropout(dropout_rate),
            LSTM(units, return_sequences=False),
            Dropout(dropout_rate),
            Dense(25, activation="relu"),
            Dense(1),
        ]
    )

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def train_and_predict(ticker: str) -> dict:
    """
    Full Hybrid SARIMAX-LSTM Univariate pipeline.

    Downloads data, trains both models, and predicts the next trading day's
    closing price.

    Parameters
    ----------
    ticker : str — Yahoo Finance ticker symbol (e.g. "AAPL", "^JKSE")

    Returns
    -------
    dict with keys:
        - ticker: str
        - predicted_price: float
        - prediction_date: str (ISO format, the next business day)
        - last_close: float
        - last_close_date: str (ISO format)
    """
    ticker = ticker.strip().upper()

    # ── 1. Download & preprocess ──────────────────────────────────────────
    df = _download_data(ticker)
    df = _preprocess(df)

    if len(df) < LOOK_BACK + 50:
        raise ValueError(
            f"Not enough data for ticker '{ticker}'. "
            f"Need at least {LOOK_BACK + 50} data points, got {len(df)}."
        )

    endog = df["Close"].squeeze()

    # ── 2. SARIMAX: grid search + fit ─────────────────────────────────────
    best_order = _sarimax_grid_search(endog)

    logger.info(f"Fitting SARIMAX with order={best_order} on {len(endog)} data points...")
    sarimax_model = SARIMAX(
        endog,
        order=best_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    sarimax_fitted = sarimax_model.fit(disp=False, maxiter=500)

    # SARIMAX forecast for next day
    sarimax_forecast = sarimax_fitted.forecast(steps=1).values[0]
    logger.info(f"SARIMAX forecast: {sarimax_forecast:.2f}")

    # ── 3. Compute residuals ──────────────────────────────────────────────
    fitted_values = sarimax_fitted.fittedvalues
    residuals = endog - fitted_values

    # ── 4. Scale residuals ────────────────────────────────────────────────
    scaler_resid = MinMaxScaler(feature_range=(0, 1))
    residuals_scaled = scaler_resid.fit_transform(residuals.values.reshape(-1, 1))

    # ── 5. Create LSTM sequences ──────────────────────────────────────────
    X, y = _create_lstm_sequences(residuals_scaled, LOOK_BACK)

    if len(X) == 0:
        raise ValueError(
            f"Not enough residual data to create LSTM sequences for '{ticker}'."
        )

    # ── 6. Train LSTM ─────────────────────────────────────────────────────
    logger.info(f"Training LSTM on {len(X)} sequences (epochs={EPOCHS})...")
    lstm_model = _build_lstm_model(LOOK_BACK, LSTM_UNITS, DROPOUT_RATE, LEARNING_RATE)

    early_stop = EarlyStopping(
        monitor="loss", patience=10, restore_best_weights=True
    )

    lstm_model.fit(
        X,
        y,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=0,
    )

    # ── 7. Predict residual correction for next day ───────────────────────
    last_sequence = residuals_scaled[-LOOK_BACK:]
    last_sequence = last_sequence.reshape((1, LOOK_BACK, 1))

    residual_pred_scaled = lstm_model.predict(last_sequence, verbose=0)
    residual_pred = scaler_resid.inverse_transform(residual_pred_scaled)[0][0]
    logger.info(f"LSTM residual correction: {residual_pred:.2f}")

    # ── 8. Combine: SARIMAX forecast + LSTM residual correction ───────────
    predicted_price = float(sarimax_forecast + residual_pred)
    logger.info(f"Final prediction: {predicted_price:.2f}")

    # ── 9. Build response ─────────────────────────────────────────────────
    last_close = float(endog.iloc[-1])
    last_close_date = endog.index[-1]

    # Next business day
    prediction_date = last_close_date + pd.tseries.offsets.BDay(1)

    return {
        "ticker": ticker,
        "predicted_price": round(predicted_price, 2),
        "prediction_date": prediction_date.strftime("%Y-%m-%d"),
        "last_close": round(last_close, 2),
        "last_close_date": last_close_date.strftime("%Y-%m-%d"),
    }
