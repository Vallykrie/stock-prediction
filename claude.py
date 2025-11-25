"""
Claude Implementation: Hybrid SARIMAX-LSTM for IHSG Prediction
===============================================================
Implementasi model hybrid SARIMAX-LSTM dengan multiple exogenous variables
untuk prediksi Indeks Harga Saham Gabungan (IHSG)

Author: Claude (Anthropic AI)
Based on: Proposal Skripsi - Pande Kadek Nathan Prabhaswara Sudiara Putra
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import pmdarima as pm
from datetime import datetime
from scipy import stats

warnings.filterwarnings("ignore")

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'START_DATE': '2019-01-01',
    'END_DATE': '2024-12-31',
    'TARGET_TICKER': '^JKSE',  # IHSG
    'EXOG_TICKERS': {
        'USD_IDR': 'IDR=X',
        'GOLD': 'GC=F',
        'SP500': '^GSPC'
    },
    'TEST_SIZE_RATIO': 0.2,
    'LOOK_BACK': 60,
    'LSTM_UNITS_1': 64,
    'LSTM_UNITS_2': 32,
    'DROPOUT_RATE': 0.2,
    'EPOCHS': 100,
    'BATCH_SIZE': 32,
    'RANDOM_SEED': 42
}

# Set random seeds for reproducibility
np.random.seed(CONFIG['RANDOM_SEED'])

# ============================================================================
# DATA ACQUISITION & PREPROCESSING
# ============================================================================

def fetch_data():
    """Download data IHSG dan variabel eksogen dari Yahoo Finance"""
    print("="*80)
    print("FETCHING DATA FROM YAHOO FINANCE")
    print("="*80)
    
    tickers = [CONFIG['TARGET_TICKER']] + list(CONFIG['EXOG_TICKERS'].values())
    print(f"Downloading: {', '.join(tickers)}")
    print(f"Period: {CONFIG['START_DATE']} to {CONFIG['END_DATE']}")
    
    data = yf.download(tickers, start=CONFIG['START_DATE'], end=CONFIG['END_DATE'], progress=False)['Close']
    
    # Rename columns for clarity
    data = data.rename(columns={
        CONFIG['TARGET_TICKER']: 'IHSG',
        CONFIG['EXOG_TICKERS']['USD_IDR']: 'USD_IDR',
        CONFIG['EXOG_TICKERS']['GOLD']: 'GOLD',
        CONFIG['EXOG_TICKERS']['SP500']: 'SP500'
    })
    
    print(f"\nData downloaded successfully!")
    print(f"Shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Missing values before preprocessing:\n{data.isnull().sum()}")
    
    # Handle missing values (interpolation + forward/backward fill)
    data = data.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    print(f"Missing values after preprocessing:\n{data.isnull().sum()}")
    
    return data


def check_stationarity(series, name):
    """Test stasioneritas menggunakan Augmented Dickey-Fuller test"""
    result = adfuller(series.dropna())
    print(f'\n--- Stationarity Test: {name} ---')
    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.3f}')
    
    is_stationary = result[1] < 0.05
    print(f'Result: {"STATIONARY" if is_stationary else "NON-STATIONARY"}')
    
    return is_stationary


def create_dataset(dataset, look_back=1):
    """Create dataset untuk LSTM dengan sliding window"""
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


def create_dataset_univariate(dataset, look_back=1):
    """Create dataset untuk LSTM univariate (residuals)"""
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model dengan metrik RMSE, MAE, MAPE"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # Convert to percentage
    
    print(f"\n{'='*80}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*80}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.4f}%")
    
    # Interpretation
    if mape < 10:
        interpretation = "Sangat Baik"
    elif mape < 20:
        interpretation = "Baik"
    elif mape < 50:
        interpretation = "Cukup"
    else:
        interpretation = "Kurang Baik"
    
    print(f"Interpretasi MAPE: {interpretation}")
    
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'Model': model_name}


# ============================================================================
# HYBRID SARIMAX-LSTM MULTIVARIATE
# ============================================================================

def build_hybrid_multivariate(train_data, test_data):
    """
    Model 1: Hybrid SARIMAX-LSTM dengan multiple exogenous variables
    """
    print("\n" + "="*80)
    print("MODEL 1: HYBRID SARIMAX-LSTM MULTIVARIATE")
    print("="*80)
    
    # Prepare exogenous variables
    exog_train = train_data[['USD_IDR', 'GOLD', 'SP500']]
    exog_test = test_data[['USD_IDR', 'GOLD', 'SP500']]
    
    # ========== STEP 1: SARIMAX for Linear Component ==========
    print("\n[Step 1] Training SARIMAX with exogenous variables...")
    
    model_sarimax = pm.auto_arima(
        train_data['IHSG'], 
        exogenous=exog_train,
        start_p=1, start_q=1,
        max_p=3, max_q=3, 
        m=1,  # Non-seasonal for daily data
        start_P=0, seasonal=False,
        d=None, D=0, 
        trace=True,
        error_action='ignore',  
        suppress_warnings=True, 
        stepwise=True
    )
    
    print(f"\nBest SARIMAX model: {model_sarimax.order}")
    
    # Predictions
    sarimax_pred_train = model_sarimax.predict_in_sample(exogenous=exog_train)
    sarimax_pred_test = model_sarimax.predict(n_periods=len(test_data), exogenous=exog_test)
    
    # ========== STEP 2: Calculate Residuals ==========
    print("\n[Step 2] Calculating residuals...")
    train_residuals = (train_data['IHSG'].values - sarimax_pred_train.values 
                      if hasattr(sarimax_pred_train, 'values') 
                      else train_data['IHSG'].values - sarimax_pred_train)
    
    print(f"Residual statistics:")
    print(f"  Mean: {np.mean(train_residuals):.4f}")
    print(f"  Std:  {np.std(train_residuals):.4f}")
    print(f"  Min:  {np.min(train_residuals):.4f}")
    print(f"  Max:  {np.max(train_residuals):.4f}")
    
    # ========== STEP 3: LSTM for Non-Linear Component ==========
    print("\n[Step 3] Training LSTM on residuals...")
    
    # Scale residuals
    scaler_res = MinMaxScaler(feature_range=(-1, 1))
    # Ensure train_residuals is numpy array before reshaping
    if not isinstance(train_residuals, np.ndarray):
        train_residuals = train_residuals.values
    train_residuals_scaled = scaler_res.fit_transform(train_residuals.reshape(-1, 1))
    
    # Create sequences
    look_back = CONFIG['LOOK_BACK']
    X_train_res, y_train_res = create_dataset_univariate(train_residuals_scaled, look_back)
    
    # Reshape for LSTM [samples, time steps, features]
    X_train_res = np.reshape(X_train_res, (X_train_res.shape[0], X_train_res.shape[1], 1))
    
    # Build LSTM model
    model_lstm = Sequential([
        LSTM(CONFIG['LSTM_UNITS_1'], return_sequences=True, input_shape=(look_back, 1)),
        Dropout(CONFIG['DROPOUT_RATE']),
        LSTM(CONFIG['LSTM_UNITS_2'], return_sequences=False),
        Dropout(CONFIG['DROPOUT_RATE']),
        Dense(1)
    ])
    
    model_lstm.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    
    # Callbacks
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001)
    
    # Train
    history = model_lstm.fit(
        X_train_res, y_train_res, 
        epochs=CONFIG['EPOCHS'], 
        batch_size=CONFIG['BATCH_SIZE'], 
        verbose=0,
        callbacks=[early_stop, reduce_lr]
    )
    
    print(f"LSTM trained for {len(history.history['loss'])} epochs")
    print(f"Final training loss: {history.history['loss'][-1]:.6f}")
    
    # ========== STEP 4: Predict Residuals for Test Set ==========
    print("\n[Step 4] Predicting residuals for test set...")
    
    # We need to use actual past residuals to predict future residuals
    # Concatenate train and test actual residuals
    test_residuals_actual = test_data['IHSG'].values - sarimax_pred_test
    full_residuals = np.concatenate([train_residuals, test_residuals_actual])
    full_residuals_scaled = scaler_res.transform(full_residuals.reshape(-1, 1))
    
    # Create test inputs
    test_inputs = full_residuals_scaled[len(train_residuals) - look_back:]
    X_test_res = []
    for i in range(len(test_data)):
        X_test_res.append(test_inputs[i:(i + look_back), 0])
    
    X_test_res = np.array(X_test_res)
    X_test_res = np.reshape(X_test_res, (X_test_res.shape[0], X_test_res.shape[1], 1))
    
    # Predict
    lstm_pred_res_scaled = model_lstm.predict(X_test_res, verbose=0)
    lstm_pred_res = scaler_res.inverse_transform(lstm_pred_res_scaled)
    
    # ========== STEP 5: Combine Predictions ==========
    print("\n[Step 5] Combining SARIMAX + LSTM predictions...")
    hybrid_pred = sarimax_pred_test + lstm_pred_res.flatten()
    
    return {
        'predictions': hybrid_pred,
        'sarimax_model': model_sarimax,
        'lstm_model': model_lstm,
        'sarimax_pred': sarimax_pred_test,
        'lstm_pred_res': lstm_pred_res.flatten(),
        'train_residuals': train_residuals,
        'test_residuals': test_residuals_actual
    }


# ============================================================================
# HYBRID SARIMAX-LSTM UNIVARIATE
# ============================================================================

def build_hybrid_univariate(train_data, test_data):
    """
    Model 2: Hybrid SARIMAX-LSTM tanpa exogenous variables (univariate)
    """
    print("\n" + "="*80)
    print("MODEL 2: HYBRID SARIMAX-LSTM UNIVARIATE")
    print("="*80)
    
    # ========== STEP 1: SARIMAX without exogenous variables ==========
    print("\n[Step 1] Training SARIMAX without exogenous variables...")
    
    model_sarimax = pm.auto_arima(
        train_data['IHSG'],
        start_p=1, start_q=1,
        max_p=3, max_q=3,
        m=1,
        start_P=0, seasonal=False,
        d=None, D=0,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )
    
    print(f"\nBest SARIMAX model: {model_sarimax.order}")
    
    # Predictions
    sarimax_pred_train = model_sarimax.predict_in_sample()
    sarimax_pred_test = model_sarimax.predict(n_periods=len(test_data))
    
    # ========== STEP 2: Calculate Residuals ==========
    print("\n[Step 2] Calculating residuals...")
    train_residuals = (train_data['IHSG'].values - sarimax_pred_train.values 
                      if hasattr(sarimax_pred_train, 'values') 
                      else train_data['IHSG'].values - sarimax_pred_train)
    
    # ========== STEP 3: LSTM for Residuals ==========
    print("\n[Step 3] Training LSTM on residuals...")
    
    scaler_res = MinMaxScaler(feature_range=(-1, 1))
    # Ensure train_residuals is numpy array before reshaping
    if not isinstance(train_residuals, np.ndarray):
        train_residuals = train_residuals.values
    train_residuals_scaled = scaler_res.fit_transform(train_residuals.reshape(-1, 1))
    
    look_back = CONFIG['LOOK_BACK']
    X_train_res, y_train_res = create_dataset_univariate(train_residuals_scaled, look_back)
    X_train_res = np.reshape(X_train_res, (X_train_res.shape[0], X_train_res.shape[1], 1))
    
    model_lstm = Sequential([
        LSTM(CONFIG['LSTM_UNITS_1'], return_sequences=True, input_shape=(look_back, 1)),
        Dropout(CONFIG['DROPOUT_RATE']),
        LSTM(CONFIG['LSTM_UNITS_2'], return_sequences=False),
        Dropout(CONFIG['DROPOUT_RATE']),
        Dense(1)
    ])
    
    model_lstm.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001)
    
    history = model_lstm.fit(
        X_train_res, y_train_res,
        epochs=CONFIG['EPOCHS'],
        batch_size=CONFIG['BATCH_SIZE'],
        verbose=0,
        callbacks=[early_stop, reduce_lr]
    )
    
    print(f"LSTM trained for {len(history.history['loss'])} epochs")
    
    # ========== STEP 4: Predict Test Residuals ==========
    test_residuals_actual = test_data['IHSG'].values - sarimax_pred_test
    full_residuals = np.concatenate([train_residuals, test_residuals_actual])
    full_residuals_scaled = scaler_res.transform(full_residuals.reshape(-1, 1))
    
    test_inputs = full_residuals_scaled[len(train_residuals) - look_back:]
    X_test_res = []
    for i in range(len(test_data)):
        X_test_res.append(test_inputs[i:(i + look_back), 0])
    
    X_test_res = np.array(X_test_res)
    X_test_res = np.reshape(X_test_res, (X_test_res.shape[0], X_test_res.shape[1], 1))
    
    lstm_pred_res_scaled = model_lstm.predict(X_test_res, verbose=0)
    lstm_pred_res = scaler_res.inverse_transform(lstm_pred_res_scaled)
    
    # ========== STEP 5: Combine ==========
    hybrid_pred = sarimax_pred_test + lstm_pred_res.flatten()
    
    return {
        'predictions': hybrid_pred,
        'sarimax_pred': sarimax_pred_test,
        'lstm_pred_res': lstm_pred_res.flatten()
    }


# ============================================================================
# SINGLE LSTM MULTIVARIATE
# ============================================================================

def build_single_lstm(df, train_size):
    """
    Model 3: Single LSTM dengan multiple variables sebagai baseline
    """
    print("\n" + "="*80)
    print("MODEL 3: SINGLE LSTM MULTIVARIATE")
    print("="*80)
    
    feat_cols = ['IHSG', 'USD_IDR', 'GOLD', 'SP500']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feat_cols])
    
    look_back = CONFIG['LOOK_BACK']
    X, y = create_dataset(scaled_data, look_back)
    
    # Split
    train_size_idx = int(len(X) * (1 - CONFIG['TEST_SIZE_RATIO']))
    X_train, X_test = X[:train_size_idx], X[train_size_idx:]
    y_train, y_test = y[:train_size_idx], y[train_size_idx:]
    
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Build model
    model = Sequential([
        LSTM(CONFIG['LSTM_UNITS_1'], return_sequences=True, input_shape=(look_back, len(feat_cols))),
        Dropout(CONFIG['DROPOUT_RATE']),
        LSTM(CONFIG['LSTM_UNITS_2'], return_sequences=False),
        Dropout(CONFIG['DROPOUT_RATE']),
        Dense(1)
    ])
    
    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    
    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.00001)
    
    print("\nTraining LSTM...")
    history = model.fit(
        X_train, y_train,
        epochs=CONFIG['EPOCHS'],
        batch_size=CONFIG['BATCH_SIZE'],
        verbose=0,
        callbacks=[early_stop, reduce_lr]
    )
    
    print(f"LSTM trained for {len(history.history['loss'])} epochs")
    
    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform
    dummy = np.zeros((len(y_pred_scaled), len(feat_cols)))
    dummy[:, 0] = y_pred_scaled.flatten()
    y_pred = scaler.inverse_transform(dummy)[:, 0]
    
    dummy_actual = np.zeros((len(y_test), len(feat_cols)))
    dummy_actual[:, 0] = y_test
    y_actual = scaler.inverse_transform(dummy_actual)[:, 0]
    
    return {
        'predictions': y_pred,
        'actual': y_actual,
        'model': model
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_prediction_comparison(test_data, hybrid_multi_pred, hybrid_uni_pred, lstm_pred, lstm_actual):
    """Plot 1: Comparison of all model predictions"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Full comparison
    ax1 = axes[0]
    ax1.plot(test_data.index, test_data['IHSG'], label='Actual IHSG', color='black', linewidth=2, alpha=0.8)
    ax1.plot(test_data.index, hybrid_multi_pred, label='Hybrid SARIMAX-LSTM (Multivariate)', 
             color='#1f77b4', linewidth=1.5, linestyle='-')
    ax1.plot(test_data.index, hybrid_uni_pred, label='Hybrid SARIMAX-LSTM (Univariate)', 
             color='#ff7f0e', linewidth=1.5, linestyle='--')
    
    lstm_dates = test_data.index[-len(lstm_pred):]
    ax1.plot(lstm_dates, lstm_pred, label='Single LSTM (Multivariate)', 
             color='#2ca02c', linewidth=1.5, linestyle='-.')
    
    ax1.set_title('IHSG Prediction Comparison - Full Test Period', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('IHSG Price', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Zoomed view (last 60 days)
    ax2 = axes[1]
    zoom_days = 60
    ax2.plot(test_data.index[-zoom_days:], test_data['IHSG'].iloc[-zoom_days:], 
             label='Actual IHSG', color='black', linewidth=2, marker='o', markersize=3, alpha=0.8)
    ax2.plot(test_data.index[-zoom_days:], hybrid_multi_pred[-zoom_days:], 
             label='Hybrid Multivariate', color='#1f77b4', linewidth=1.5, marker='s', markersize=3)
    ax2.plot(test_data.index[-zoom_days:], hybrid_uni_pred[-zoom_days:], 
             label='Hybrid Univariate', color='#ff7f0e', linewidth=1.5, marker='^', markersize=3)
    
    if len(lstm_pred) >= zoom_days:
        ax2.plot(lstm_dates[-zoom_days:], lstm_pred[-zoom_days:], 
                 label='Single LSTM', color='#2ca02c', linewidth=1.5, marker='d', markersize=3)
    
    ax2.set_title(f'IHSG Prediction Comparison - Last {zoom_days} Days (Zoomed)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('IHSG Price', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('claude_prediction_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: claude_prediction_comparison.png")
    plt.close()


def plot_residual_analysis(train_residuals, test_residuals, sarimax_pred, lstm_pred_res):
    """Plot 2: Residual analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Residual time series
    ax1 = axes[0, 0]
    ax1.plot(train_residuals, color='blue', alpha=0.6, label='Train Residuals')
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax1.set_title('SARIMAX Residuals (Training Set)', fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Residual Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Residual distribution
    ax2 = axes[0, 1]
    ax2.hist(train_residuals, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.set_title('Distribution of SARIMAX Residuals', fontweight='bold')
    ax2.set_xlabel('Residual Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Q-Q plot
    ax3 = axes[1, 0]
    stats.probplot(train_residuals, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot of Residuals', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # LSTM residual prediction
    ax4 = axes[1, 1]
    ax4.plot(test_residuals, label='Actual Residuals', color='black', linewidth=2, alpha=0.7)
    ax4.plot(lstm_pred_res, label='LSTM Predicted Residuals', color='red', linewidth=1.5, linestyle='--')
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_title('LSTM Residual Prediction vs Actual', fontweight='bold')
    ax4.set_xlabel('Test Sample Index')
    ax4.set_ylabel('Residual Value')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('claude_residual_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: claude_residual_analysis.png")
    plt.close()


def plot_performance_metrics(metrics_list):
    """Plot 3: Performance metrics comparison"""
    df_metrics = pd.DataFrame(metrics_list)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics_to_plot = ['RMSE', 'MAE', 'MAPE']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        bars = ax.bar(df_metrics['Model'], df_metrics[metric], color=colors, alpha=0.7, edgecolor='black')
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric, fontsize=12)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('claude_performance_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: claude_performance_metrics.png")
    plt.close()


def plot_error_distribution(test_data, hybrid_multi_pred, hybrid_uni_pred, lstm_pred, lstm_actual):
    """Plot 4: Error distribution"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Calculate errors
    error_hybrid_multi = test_data['IHSG'].values - hybrid_multi_pred
    error_hybrid_uni = test_data['IHSG'].values - hybrid_uni_pred
    error_lstm = lstm_actual - lstm_pred
    
    # Error time series
    ax1 = axes[0, 0]
    ax1.plot(error_hybrid_multi, label='Hybrid Multivariate', color='#1f77b4', alpha=0.7)
    ax1.plot(error_hybrid_uni, label='Hybrid Univariate', color='#ff7f0e', alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax1.set_title('Prediction Errors Over Time', fontweight='bold')
    ax1.set_xlabel('Test Sample Index')
    ax1.set_ylabel('Prediction Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Error distribution - Hybrid Multi
    ax2 = axes[0, 1]
    ax2.hist(error_hybrid_multi, bins=30, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.set_title('Error Distribution - Hybrid Multivariate', fontweight='bold')
    ax2.set_xlabel('Error')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)
    
    # Error distribution - Hybrid Uni
    ax3 = axes[1, 0]
    ax3.hist(error_hybrid_uni, bins=30, color='#ff7f0e', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_title('Error Distribution - Hybrid Univariate', fontweight='bold')
    ax3.set_xlabel('Error')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3)
    
    # Error distribution - LSTM
    ax4 = axes[1, 1]
    ax4.hist(error_lstm, bins=30, color='#2ca02c', alpha=0.7, edgecolor='black')
    ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax4.set_title('Error Distribution - Single LSTM', fontweight='bold')
    ax4.set_xlabel('Error')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('claude_error_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: claude_error_distribution.png")
    plt.close()


def plot_correlation_heatmap(df):
    """Plot 5: Correlation heatmap"""
    plt.figure(figsize=(10, 8))
    
    corr_matrix = df[['IHSG', 'USD_IDR', 'GOLD', 'SP500']].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                fmt='.3f', annot_kws={'size': 12, 'weight': 'bold'})
    
    plt.title('Correlation Matrix: IHSG and Exogenous Variables', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('claude_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: claude_correlation_heatmap.png")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("CLAUDE IMPLEMENTATION: HYBRID SARIMAX-LSTM FOR IHSG PREDICTION")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Data Acquisition
    df = fetch_data()
    
    # 2. Check Stationarity
    print("\n" + "="*80)
    print("STATIONARITY TESTS")
    print("="*80)
    check_stationarity(df['IHSG'], 'IHSG')
    check_stationarity(df['USD_IDR'], 'USD_IDR')
    check_stationarity(df['GOLD'], 'GOLD')
    check_stationarity(df['SP500'], 'SP500')
    
    # 3. Train-Test Split
    train_size = int(len(df) * (1 - CONFIG['TEST_SIZE_RATIO']))
    train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]
    
    print(f"\n{'='*80}")
    print("DATA SPLIT")
    print(f"{'='*80}")
    print(f"Total samples: {len(df)}")
    print(f"Training samples: {len(train_data)} ({(1-CONFIG['TEST_SIZE_RATIO'])*100:.0f}%)")
    print(f"Test samples: {len(test_data)} ({CONFIG['TEST_SIZE_RATIO']*100:.0f}%)")
    print(f"Training period: {train_data.index[0]} to {train_data.index[-1]}")
    print(f"Test period: {test_data.index[0]} to {test_data.index[-1]}")
    
    # 4. Build Models
    results = {}
    
    # Model 1: Hybrid Multivariate
    results['hybrid_multi'] = build_hybrid_multivariate(train_data, test_data)
    
    # Model 2: Hybrid Univariate
    results['hybrid_uni'] = build_hybrid_univariate(train_data, test_data)
    
    # Model 3: Single LSTM
    results['lstm'] = build_single_lstm(df, train_size)
    
    # 5. Evaluation
    metrics_list = []
    
    metrics_list.append(evaluate_model(
        test_data['IHSG'].values,
        results['hybrid_multi']['predictions'],
        'Hybrid SARIMAX-LSTM (Multivariate)'
    ))
    
    metrics_list.append(evaluate_model(
        test_data['IHSG'].values,
        results['hybrid_uni']['predictions'],
        'Hybrid SARIMAX-LSTM (Univariate)'
    ))
    
    metrics_list.append(evaluate_model(
        results['lstm']['actual'],
        results['lstm']['predictions'],
        'Single LSTM (Multivariate)'
    ))
    
    # 6. Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_prediction_comparison(
        test_data,
        results['hybrid_multi']['predictions'],
        results['hybrid_uni']['predictions'],
        results['lstm']['predictions'],
        results['lstm']['actual']
    )
    
    plot_residual_analysis(
        results['hybrid_multi']['train_residuals'],
        results['hybrid_multi']['test_residuals'],
        results['hybrid_multi']['sarimax_pred'],
        results['hybrid_multi']['lstm_pred_res']
    )
    
    plot_performance_metrics(metrics_list)
    
    plot_error_distribution(
        test_data,
        results['hybrid_multi']['predictions'],
        results['hybrid_uni']['predictions'],
        results['lstm']['predictions'],
        results['lstm']['actual']
    )
    
    plot_correlation_heatmap(df)
    
    # 7. Save Results to CSV
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    results_df = pd.DataFrame({
        'Date': test_data.index,
        'Actual_IHSG': test_data['IHSG'].values,
        'Hybrid_Multivariate': results['hybrid_multi']['predictions'],
        'Hybrid_Univariate': results['hybrid_uni']['predictions'],
        'Single_LSTM': np.concatenate([
            np.full(len(test_data) - len(results['lstm']['predictions']), np.nan),
            results['lstm']['predictions']
        ]),
        'Error_Hybrid_Multi': test_data['IHSG'].values - results['hybrid_multi']['predictions'],
        'Error_Hybrid_Uni': test_data['IHSG'].values - results['hybrid_uni']['predictions']
    })
    
    results_df.to_csv('claude_results.csv', index=False)
    print("✓ Saved: claude_results.csv")
    
    # 8. Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    metrics_df = pd.DataFrame(metrics_list)
    print("\nPerformance Metrics:")
    print(metrics_df.to_string(index=False))
    
    best_model_idx = metrics_df['RMSE'].idxmin()
    best_model = metrics_df.iloc[best_model_idx]
    
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   RMSE: {best_model['RMSE']:.4f}")
    print(f"   MAE:  {best_model['MAE']:.4f}")
    print(f"   MAPE: {best_model['MAPE']:.4f}%")
    
    print(f"\n{'='*80}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print("\n✅ ALL PROCESSING COMPLETE!")
    print("\nGenerated Files:")
    print("  • claude_prediction_comparison.png")
    print("  • claude_residual_analysis.png")
    print("  • claude_performance_metrics.png")
    print("  • claude_error_distribution.png")
    print("  • claude_correlation_heatmap.png")
    print("  • claude_results.csv")


if __name__ == "__main__":
    main()
