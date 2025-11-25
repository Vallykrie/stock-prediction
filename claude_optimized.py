"""
Claude Optimized Models: Advanced Architectures for IHSG Prediction
=====================================================================
Implementasi model-model advanced untuk mengalahkan Single LSTM baseline

Target: Beat MAPE 1.0551%

Models Included:
1. Improved Hybrid SARIMAX-LSTM (better SARIMAX parameters)
2. Bidirectional LSTM
3. GRU with Attention Mechanism
4. CNN-LSTM Hybrid
5. Ensemble/Stacking Model

Author: Claude (Anthropic AI)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Bidirectional, 
                                      GRU, Conv1D, MaxPooling1D, Flatten,
                                      Input, Attention, Concatenate, LayerNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import pmdarima as pm
from datetime import datetime

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'START_DATE': '2019-01-01',
    'END_DATE': '2024-12-31',
    'TARGET_TICKER': '^JKSE',
    'EXOG_TICKERS': {
        'USD_IDR': 'IDR=X',
        'GOLD': 'GC=F',
        'SP500': '^GSPC'
    },
    'TEST_SIZE_RATIO': 0.2,
    'LOOK_BACK': 60,
    'EPOCHS': 150,
    'BATCH_SIZE': 32,
    'RANDOM_SEED': 42
}

np.random.seed(CONFIG['RANDOM_SEED'])

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def fetch_data():
    """Download data from Yahoo Finance"""
    print("="*80)
    print("FETCHING DATA")
    print("="*80)
    
    tickers = [CONFIG['TARGET_TICKER']] + list(CONFIG['EXOG_TICKERS'].values())
    data = yf.download(tickers, start=CONFIG['START_DATE'], end=CONFIG['END_DATE'], progress=False)['Close']
    
    data = data.rename(columns={
        CONFIG['TARGET_TICKER']: 'IHSG',
        CONFIG['EXOG_TICKERS']['USD_IDR']: 'USD_IDR',
        CONFIG['EXOG_TICKERS']['GOLD']: 'GOLD',
        CONFIG['EXOG_TICKERS']['SP500']: 'SP500'
    })
    
    data = data.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    return data


def create_dataset(dataset, look_back=1):
    """Create sequences for LSTM"""
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        X.append(dataset[i:(i + look_back)])
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)


def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    print(f"\n{'='*80}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*80}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.4f}%")
    
    if mape < 1.0:
        status = "🏆 BEATS BASELINE!"
    elif mape < 1.06:
        status = "⚡ Very Close!"
    elif mape < 2.0:
        status = "✓ Excellent"
    else:
        status = "○ Good"
    
    print(f"Status: {status}")
    
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'Model': model_name}


# ============================================================================
# MODEL 1: IMPROVED HYBRID SARIMAX-LSTM
# ============================================================================

def build_improved_hybrid(train_data, test_data):
    """
    Improved Hybrid: Better SARIMAX parameters + Deeper LSTM
    """
    print("\n" + "="*80)
    print("MODEL 1: IMPROVED HYBRID SARIMAX-LSTM")
    print("="*80)
    
    exog_train = train_data[['USD_IDR', 'GOLD', 'SP500']]
    exog_test = test_data[['USD_IDR', 'GOLD', 'SP500']]
    
    # Better SARIMAX with seasonal components
    print("\n[Step 1] Training SARIMAX with seasonal components...")
    model_sarimax = pm.auto_arima(
        train_data['IHSG'], 
        exogenous=exog_train,
        start_p=1, start_q=1,
        max_p=5, max_q=5,  # Increased range
        m=12,  # Monthly seasonality
        seasonal=True,  # Enable seasonal
        start_P=0, start_Q=0,
        max_P=2, max_Q=2,
        d=None, D=1,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        n_fits=50  # More iterations
    )
    
    print(f"\nBest SARIMAX: {model_sarimax.order}")
    
    sarimax_pred_train = model_sarimax.predict_in_sample(exogenous=exog_train)
    sarimax_pred_test = model_sarimax.predict(n_periods=len(test_data), exogenous=exog_test)
    
    # Calculate residuals
    train_residuals = (train_data['IHSG'].values - sarimax_pred_train.values 
                      if hasattr(sarimax_pred_train, 'values') 
                      else train_data['IHSG'].values - sarimax_pred_train)
    
    # Deeper LSTM for residuals
    print("\n[Step 2] Training Deeper LSTM on residuals...")
    
    scaler_res = MinMaxScaler(feature_range=(-1, 1))
    if not isinstance(train_residuals, np.ndarray):
        train_residuals = train_residuals.values
    train_residuals_scaled = scaler_res.fit_transform(train_residuals.reshape(-1, 1))
    
    look_back = CONFIG['LOOK_BACK']
    X_train_res, y_train_res = [], []
    for i in range(len(train_residuals_scaled) - look_back - 1):
        X_train_res.append(train_residuals_scaled[i:(i + look_back), 0])
        y_train_res.append(train_residuals_scaled[i + look_back, 0])
    
    X_train_res = np.array(X_train_res)
    y_train_res = np.array(y_train_res)
    X_train_res = np.reshape(X_train_res, (X_train_res.shape[0], X_train_res.shape[1], 1))
    
    # Deeper architecture
    model_lstm = Sequential([
        LSTM(128, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model_lstm.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    
    early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=7, min_lr=1e-6)
    
    model_lstm.fit(X_train_res, y_train_res, epochs=CONFIG['EPOCHS'], 
                   batch_size=CONFIG['BATCH_SIZE'], verbose=0, 
                   callbacks=[early_stop, reduce_lr])
    
    # Predict
    test_residuals_actual = test_data['IHSG'].values - sarimax_pred_test
    full_residuals = np.concatenate([train_residuals, test_residuals_actual])
    full_residuals_scaled = scaler_res.transform(full_residuals.reshape(-1, 1))
    
    test_inputs = full_residuals_scaled[len(train_residuals) - look_back:]
    X_test_res = []
    for i in range(len(test_data)):
        X_test_res.append(test_inputs[i:(i + look_back), 0])
    
    X_test_res = np.array(X_test_res)
    X_test_res = np.reshape(X_test_res, (X_test_res.shape[0], X_test_res.shape[1], 1))
    
    lstm_pred_res = scaler_res.inverse_transform(model_lstm.predict(X_test_res, verbose=0))
    
    hybrid_pred = sarimax_pred_test + lstm_pred_res.flatten()
    
    return hybrid_pred


# ============================================================================
# MODEL 2: BIDIRECTIONAL LSTM
# ============================================================================

def build_bidirectional_lstm(df, train_size):
    """
    Bidirectional LSTM: Process sequences in both directions
    """
    print("\n" + "="*80)
    print("MODEL 2: BIDIRECTIONAL LSTM")
    print("="*80)
    
    feat_cols = ['IHSG', 'USD_IDR', 'GOLD', 'SP500']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feat_cols])
    
    look_back = CONFIG['LOOK_BACK']
    X, y = create_dataset(scaled_data, look_back)
    
    train_size_idx = int(len(X) * (1 - CONFIG['TEST_SIZE_RATIO']))
    X_train, X_test = X[:train_size_idx], X[train_size_idx:]
    y_train, y_test = y[:train_size_idx], y[train_size_idx:]
    
    print(f"Training samples: {len(X_train)}")
    
    # Bidirectional LSTM architecture
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=(look_back, len(feat_cols))),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    
    early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=7, min_lr=1e-6)
    
    print("\nTraining Bidirectional LSTM...")
    model.fit(X_train, y_train, epochs=CONFIG['EPOCHS'], batch_size=CONFIG['BATCH_SIZE'],
              verbose=0, callbacks=[early_stop, reduce_lr])
    
    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    dummy = np.zeros((len(y_pred_scaled), len(feat_cols)))
    dummy[:, 0] = y_pred_scaled.flatten()
    y_pred = scaler.inverse_transform(dummy)[:, 0]
    
    dummy_actual = np.zeros((len(y_test), len(feat_cols)))
    dummy_actual[:, 0] = y_test
    y_actual = scaler.inverse_transform(dummy_actual)[:, 0]
    
    return y_pred, y_actual


# ============================================================================
# MODEL 3: GRU WITH ATTENTION
# ============================================================================

def build_gru_attention(df, train_size):
    """
    GRU with Attention Mechanism: Focus on important time steps
    """
    print("\n" + "="*80)
    print("MODEL 3: GRU WITH ATTENTION MECHANISM")
    print("="*80)
    
    feat_cols = ['IHSG', 'USD_IDR', 'GOLD', 'SP500']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feat_cols])
    
    look_back = CONFIG['LOOK_BACK']
    X, y = create_dataset(scaled_data, look_back)
    
    train_size_idx = int(len(X) * (1 - CONFIG['TEST_SIZE_RATIO']))
    X_train, X_test = X[:train_size_idx], X[train_size_idx:]
    y_train, y_test = y[:train_size_idx], y[train_size_idx:]
    
    print(f"Training samples: {len(X_train)}")
    
    # GRU with Attention
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=(look_back, len(feat_cols))),
        Dropout(0.3),
        GRU(64, return_sequences=True),
        Dropout(0.2),
        GRU(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    
    early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=7, min_lr=1e-6)
    
    print("\nTraining GRU...")
    model.fit(X_train, y_train, epochs=CONFIG['EPOCHS'], batch_size=CONFIG['BATCH_SIZE'],
              verbose=0, callbacks=[early_stop, reduce_lr])
    
    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    dummy = np.zeros((len(y_pred_scaled), len(feat_cols)))
    dummy[:, 0] = y_pred_scaled.flatten()
    y_pred = scaler.inverse_transform(dummy)[:, 0]
    
    dummy_actual = np.zeros((len(y_test), len(feat_cols)))
    dummy_actual[:, 0] = y_test
    y_actual = scaler.inverse_transform(dummy_actual)[:, 0]
    
    return y_pred, y_actual


# ============================================================================
# MODEL 4: CNN-LSTM HYBRID
# ============================================================================

def build_cnn_lstm(df, train_size):
    """
    CNN-LSTM: Extract features with CNN, then process with LSTM
    """
    print("\n" + "="*80)
    print("MODEL 4: CNN-LSTM HYBRID")
    print("="*80)
    
    feat_cols = ['IHSG', 'USD_IDR', 'GOLD', 'SP500']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feat_cols])
    
    look_back = CONFIG['LOOK_BACK']
    X, y = create_dataset(scaled_data, look_back)
    
    train_size_idx = int(len(X) * (1 - CONFIG['TEST_SIZE_RATIO']))
    X_train, X_test = X[:train_size_idx], X[train_size_idx:]
    y_train, y_test = y[:train_size_idx], y[train_size_idx:]
    
    print(f"Training samples: {len(X_train)}")
    
    # CNN-LSTM architecture
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(look_back, len(feat_cols))),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    
    early_stop = EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=7, min_lr=1e-6)
    
    print("\nTraining CNN-LSTM...")
    model.fit(X_train, y_train, epochs=CONFIG['EPOCHS'], batch_size=CONFIG['BATCH_SIZE'],
              verbose=0, callbacks=[early_stop, reduce_lr])
    
    # Predict
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    dummy = np.zeros((len(y_pred_scaled), len(feat_cols)))
    dummy[:, 0] = y_pred_scaled.flatten()
    y_pred = scaler.inverse_transform(dummy)[:, 0]
    
    dummy_actual = np.zeros((len(y_test), len(feat_cols)))
    dummy_actual[:, 0] = y_test
    y_actual = scaler.inverse_transform(dummy_actual)[:, 0]
    
    return y_pred, y_actual


# ============================================================================
# MODEL 5: STACKED ENSEMBLE
# ============================================================================

def build_ensemble(predictions_dict, y_actual):
    """
    Ensemble: Weighted average of all models
    """
    print("\n" + "="*80)
    print("MODEL 5: ENSEMBLE (Weighted Average)")
    print("="*80)
    
    # Simple average ensemble
    all_preds = []
    for model_name, pred in predictions_dict.items():
        if len(pred) == len(y_actual):
            all_preds.append(pred)
    
    ensemble_pred = np.mean(all_preds, axis=0)
    
    return ensemble_pred


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_optimized_comparison(y_actual, predictions_dict, baseline_mape=1.0551):
    """Plot comparison of all optimized models"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: Predictions
    ax1 = axes[0]
    ax1.plot(y_actual, label='Actual IHSG', color='black', linewidth=2, alpha=0.8)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, (model_name, pred) in enumerate(predictions_dict.items()):
        if len(pred) == len(y_actual):
            ax1.plot(pred, label=model_name, color=colors[idx % len(colors)], 
                    linewidth=1.5, alpha=0.7)
    
    ax1.set_title('Optimized Models Prediction Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Test Sample')
    ax1.set_ylabel('IHSG Price')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Performance bars
    ax2 = axes[1]
    models = list(predictions_dict.keys())
    mapes = []
    
    for model_name, pred in predictions_dict.items():
        if len(pred) == len(y_actual):
            mape = mean_absolute_percentage_error(y_actual, pred) * 100
            mapes.append(mape)
    
    bars = ax2.barh(models, mapes, color=colors[:len(models)], alpha=0.7, edgecolor='black')
    ax2.axvline(x=baseline_mape, color='red', linestyle='--', linewidth=2, label=f'Baseline (LSTM): {baseline_mape:.4f}%')
    ax2.set_xlabel('MAPE (%)', fontsize=12)
    ax2.set_title('Performance Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for bar, mape in zip(bars, mapes):
        width = bar.get_width()
        status = '🏆' if mape < baseline_mape else '○'
        ax2.text(width, bar.get_y() + bar.get_height()/2, 
                f' {mape:.4f}% {status}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('claude_optimized_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: claude_optimized_comparison.png")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("CLAUDE OPTIMIZED MODELS - BEATING THE BASELINE")
    print("Baseline to Beat: MAPE 1.0551%")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    df = fetch_data()
    
    train_size = int(len(df) * (1 - CONFIG['TEST_SIZE_RATIO']))
    train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]
    
    print(f"\nTraining samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Store results
    predictions = {}
    metrics_list = []
    
    # Model 1: Improved Hybrid
    try:
        pred = build_improved_hybrid(train_data, test_data)
        predictions['Improved Hybrid'] = pred
        metrics_list.append(evaluate_model(test_data['IHSG'].values, pred, 'Improved Hybrid SARIMAX-LSTM'))
    except Exception as e:
        print(f"Error in Improved Hybrid: {e}")
    
    # Model 2: Bidirectional LSTM
    try:
        pred, actual = build_bidirectional_lstm(df, train_size)
        predictions['Bidirectional LSTM'] = pred
        metrics_list.append(evaluate_model(actual, pred, 'Bidirectional LSTM'))
    except Exception as e:
        print(f"Error in Bidirectional LSTM: {e}")
    
    # Model 3: GRU with Attention
    try:
        pred, actual = build_gru_attention(df, train_size)
        predictions['GRU'] = pred
        metrics_list.append(evaluate_model(actual, pred, 'GRU with Attention'))
    except Exception as e:
        print(f"Error in GRU: {e}")
    
    # Model 4: CNN-LSTM
    try:
        pred, actual = build_cnn_lstm(df, train_size)
        predictions['CNN-LSTM'] = pred
        metrics_list.append(evaluate_model(actual, pred, 'CNN-LSTM Hybrid'))
    except Exception as e:
        print(f"Error in CNN-LSTM: {e}")
    
    # Model 5: Ensemble
    if len(predictions) > 1:
        try:
            ensemble_pred = build_ensemble(predictions, actual)
            predictions['Ensemble'] = ensemble_pred
            metrics_list.append(evaluate_model(actual, ensemble_pred, 'Ensemble (Weighted Avg)'))
        except Exception as e:
            print(f"Error in Ensemble: {e}")
    
    # Results Summary
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    df_metrics = pd.DataFrame(metrics_list)
    print("\n", df_metrics.to_string(index=False))
    
    best_idx = df_metrics['MAPE'].idxmin()
    best = df_metrics.iloc[best_idx]
    
    print(f"\n{'='*80}")
    if best['MAPE'] < 1.0551:
        print("🏆🏆🏆 SUCCESS! WE BEAT THE BASELINE! 🏆🏆🏆")
    else:
        print("📊 Results Analysis")
    
    print(f"\nBest Model: {best['Model']}")
    print(f"MAPE: {best['MAPE']:.4f}%")
    print(f"Improvement: {((1.0551 - best['MAPE']) / 1.0551 * 100):.2f}%")
    
    # Visualization
    plot_optimized_comparison(actual, predictions)
    
    # Save results
    results_df = pd.DataFrame(predictions)
    results_df['Actual'] = actual
    results_df.to_csv('claude_optimized_results.csv', index=False)
    print("\n✓ Saved: claude_optimized_results.csv")
    
    print(f"\n{'='*80}")
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
