import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
import pmdarima as pm

warnings.filterwarnings("ignore")

# --- Configuration ---
START_DATE = '2019-01-01'
END_DATE = '2024-12-31'
TARGET_TICKER = '^JKSE'  # IHSG
EXOG_TICKERS = {
    'USD_IDR': 'IDR=X',
    'GOLD': 'GC=F',
    'SP500': '^GSPC'
}
TEST_SIZE_RATIO = 0.2

def fetch_data():
    print("Fetching data...")
    tickers = [TARGET_TICKER] + list(EXOG_TICKERS.values())
    data = yf.download(tickers, start=START_DATE, end=END_DATE)['Close']
    
    # Rename columns
    data = data.rename(columns={
        TARGET_TICKER: 'IHSG',
        EXOG_TICKERS['USD_IDR']: 'USD_IDR',
        EXOG_TICKERS['GOLD']: 'GOLD',
        EXOG_TICKERS['SP500']: 'SP500'
    })
    
    # Handle missing values (Interpolation)
    data = data.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    return data

def check_stationarity(series, name):
    result = adfuller(series)
    print(f'ADF Statistic for {name}: {result[0]}')
    print(f'p-value: {result[1]}')
    return result[1] < 0.05

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f"--- {model_name} Evaluation ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}")
    return rmse, mae, mape

def run_hybrid_model():
    # 1. Data Acquisition & Preprocessing
    df = fetch_data()
    
    # Train-Test Split
    train_size = int(len(df) * (1 - TEST_SIZE_RATIO))
    train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]
    
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    
    # 2. SARIMAX Model (Linear Component)
    print("\nTraining SARIMAX Model...")
    
    # Auto-ARIMA to find best parameters
    # We use exogenous variables for SARIMAX
    exog_train = train_data[['USD_IDR', 'GOLD', 'SP500']]
    exog_test = test_data[['USD_IDR', 'GOLD', 'SP500']]
    
    # Note: Auto-arima can be slow. For demonstration, we might use fixed order or a smaller search.
    # Using a simple grid search or auto_arima from pmdarima
    model_sarimax = pm.auto_arima(train_data['IHSG'], exogenous=exog_train,
                                  start_p=1, start_q=1,
                                  max_p=3, max_q=3, m=12,
                                  start_P=0, seasonal=False, # Assuming non-seasonal for daily data for simplicity or set seasonal=True
                                  d=None, D=1, trace=True,
                                  error_action='ignore',  
                                  suppress_warnings=True, 
                                  stepwise=True)

    print(model_sarimax.summary())
    
    # Predict with SARIMAX
    sarimax_pred_train = model_sarimax.predict_in_sample(exogenous=exog_train)
    sarimax_pred_test = model_sarimax.predict(n_periods=len(test_data), exogenous=exog_test)
    
    # 3. Residuals Calculation
    train_residuals = train_data['IHSG'] - sarimax_pred_train
    test_residuals = test_data['IHSG'] - sarimax_pred_test # Ideally we don't know test residuals, but for hybrid we predict them.
    # Wait, for hybrid, we train LSTM on TRAIN residuals, then predict TEST residuals.
    
    # 4. LSTM Model (Non-Linear Component on Residuals)
    print("\nTraining LSTM on Residuals...")
    
    # Scale residuals
    scaler = MinMaxScaler(feature_range=(-1, 1)) # Residuals can be negative
    train_residuals_scaled = scaler.fit_transform(train_residuals.values.reshape(-1, 1))
    
    # Create sequences
    look_back = 60
    X_train_res, y_train_res = create_dataset(train_residuals_scaled, look_back)
    
    # Reshape input to be [samples, time steps, features]
    X_train_res = np.reshape(X_train_res, (X_train_res.shape[0], 1, X_train_res.shape[1]))
    
    # Build LSTM
    model_lstm = Sequential()
    model_lstm.add(LSTM(50, input_shape=(1, look_back)))
    model_lstm.add(Dense(1))
    model_lstm.compile(loss='mean_squared_error', optimizer='adam')
    model_lstm.fit(X_train_res, y_train_res, epochs=50, batch_size=32, verbose=1)
    
    # Predict Residuals for Test Set
    # We need to construct the input for test prediction carefully
    # We need the last 'look_back' residuals from train set to start predicting first test point?
    # Or we use the residuals calculated from SARIMAX test predictions?
    # Strictly speaking, in a real forecasting scenario, we predict SARIMAX for t+1, then predict Residual for t+1 using past known residuals.
    # For the test set, if we do one-step ahead, we can use actual past residuals.
    # If we do multi-step, we use predicted residuals.
    # Here, let's assume we want to correct the SARIMAX trend.
    # We will generate inputs for LSTM from the full residual series (train + test 'proxy')
    # But wait, we can't use test_residuals (actual - sarimax_pred) because we don't know actual in future.
    # We must use the *predicted* residuals? No, LSTM predicts the residual.
    # Input to LSTM is *past* residuals.
    # So for Test point 0, we need Train residuals [-look_back:].
    
    # Concatenate train and test residuals (using SARIMAX predictions vs Actual for test is cheating?)
    # No, we use the residuals from the training phase to predict the next residual.
    # But for the test set, we don't have "actual" residuals to feed back in if we are doing pure forecasting.
    # However, usually in these papers, they do one-step ahead prediction or use the error from the previous step (if available).
    # Let's assume one-step ahead forecasting where we know the previous actual values (and thus previous residuals).
    
    full_residuals = pd.concat([train_residuals, test_data['IHSG'] - sarimax_pred_test]) # This is "actual" residuals if we knew IHSG
    # Actually, standard practice for these hybrid models often uses the past known residuals.
    # If we are testing on historical data, we *do* know the past residuals at each step.
    
    full_residuals_scaled = scaler.transform(full_residuals.values.reshape(-1, 1))
    
    test_inputs = full_residuals_scaled[len(full_residuals_scaled) - len(test_data) - look_back:]
    X_test_res = []
    for i in range(len(test_data)):
        X_test_res.append(test_inputs[i:(i + look_back), 0])
    X_test_res = np.array(X_test_res)
    X_test_res = np.reshape(X_test_res, (X_test_res.shape[0], 1, X_test_res.shape[1]))
    
    lstm_pred_res_scaled = model_lstm.predict(X_test_res)
    lstm_pred_res = scaler.inverse_transform(lstm_pred_res_scaled)
    
    # 5. Hybrid Prediction
    # Hybrid = SARIMAX_Pred + LSTM_Pred_Residual
    hybrid_pred = sarimax_pred_test.values + lstm_pred_res.flatten()
    
    # --- Evaluation ---
    evaluate_model(test_data['IHSG'], hybrid_pred, "Hybrid SARIMAX-LSTM")
    
    # --- Comparison Models ---
    
    # A. Single LSTM (Multivariate)
    print("\nTraining Single LSTM (Multivariate)...")
    # Prepare data for Multivariate LSTM
    # Inputs: IHSG, USD_IDR, GOLD, SP500
    # Target: IHSG
    
    feat_cols = ['IHSG', 'USD_IDR', 'GOLD', 'SP500']
    scaler_multi = MinMaxScaler()
    scaled_data = scaler_multi.fit_transform(df[feat_cols])
    
    X_multi, y_multi = [], []
    for i in range(len(scaled_data) - look_back - 1):
        X_multi.append(scaled_data[i:(i + look_back)])
        y_multi.append(scaled_data[i + look_back, 0]) # 0 is IHSG
        
    X_multi, y_multi = np.array(X_multi), np.array(y_multi)
    
    # Split
    train_size_lstm = int(len(X_multi) * (1 - TEST_SIZE_RATIO))
    X_train_m, X_test_m = X_multi[:train_size_lstm], X_multi[train_size_lstm:]
    y_train_m, y_test_m = y_multi[:train_size_lstm], y_multi[train_size_lstm:]
    
    model_single_lstm = Sequential()
    model_single_lstm.add(LSTM(50, input_shape=(look_back, len(feat_cols))))
    model_single_lstm.add(Dense(1))
    model_single_lstm.compile(loss='mean_squared_error', optimizer='adam')
    model_single_lstm.fit(X_train_m, y_train_m, epochs=50, batch_size=32, verbose=0)
    
    single_lstm_pred_scaled = model_single_lstm.predict(X_test_m)
    
    # Inverse transform
    # We need to inverse transform only the target column. 
    # Create a dummy array to inverse transform
    dummy_pred = np.zeros((len(single_lstm_pred_scaled), len(feat_cols)))
    dummy_pred[:, 0] = single_lstm_pred_scaled.flatten()
    single_lstm_pred = scaler_multi.inverse_transform(dummy_pred)[:, 0]
    
    # Align y_test_m for evaluation
    dummy_actual = np.zeros((len(y_test_m), len(feat_cols)))
    dummy_actual[:, 0] = y_test_m
    y_test_m_actual = scaler_multi.inverse_transform(dummy_actual)[:, 0]
    
    evaluate_model(y_test_m_actual, single_lstm_pred, "Single LSTM (Multivariate)")
    
    # --- Visualization ---
    plt.figure(figsize=(14, 7))
    plt.plot(test_data.index, test_data['IHSG'], label='Actual IHSG', color='black')
    plt.plot(test_data.index, hybrid_pred, label='Hybrid SARIMAX-LSTM', color='blue')
    # Align LSTM dates (might be slightly off due to look_back)
    lstm_dates = test_data.index[-len(single_lstm_pred):]
    plt.plot(lstm_dates, single_lstm_pred, label='Single LSTM', color='green', linestyle='--')
    
    plt.title('IHSG Prediction Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('prediction_comparison.png')
    print("Plot saved as prediction_comparison.png")

if __name__ == "__main__":
    run_hybrid_model()
