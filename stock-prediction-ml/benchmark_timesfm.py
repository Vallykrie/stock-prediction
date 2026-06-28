import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    print("Loading datasets...")
    results_path = "results/predictions_results.csv"
    clean_path = "results/cleaned_dataset.csv"
    
    df_results = pd.read_csv(results_path)
    df_clean = pd.read_csv(clean_path)
    
    # Ensure date columns are strings for matching
    df_results["Date"] = df_results["Date"].astype(str)
    df_clean["Date"] = df_clean["Date"].astype(str)
    
    # Map Date to index in cleaned_dataset
    date_to_idx = {date: idx for idx, date in enumerate(df_clean["Date"])}
    
    print(f"Total test steps to evaluate: {len(df_results)}")
    
    # Prepare input context windows for each test date
    # TimesFM 2.5 supports context lengths up to 2048. Let's use 512 trading days (~2 years) as lookback context.
    context_len = 512
    inputs = []
    valid_indices = []
    
    ihsg_values = df_clean["IHSG"].values.astype(np.float32)
    
    for i, row in df_results.iterrows():
        date = row["Date"]
        if date not in date_to_idx:
            continue
        idx = date_to_idx[date]
        if idx < context_len:
            # Fallback if idx < context_len
            ctx = ihsg_values[:idx]
        else:
            ctx = ihsg_values[idx - context_len : idx]
        inputs.append(ctx)
        valid_indices.append(i)
        
    print(f"Prepared {len(inputs)} context windows.")
    
    print("Initializing TimesFM 2.5 Torch model...")
    start_time = time.time()
    from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
    
    tfm = TimesFM_2p5_200M_torch.from_pretrained(
        "google/timesfm-2.5-200m-pytorch",
        torch_compile=False # Disable compile for faster startup on local macOS
    )
    print("Compiling model for forecasting...")
    tfm.compile(ForecastConfig(max_context=512, max_horizon=128))
    print(f"Model loaded and compiled in {time.time() - start_time:.2f}s.")
    
    print("Running batch forecasting...")
    forecast_start = time.time()
    
    # Run forecast in batches of 32 to avoid OOM or slowdowns
    batch_size = 32
    preds = []
    for b in range(0, len(inputs), batch_size):
        batch_inputs = inputs[b : b + batch_size]
        point_forecast, _ = tfm.forecast(horizon=1, inputs=batch_inputs)
        # point_forecast shape: (batch_size, horizon)
        preds.extend(point_forecast[:, 0].tolist())
        print(f"  Forecasted {min(b + batch_size, len(inputs))}/{len(inputs)} steps...")
        
    print(f"Forecasting finished in {time.time() - forecast_start:.2f}s.")
    
    df_results["TimesFM"] = np.nan
    df_results.loc[valid_indices, "TimesFM"] = preds
    
    # Evaluate accuracy
    y_true = df_results["Actual_IHSG"].values
    y_tfm = df_results["TimesFM"].values
    
    rmse_tfm = np.sqrt(mean_squared_error(y_true, y_tfm))
    mae_tfm = mean_absolute_error(y_true, y_tfm)
    mape_tfm = mean_absolute_percentage_error(y_true, y_tfm)
    
    print("\n" + "="*60)
    print("ACCURACY BENCHMARK COMPARISON (Test Set: 427 steps)")
    print("="*60)
    print(f"{'Model':<35} | {'RMSE':<8} | {'MAE':<8} | {'MAPE (%)':<8}")
    print("-"*60)
    
    # Print existing models
    for col, name in [
        ("Hybrid_Univariat", "Hybrid SARIMAX-LSTM Univariat"),
        ("Hybrid_Multivariat", "Hybrid SARIMAX-LSTM Multivariat"),
        ("LSTM_Tunggal", "LSTM Tunggal")
    ]:
        rmse = np.sqrt(mean_squared_error(y_true, df_results[col]))
        mae = mean_absolute_error(y_true, df_results[col])
        mape = mean_absolute_percentage_error(y_true, df_results[col])
        print(f"{name:<35} | {rmse:<8.2f} | {mae:<8.2f} | {mape:<8.4f}")
        
    print(f"{'Google TimesFM 2.5 (Zero-Shot)':<35} | {rmse_tfm:<8.2f} | {mae_tfm:<8.2f} | {mape_tfm:<8.4f}")
    print("="*60)
    
    # Save test prediction for analysis
    df_results.to_csv("results/benchmark_timesfm_temp.csv", index=False)
    print("\nTemporary predictions saved to results/benchmark_timesfm_temp.csv")

if __name__ == "__main__":
    main()
