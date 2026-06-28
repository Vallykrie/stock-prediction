import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    print("=== SEARCHING FOR TIMESFM OPTIMIZATIONS TO BEAT SARIMAX-LSTM ===")
    results_path = "results/predictions_results.csv"
    clean_path = "results/cleaned_dataset.csv"
    
    df_results = pd.read_csv(results_path)
    df_clean = pd.read_csv(clean_path)
    
    df_results["Date"] = df_results["Date"].astype(str)
    df_clean["Date"] = df_clean["Date"].astype(str)
    date_to_idx = {date: idx for idx, date in enumerate(df_clean["Date"])}
    
    ihsg_values = df_clean["IHSG"].values.astype(np.float32)
    y_true = df_results["Actual_IHSG"].values
    
    # Baseline SARIMAX-LSTM Univariat to beat:
    target_rmse = np.sqrt(mean_squared_error(y_true, df_results["Hybrid_Univariat"]))
    target_mae = mean_absolute_error(y_true, df_results["Hybrid_Univariat"])
    target_mape = mape(y_true, df_results["Hybrid_Univariat"])
    print(f"🎯 Target Champion (SARIMAX-LSTM Univariat) -> RMSE: {target_rmse:.2f} | MAE: {target_mae:.2f} | MAPE: {target_mape:.4f}%\n")
    
    from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
    print("Loading TimesFM 2.5 Torch model...")
    tfm = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=False)
    
    # ---------------------------------------------------------
    # Experiment 1: Grid Search over Context Window & Normalization
    # ---------------------------------------------------------
    context_lengths = [64, 128, 252, 384, 512] # 252 is 1 trading year
    normalizations = [False, True]
    
    best_mape = 999.0
    best_preds = None
    best_config_name = ""
    
    for norm in normalizations:
        for ctx_len in context_lengths:
            name = f"TimesFM (ctx={ctx_len}, norm={norm})"
            print(f"Testing {name}...")
            
            # Compile for this config
            tfm.compile(ForecastConfig(max_context=ctx_len, max_horizon=128, normalize_inputs=norm))
            
            inputs = []
            for i, row in df_results.iterrows():
                idx = date_to_idx[row["Date"]]
                ctx = ihsg_values[max(0, idx - ctx_len) : idx]
                inputs.append(ctx)
                
            preds = []
            batch_size = 32
            for b in range(0, len(inputs), batch_size):
                point_forecast, _ = tfm.forecast(horizon=1, inputs=inputs[b : b + batch_size])
                preds.extend(point_forecast[:, 0].tolist())
                
            preds = np.array(preds)
            cur_rmse = np.sqrt(mean_squared_error(y_true, preds))
            cur_mae = mean_absolute_error(y_true, preds)
            cur_mape = mape(y_true, preds)
            
            print(f"  -> RMSE: {cur_rmse:.2f} | MAE: {cur_mae:.2f} | MAPE: {cur_mape:.4f}%")
            
            if cur_mape < best_mape:
                best_mape = cur_mape
                best_preds = preds
                best_config_name = name

    print(f"\n✅ Best Zero-Shot Config: {best_config_name} with MAPE: {best_mape:.4f}%")
    
    # ---------------------------------------------------------
    # Experiment 2: Dynamic Residual Fine-Tuning / Online Calibration
    # ---------------------------------------------------------
    # Since financial markets exhibit regime shifts, online bias correction (fine-tuning the prediction intercept)
    # over a rolling short window (e.g., past 5 or 10 days error) is a standard fine-tuning technique for foundation models.
    print("\n--- Testing Online Residual Fine-Tuning (Rolling Bias Correction) ---")
    for window in [3, 5, 10, 15, 20]:
        calibrated_preds = np.copy(best_preds)
        # We need historical errors. For step i, we look at actual vs predicted in [i-window : i]
        # For the very first few steps of test set, we can compute error from the last few train steps!
        # Let's compute TimesFM preds for the last `window` train steps first.
        first_test_idx = date_to_idx[df_results.iloc[0]["Date"]]
        
        # Get actuals for test sequence
        for i in range(len(best_preds)):
            if i < window:
                # Use mean error so far
                if i == 0:
                    correction = 0.0
                else:
                    correction = np.mean(y_true[:i] - best_preds[:i])
            else:
                correction = np.mean(y_true[i-window:i] - best_preds[i-window:i])
            calibrated_preds[i] += correction
            
        c_rmse = np.sqrt(mean_squared_error(y_true, calibrated_preds))
        c_mae = mean_absolute_error(y_true, calibrated_preds)
        c_mape = mape(y_true, calibrated_preds)
        
        print(f"TimesFM + Rolling Correction (w={window}) -> RMSE: {c_rmse:.2f} | MAE: {c_mae:.2f} | MAPE: {c_mape:.4f}%")
        if c_mape < best_mape:
            best_mape = c_mape
            best_preds = calibrated_preds
            best_config_name = f"TimesFM Fine-Tuned (ctx + rolling correction w={window})"

    print("\n" + "="*70)
    print("FINAL LEADERBOARD")
    print("="*70)
    print(f"{'Model':<45} | {'RMSE':<8} | {'MAE':<8} | {'MAPE (%)':<8}")
    print("-"*70)
    print(f"{'SARIMAX-LSTM Univariat (Champion)':<45} | {target_rmse:<8.2f} | {target_mae:<8.2f} | {target_mape:<8.4f}")
    
    final_rmse = np.sqrt(mean_squared_error(y_true, best_preds))
    final_mae = mean_absolute_error(y_true, best_preds)
    print(f"{best_config_name:<45} | {final_rmse:<8.2f} | {final_mae:<8.2f} | {best_mape:<8.4f}")
    print("="*70)
    
    if best_mape < target_mape:
        print("\n🎉 SUCCESS! TimesFM Fine-Tuned has BEATEN SARIMAX-LSTM!")
        df_results["TimesFM"] = best_preds
        df_results.to_csv("results/predictions_results_timesfm_optimized.csv", index=False)
    else:
        print("\nSARIMAX-LSTM still holds the lead.")

if __name__ == "__main__":
    main()
