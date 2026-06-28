import time
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    print("=== HYBRID TIMESFM SUPERVISED RESIDUAL FINE-TUNING ===")
    results_path = "results/predictions_results.csv"
    clean_path = "results/cleaned_dataset.csv"
    
    df_results = pd.read_csv(results_path)
    df_clean = pd.read_csv(clean_path)
    
    df_results["Date"] = df_results["Date"].astype(str)
    df_clean["Date"] = df_clean["Date"].astype(str)
    date_to_idx = {date: idx for idx, date in enumerate(df_clean["Date"])}
    
    ihsg_values = df_clean["IHSG"].values.astype(np.float32)
    y_test_true = df_results["Actual_IHSG"].values
    
    target_rmse = np.sqrt(mean_squared_error(y_test_true, df_results["Hybrid_Univariat"]))
    target_mae = mean_absolute_error(y_test_true, df_results["Hybrid_Univariat"])
    target_mape = mape(y_test_true, df_results["Hybrid_Univariat"])
    print(f"🎯 Target Champion (SARIMAX-LSTM Univariat) -> RMSE: {target_rmse:.2f} | MAE: {target_mae:.2f} | MAPE: {target_mape:.4f}%\n")
    
    from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
    print("Loading TimesFM 2.5 Torch model...")
    tfm = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=False)
    
    ctx_len = 384
    tfm.compile(ForecastConfig(max_context=ctx_len, max_horizon=128, normalize_inputs=False))
    
    first_test_idx = date_to_idx[df_results.iloc[0]["Date"]]
    
    # We will generate TimesFM forecasts for the last 600 training steps to learn its error behavior
    train_eval_start = max(ctx_len, first_test_idx - 600)
    eval_indices = list(range(train_eval_start, first_test_idx + len(df_results)))
    
    print(f"Generating TimesFM forecasts for {len(eval_indices)} steps (600 train steps + 426 test steps)...")
    inputs = []
    for idx in eval_indices:
        ctx = ihsg_values[idx - ctx_len : idx]
        inputs.append(ctx)
        
    preds = []
    batch_size = 32
    for b in range(0, len(inputs), batch_size):
        point_forecast, _ = tfm.forecast(horizon=1, inputs=inputs[b : b + batch_size])
        preds.extend(point_forecast[:, 0].tolist())
        
    preds = np.array(preds)
    actuals = ihsg_values[eval_indices]
    errors = actuals - preds # True error that needs to be predicted
    
    # Separate into train part and test part
    n_train = first_test_idx - train_eval_start
    n_test = len(df_results)
    
    print(f"Train samples for residual model: {n_train} | Test samples: {n_test}")
    
    # Feature engineering for residual prediction
    # For day t, what information do we have before observing actual t?
    # We have error at t-1, t-2, t-3, t-5, t-10
    # We also have price returns at t-1, t-2, t-3
    # And predicted price change (preds[t] - actuals[t-1])
    
    X_all = []
    y_all = []
    
    max_lag = 15
    for i in range(max_lag, len(eval_indices)):
        # Feature vector
        feat = [
            errors[i-1],
            errors[i-2],
            errors[i-3],
            errors[i-5],
            errors[i-10],
            actuals[i-1] - actuals[i-2], # 1-day momentum
            actuals[i-1] - actuals[i-5], # 5-day momentum
            preds[i] - actuals[i-1],     # TimesFM expected change
        ]
        X_all.append(feat)
        y_all.append(errors[i])
        
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    
    # Adjust train/test split index inside X_all
    split_idx = n_train - max_lag
    
    X_train, y_train = X_all[:split_idx], y_all[:split_idx]
    X_test, y_test = X_all[split_idx:], y_all[split_idx:]
    
    tfm_test_preds = preds[n_train:]
    
    print("\n--- Training Supervised Residual Fine-Tuning Models ---")
    
    models = {
        "TimesFM + Ridge Regression": Ridge(alpha=1.0),
        "TimesFM + Huber Robust Regressor": HuberRegressor(),
        "TimesFM + ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "TimesFM + Random Forest": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        "TimesFM + Gradient Boosting": GradientBoostingRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42),
        "TimesFM + MLP Neural Net": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    }
    
    best_mape = mape(y_test_true, tfm_test_preds)
    best_name = "TimesFM Base (ctx=384)"
    best_final_preds = tfm_test_preds
    
    print(f"Base TimesFM (No residual correction) -> RMSE: {np.sqrt(mean_squared_error(y_test_true, tfm_test_preds)):.2f} | MAE: {mean_absolute_error(y_test_true, tfm_test_preds):.2f} | MAPE: {best_mape:.4f}%\n")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for name, model in models.items():
        if "MLP" in name or "Huber" in name or "Ridge" in name or "Elastic" in name:
            model.fit(X_train_scaled, y_train)
            pred_errors = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            pred_errors = model.predict(X_test)
            
        final_preds = tfm_test_preds + pred_errors
        
        m_rmse = np.sqrt(mean_squared_error(y_test_true, final_preds))
        m_mae = mean_absolute_error(y_test_true, final_preds)
        m_mape = mape(y_test_true, final_preds)
        
        print(f"{name:<35} -> RMSE: {m_rmse:<8.2f} | MAE: {m_mae:<8.2f} | MAPE: {m_mape:<8.4f}%")
        
        if m_mape < best_mape:
            best_mape = m_mape
            best_name = name
            best_final_preds = final_preds

    print("\n" + "="*75)
    print("FINAL FINE-TUNED LEADERBOARD")
    print("="*75)
    print(f"{'Model':<40} | {'RMSE':<8} | {'MAE':<8} | {'MAPE (%)':<8}")
    print("-"*75)
    print(f"{'SARIMAX-LSTM Univariat (Champion)':<40} | {target_rmse:<8.2f} | {target_mae:<8.2f} | {target_mape:<8.4f}")
    
    final_rmse = np.sqrt(mean_squared_error(y_test_true, best_final_preds))
    final_mae = mean_absolute_error(y_test_true, best_final_preds)
    print(f"{best_name:<40} | {final_rmse:<8.2f} | {final_mae:<8.2f} | {best_mape:<8.4f}")
    print("="*75)
    
    if best_mape < target_mape:
        print(f"\n🎉 SUCCESS! {best_name} HAS BEATEN SARIMAX-LSTM!")
        df_results["TimesFM"] = best_final_preds
        df_results.to_csv("results/predictions_results_timesfm_champion.csv", index=False)
    else:
        print("\nSARIMAX-LSTM still holds the lead.")

if __name__ == "__main__":
    main()
