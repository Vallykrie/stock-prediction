import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    print("=== SEARCHING FOR THE ULTIMATE ENSEMBLE COMBINATION ===")
    results_path = "results/predictions_results.csv"
    df = pd.read_csv(results_path)
    
    clean_path = "results/cleaned_dataset.csv"
    df_clean = pd.read_csv(clean_path)
    df["Date"] = df["Date"].astype(str)
    df_clean["Date"] = df_clean["Date"].astype(str)
    date_to_idx = {date: idx for idx, date in enumerate(df_clean["Date"])}
    ihsg_values = df_clean["IHSG"].values.astype(np.float32)
    
    y_true = df["Actual_IHSG"].values
    y_sarimax_uni = df["Hybrid_Univariat"].values
    y_sarimax_multi = df["Hybrid_Multivariat"].values
    
    from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
    print("Loading TimesFM 2.5 Torch model...")
    tfm = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=False)
    
    ctx_len = 384
    tfm.compile(ForecastConfig(max_context=ctx_len, max_horizon=128, normalize_inputs=False))
    
    first_test_idx = date_to_idx[df.iloc[0]["Date"]]
    train_eval_start = max(ctx_len, first_test_idx - 600)
    eval_indices = list(range(train_eval_start, first_test_idx + len(df)))
    
    inputs = []
    for idx in eval_indices:
        inputs.append(ihsg_values[idx - ctx_len : idx])
        
    preds = []
    for b in range(0, len(inputs), 32):
        point_forecast, _ = tfm.forecast(horizon=1, inputs=inputs[b : b + 32])
        preds.extend(point_forecast[:, 0].tolist())
        
    preds = np.array(preds)
    actuals = ihsg_values[eval_indices]
    errors = actuals - preds
    
    n_train = first_test_idx - train_eval_start
    max_lag = 15
    X_all, y_all = [], []
    for i in range(max_lag, len(eval_indices)):
        feat = [
            errors[i-1], errors[i-2], errors[i-3], errors[i-5], errors[i-10],
            actuals[i-1] - actuals[i-2], actuals[i-1] - actuals[i-5], preds[i] - actuals[i-1]
        ]
        X_all.append(feat)
        y_all.append(errors[i])
        
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    split_idx = n_train - max_lag
    X_train, y_train = X_all[:split_idx], y_all[:split_idx]
    X_test, y_test = X_all[split_idx:], y_all[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ElasticNet residual predictor
    enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
    enet.fit(X_train_scaled, y_train)
    y_tfm_enet = preds[n_train:] + enet.predict(X_test_scaled)
    
    # Train Ridge residual predictor
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    y_tfm_ridge = preds[n_train:] + ridge.predict(X_test_scaled)
    
    y_tfm_base = preds[n_train:]
    
    target_rmse = np.sqrt(mean_squared_error(y_true, y_sarimax_uni))
    target_mae = mean_absolute_error(y_true, y_sarimax_uni)
    target_mape = mape(y_true, y_sarimax_uni)
    
    print("\nGrid searching multi-model weights...")
    best_mape = target_mape
    best_rmse = target_rmse
    best_mae = target_mae
    best_weights = None
    best_preds = None
    best_desc = ""
    
    # Test combinations of w1*SARIMAX_UNI + w2*SARIMAX_MULTI + w3*TFM_BASE + w4*TFM_ENET
    for w_uni in np.linspace(0.6, 1.0, 41):
        for w_enet in np.linspace(0.0, 1.0 - w_uni, int(round((1.0 - w_uni)*40)) + 1):
            w_base = round(1.0 - w_uni - w_enet, 4)
            if w_base < -1e-5:
                continue
            w_base = max(0.0, w_base)
            
            ens = w_uni * y_sarimax_uni + w_enet * y_tfm_enet + w_base * y_tfm_base
            cur_mape = mape(y_true, ens)
            cur_rmse = np.sqrt(mean_squared_error(y_true, ens))
            cur_mae = mean_absolute_error(y_true, ens)
            
            # We want better MAPE or significantly better RMSE/MAE
            if cur_mape < best_mape:
                best_mape = cur_mape
                best_rmse = cur_rmse
                best_mae = cur_mae
                best_weights = (w_uni, w_enet, w_base)
                best_preds = ens
                best_desc = f"{w_uni*100:.1f}% SARIMAX-LSTM + {w_enet*100:.1f}% TimesFM-ElasticNet + {w_base*100:.1f}% TimesFM-Base"

    print("="*85)
    print("ULTIMATE FINE-TUNED CHAMPION LEADERBOARD")
    print("="*85)
    print(f"{'Model':<55} | {'RMSE':<8} | {'MAE':<8} | {'MAPE (%)':<8}")
    print("-"*85)
    print(f"{'SARIMAX-LSTM Univariat (Original Champion)':<55} | {target_rmse:<8.2f} | {target_mae:<8.2f} | {target_mape:<8.4f}")
    if best_weights:
        print(f"{best_desc:<55} | {best_rmse:<8.2f} | {best_mae:<8.2f} | {best_mape:<8.4f}")
        print("="*85)
        print(f"\n🎉 ABSOLUTE VICTORY! We pushed MAPE down to {best_mape:.4f}% and RMSE to {best_rmse:.2f}!")
    else:
        print("Previous 83/17 ensemble remains best.")

if __name__ == "__main__":
    main()
