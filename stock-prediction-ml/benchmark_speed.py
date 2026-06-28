import time
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

def main():
    print("=== BENCHMARKING SINGLE-STEP INFERENCE SPEED (LATENCY) ===")
    
    clean_path = "results/cleaned_dataset.csv"
    df_clean = pd.read_csv(clean_path)
    ihsg_values = df_clean["IHSG"].values.astype(np.float32)
    
    # 1. Measure TimesFM 2.5 Torch Single-Step Inference
    from timesfm import TimesFM_2p5_200M_torch, ForecastConfig
    print("Loading TimesFM...")
    tfm = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=False)
    tfm.compile(ForecastConfig(max_context=384, max_horizon=128, normalize_inputs=False))
    
    # Warmup
    ctx = ihsg_values[-384:]
    tfm.forecast(horizon=1, inputs=[ctx])
    
    print("Measuring TimesFM latency over 50 individual iterations...")
    start_time = time.time()
    for _ in range(50):
        tfm.forecast(horizon=1, inputs=[ctx])
    tfm_total_time = time.time() - start_time
    tfm_ms_per_step = (tfm_total_time / 50.0) * 1000.0
    print(f"TimesFM Base -> {tfm_ms_per_step:.2f} ms / step")
    
    # 2. Measure ElasticNet overhead
    X_dummy = np.random.randn(100, 8)
    y_dummy = np.random.randn(100)
    scaler = StandardScaler().fit(X_dummy)
    enet = ElasticNet(alpha=0.1).fit(scaler.transform(X_dummy), y_dummy)
    
    start_time = time.time()
    for _ in range(1000):
        feat = np.random.randn(1, 8)
        enet.predict(scaler.transform(feat))
    enet_ms_per_step = ((time.time() - start_time) / 1000.0) * 1000.0
    print(f"ElasticNet Residual Layer -> {enet_ms_per_step:.4f} ms / step")
    
    print("\nSUMMARY ESTIMATES FOR BACKEND REAL-TIME COMPUTING:")
    print(f"1. SARIMAX-LSTM Univariat (Statistical + Small LSTM): ~5.00 - 15.00 ms")
    print(f"2. Google TimesFM Base (200M Transformer): {tfm_ms_per_step:.2f} ms")
    print(f"3. TimesFM + ElasticNet: {tfm_ms_per_step + enet_ms_per_step:.2f} ms")
    print(f"4. Ultra Hybrid Ensemble (All combined): ~{tfm_ms_per_step + 15.0:.2f} ms")

if __name__ == "__main__":
    main()
