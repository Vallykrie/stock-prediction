"""Challenger: ensemble regresi return (Ridge + Huber + HistGradientBoosting).

Ide inti: S&P 500 tutup dini hari WIB SEBELUM IHSG tutup, sehingga return
S&P 500 kemarin adalah prediktor sah (tanpa leakage) untuk return IHSG hari
ini (korelasi ~0.27). Model memprediksi log-return harian lalu dikonversi ke
harga: pred = close_kemarin * exp(pred_return * shrink).

Protokol jujur:
- Fit HANYA pada data sebelum test set (test = periode predictions_results.csv).
- Semua hyperparameter (fitur, model, shrink) dipilih via validasi pada
  426 hari terakhir data training — test set tidak pernah dilihat saat tuning.

Hasil (test 426 hari, 2024-09-12 s/d 2026-04-30):
                                  RMSE     MAE    MAPE
  Challenger Ensemble            90.88   61.34  0.8307%
  Hybrid SARIMAX-LSTM Univariat  92.62   61.61  0.8343%  (juara lama)
  Naive (close kemarin)          92.49   61.41  0.8315%
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

SHRINK = 0.8  # dipilih via validasi: meredam overconfidence prediksi return


def main():
    clean = pd.read_csv("results/cleaned_dataset.csv")
    res = pd.read_csv("results/predictions_results.csv")
    test_dates = set(res["Date"].astype(str))
    y_true = res["Actual_IHSG"].values

    px = clean[["IHSG", "USD_IDR", "Gold", "SP500"]].values
    dates = clean["Date"].astype(str).values
    ihsg = clean["IHSG"].values
    lr = np.diff(np.log(px), axis=0)  # lr[i] = return dari hari i ke i+1

    # Fitur pada hari t memakai return s/d hari t-1 saja (bebas leakage).
    W = 22
    rows, ys, didx = [], [], []
    for i in range(W, len(lr)):
        ih, fx, gd, sp = (lr[:i, k] for k in range(4))
        rows.append([
            ih[-1], ih[-2], sp[-1], sp[-2], gd[-1], fx[-1],
            ih[-5:].sum(), sp[-5:].sum(), ih[-22:].sum(),
            ih[-22:].std(), sp[-22:].std(),
        ])
        ys.append(lr[i, 0])
        didx.append(i + 1)
    X, y, didx = np.array(rows), np.array(ys), np.array(didx)

    is_test = np.array([d in test_dates for d in dates[didx]])
    ft = int(np.argmax(is_test))
    assert is_test[ft:].all() and not is_test[:ft].any()

    models = [
        Ridge(alpha=1.0),
        HuberRegressor(alpha=1e-4, max_iter=1000),
        HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.03, max_iter=300,
            min_samples_leaf=100, random_state=0,
        ),
    ]
    pred_r = np.mean(
        [m.fit(X[:ft], y[:ft]).predict(X[ft:]) for m in models], axis=0
    ) * SHRINK
    pred_px = ihsg[didx[ft:] - 1] * np.exp(pred_r)

    def report(name, p):
        rmse = np.sqrt(mean_squared_error(y_true, p))
        mae = mean_absolute_error(y_true, p)
        mape = np.mean(np.abs((y_true - p) / y_true)) * 100
        print(f"{name:<35} RMSE={rmse:7.2f}  MAE={mae:7.2f}  MAPE={mape:.4f}%")
        return rmse, mae, mape

    print(f"Test set: {len(y_true)} hari ({res['Date'].iloc[0]} s/d {res['Date'].iloc[-1]})\n")
    champ = report("Hybrid SARIMAX-LSTM Univariat", res["Hybrid_Univariat"].values)
    report("Naive (close kemarin)", ihsg[didx[ft:] - 1])
    ours = report("Challenger Ensemble", pred_px)

    # Diebold-Mariano sederhana: paired t-test pada selisih squared error
    d = (y_true - res["Hybrid_Univariat"].values) ** 2 - (y_true - pred_px) ** 2
    t, pval = stats.ttest_1samp(d, 0)
    print(f"\nPaired t-test squared errors vs juara: t={t:.2f}, p={pval:.4f}")

    assert ours[0] < champ[0] and ours[1] < champ[1] and ours[2] < champ[2], \
        "Challenger gagal mengalahkan juara di salah satu metrik"
    print("✅ Challenger mengalahkan juara di RMSE, MAE, dan MAPE.")

    out = res[["Date", "Actual_IHSG", "Hybrid_Univariat"]].copy()
    out["Challenger_Ensemble"] = pred_px
    out.to_csv("results/predictions_challenger.csv", index=False)

    metrics = pd.read_csv("results/evaluation_metrics.csv")
    metrics = metrics[metrics["Model"] != "Challenger Ensemble"]
    metrics.loc[len(metrics)] = ["Challenger Ensemble", ours[0], ours[1], ours[2]]
    metrics.to_csv("results/evaluation_metrics.csv", index=False)
    print("Tersimpan: results/predictions_challenger.csv, results/evaluation_metrics.csv")


if __name__ == "__main__":
    main()
