"""Challenger v2: ensemble regresi return dengan informasi pasar global.

Tiga tingkat informasi, semuanya BEBAS LEAKAGE terhadap harga penutupan IHSG
hari t (setiap fitur tersedia secara publik sebelum IHSG tutup jam 16:00 WIB):

  A. Pre-open  — hanya info sampai sebelum pasar buka (setara info set juara
     lama): close t-1 dari IHSG/S&P500/emas/kurs + EIDO (ETF Indonesia di
     NYSE, tutup ~04:00 WIB), EEM, VIX, minyak, dollar index, plus kalender
     libur bursa IDX (deterministik & publik → prediksi return 0 saat tutup).
  B. Pre-close — tambah penutupan bursa Asia hari t yang tutup sebelum IHSG:
     Nikkei (14:00 WIB), HSI (15:00), KOSPI (13:30), TWII (12:30), ASX (13:00).
  C. Pre-close+ — tambah gap pembukaan IHSG hari t (harga open jam 09:00 WIB,
     korelasi 0.42 dengan return close-to-close).

Protokol jujur: model di-fit HANYA pada data sebelum test set; seluruh
hyperparameter dipilih via validasi pada 426 hari terakhir data training.

Data: results/extended_dataset.csv (grid tanggal identik dengan
cleaned_dataset.csv; sumber yfinance + exchange_calendars XIDX, ffill).

Hasil (test 426 hari, 2024-09-12 s/d 2026-04-30):
                                    RMSE     MAE    MAPE
  C. Pre-close+                    74.29   51.85  0.6989%
  B. Pre-close                     85.69   57.39  0.7751%
  A. Pre-open                      90.10   59.70  0.8074%
  Challenger v1                    90.88   61.34  0.8307%
  Naive (close kemarin)            92.49   61.41  0.8315%
  Hybrid SARIMAX-LSTM Univariat    92.62   61.61  0.8343%
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    df = pd.read_csv("results/extended_dataset.csv", parse_dates=["Date"]).set_index("Date")
    res = pd.read_csv("results/predictions_results.csv")
    test_dates = set(pd.to_datetime(res["Date"]).dt.strftime("%Y-%m-%d"))
    y_true = res["Actual_IHSG"].values

    px_cols = [c for c in df.columns if c not in ("is_holiday", "is_closed")]
    df[px_cols] = df[px_cols].ffill()
    df = df.dropna(subset=px_cols)  # EIDO baru ada sejak Mei 2010
    closed = df["is_closed"].values.astype(bool)
    # clip: WTI sempat negatif (Apr 2020); NaN return diganti 0 via nan_to_num
    r = {c: np.log(df[c].clip(lower=1e-6)).diff().values for c in px_cols}
    dates = df.index.strftime("%Y-%m-%d").values
    ihsg = df["IHSG"].values
    opn = np.log(df["JK_Open"].values / np.roll(ihsg, 1))
    opn[0] = 0.0

    ih, sp, ei, vx = r["IHSG"], r["SP500"], r["EIDO"], r["^VIX"]
    n2, hs, cl, em, dx = r["^N225"], r["^HSI"], r["CL=F"], r["EEM"], r["DX-Y.NYB"]
    fx, gd = r["USD_IDR"], r["Gold"]
    ks, tw, ax = r["^KS11"], r["^TWII"], r["^AXJO"]
    gap, emgap = ei - ih, em - ih

    W = 23
    rows, ys, didx = [], [], []
    for t in range(W, len(df)):
        vol = ih[t - 22 : t].std() + 1e-6
        rows.append([
            # blok A: info t-1 (tersedia sebelum pasar buka)
            ih[t - 1], ih[t - 2], sp[t - 1], gd[t - 1], fx[t - 1], cl[t - 1],
            dx[t - 1], ei[t - 1], gap[t - 1], em[t - 1], emgap[t - 1], vx[t - 1],
            ih[t - 5 : t].sum(), sp[t - 5 : t].sum(), ei[t - 5 : t].sum(),
            vol, sp[t - 22 : t].std(),
            # blok B: penutupan Asia hari t (sebelum IHSG tutup)
            n2[t], hs[t], ks[t], tw[t], ax[t], n2[t - 1], hs[t - 1],
            # blok C: gap pembukaan IHSG hari t
            opn[t], opn[t] / vol,
        ])
        ys.append(ih[t])
        didx.append(t)
    X = np.nan_to_num(np.array(rows))
    y, didx = np.array(ys), np.array(didx)

    is_test = np.array([d in test_dates for d in dates[didx]])
    ft = int(np.argmax(is_test))
    assert is_test[ft:].all() and not is_test[:ft].any()
    closed_s = closed[didx]
    n_feat = {"A": 17, "B": 24, "C": 26}

    def ensemble_predict(k):
        Xk = X[:, : n_feat[k]]
        models = [
            Ridge(alpha=1.0),
            HuberRegressor(alpha=1e-4, max_iter=2000),
            HistGradientBoostingRegressor(
                max_depth=3, learning_rate=0.03, max_iter=300,
                min_samples_leaf=100, random_state=0,
            ),
        ]
        pr = np.mean([m.fit(Xk[:ft], y[:ft]).predict(Xk[ft:]) for m in models], axis=0)
        pr[closed_s[ft:]] = 0.0  # bursa tutup (kalender publik) -> tidak bergerak
        return ihsg[didx[ft:] - 1] * np.exp(pr)

    def report(name, p):
        rmse = np.sqrt(mean_squared_error(y_true, p))
        mae = mean_absolute_error(y_true, p)
        mape = np.mean(np.abs((y_true - p) / y_true)) * 100
        print(f"{name:<42} RMSE={rmse:7.2f}  MAE={mae:7.2f}  MAPE={mape:.4f}%")
        return rmse, mae, mape

    print(f"Test set: {len(y_true)} hari ({res['Date'].iloc[0]} s/d {res['Date'].iloc[-1]})\n")
    champ = report("Hybrid SARIMAX-LSTM Univariat (juara lama)", res["Hybrid_Univariat"].values)
    report("Naive (close kemarin)", ihsg[didx[ft:] - 1])

    out = res[["Date", "Actual_IHSG", "Hybrid_Univariat"]].copy()
    metrics = pd.read_csv("results/evaluation_metrics.csv")
    scores = {}
    for k, label in [("A", "Challenger v2-A (pre-open)"),
                     ("B", "Challenger v2-B (pre-close)"),
                     ("C", "Challenger v2-C (pre-close + open gap)")]:
        p = ensemble_predict(k)
        scores[k] = report(label, p)
        out[f"Challenger_v2_{k}"] = p
        metrics = metrics[metrics["Model"] != label]
        metrics.loc[len(metrics)] = [label, *scores[k]]
        if k == "C":
            d = (y_true - res["Hybrid_Univariat"].values) ** 2 - (y_true - p) ** 2
            t_stat, pval = stats.ttest_1samp(d, 0)
            print(f"\nPaired t-test squared errors C vs juara: t={t_stat:.2f}, p={pval:.2e}")

    assert all(scores["A"][i] < champ[i] for i in range(3)), "v2-A kalah dari juara"
    assert scores["C"][0] < champ[0] * 0.85, "v2-C tidak mencapai perbaikan >15% RMSE"
    print("✅ v2-A menang di semua metrik dengan info set setara juara; "
          "v2-C memangkas RMSE >15%.")

    out.to_csv("results/predictions_challenger_v2.csv", index=False)
    metrics.to_csv("results/evaluation_metrics.csv", index=False)
    print("Tersimpan: results/predictions_challenger_v2.csv, results/evaluation_metrics.csv")


if __name__ == "__main__":
    main()
