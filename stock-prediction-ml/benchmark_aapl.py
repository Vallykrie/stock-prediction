"""Uji generalisasi metodologi challenger v2 pada saham Apple (AAPL).

Struktur informasi bergeser karena AAPL tutup 16:00 ET (bukan 16:00 WIB):
  A. Pre-open  — t-1 AS (S&P500, Nasdaq, VIX, DXY, minyak, emas) + penutupan
     Asia hari t (Nikkei/HSI tutup SEBELUM NY buka) + kalender NYSE.
  B. Pre-close — + penutupan Eropa hari t (DAX/FTSE tutup 11:30 ET).
  C. Pre-close+ — + gap pembukaan AAPL jam 09:30 ET.

Protokol identik dengan challenger_v2.py: grid hari kerja + ffill, fit hanya
pada data sebelum 2024-09-12, test 426 hari s/d 2026-04-30.

Hasil (test):
                            RMSE     MAE    MAPE
  Naive (close kemarin)    3.923   2.619  1.1321%
  v2-A (pre-open)          3.964   2.675  1.1546%   <- KALAH dari naive
  v2-B (pre-close)         3.889   2.623  1.1318%   <- setara naive
  v2-C (+ open gap)        3.350   2.250  0.9626%   <- -14.6% RMSE

Kesimpulan: edge pre-open yang besar di IHSG (spillover global) nyaris nol
di AAPL — AAPL memimpin pasar global, bukan mengikutinya, sehingga seluruh
informasi semalam sudah terserap di harga penutupan kemarin (pasar AS jauh
lebih efisien). Satu-satunya sinyal yang tersisa adalah gap pembukaan AAPL
sendiri. Data: results/aapl_dataset.csv (yfinance + exchange_calendars XNYS).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge

TEST_START = "2024-09-12"


def main():
    df = pd.read_csv("results/aapl_dataset.csv", parse_dates=["Date"]).set_index("Date")
    closed = df["is_closed"].values.astype(bool)
    px = [c for c in df.columns if c not in ("is_closed", "AAPL_Open")]
    r = {c: np.log(df[c].clip(lower=1e-6)).diff().values for c in px}
    aapl = df["AAPL"].values
    opn = np.log(df["AAPL_Open"].values / np.roll(aapl, 1))
    opn[0] = 0.0

    ap, sx, nq, vx = r["AAPL"], r["^GSPC"], r["^IXIC"], r["^VIX"]
    dx, cl, gd = r["DX-Y.NYB"], r["CL=F"], r["GC=F"]
    n2, hs, dax, fts = r["^N225"], r["^HSI"], r["^GDAXI"], r["^FTSE"]
    rel = ap - nq

    W = 23
    rows, ys, didx = [], [], []
    for t in range(W, len(df)):
        vol = ap[t - 22 : t].std() + 1e-6
        rows.append([
            # blok A: t-1 AS + Asia same-day (pre-open NY)
            ap[t - 1], ap[t - 2], sx[t - 1], nq[t - 1], vx[t - 1], dx[t - 1],
            cl[t - 1], gd[t - 1], rel[t - 1],
            ap[t - 5 : t].sum(), nq[t - 5 : t].sum(), vol, nq[t - 22 : t].std(),
            n2[t], hs[t],
            # blok B: Eropa same-day (pre-close NY)
            dax[t], fts[t],
            # blok C: gap pembukaan AAPL
            opn[t], opn[t] / vol,
        ])
        ys.append(ap[t])
        didx.append(t)
    X = np.nan_to_num(np.array(rows))
    y, didx = np.array(ys), np.array(didx)
    dates = df.index.strftime("%Y-%m-%d").values
    ft = int(np.argmax(dates[didx] >= TEST_START))
    closed_s = closed[didx]
    y_true = aapl[didx[ft:]]
    n_feat = {"A": 15, "B": 17, "C": 19}

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
        pr[closed_s[ft:]] = 0.0
        return aapl[didx[ft:] - 1] * np.exp(pr)

    def report(name, p):
        rmse = np.sqrt(np.mean((y_true - p) ** 2))
        mae = np.mean(np.abs(y_true - p))
        mape = np.mean(np.abs((y_true - p) / y_true)) * 100
        print(f"{name:<32} RMSE={rmse:7.3f}  MAE={mae:7.3f}  MAPE={mape:.4f}%")
        return rmse, mae, mape

    print(f"AAPL test: {len(y_true)} hari ({dates[didx][ft]} s/d {dates[didx][-1]})\n")
    naive = report("Naive (close kemarin)", aapl[didx[ft:] - 1])
    preds = {k: ensemble_predict(k) for k in "ABC"}
    scores = {k: report(f"Challenger v2-{k}", preds[k]) for k in "ABC"}

    # v2-C tetap harus menang telak; A/B memang tidak (pasar AS efisien)
    assert scores["C"][0] < naive[0] * 0.90, "v2-C tidak mencapai perbaikan >10% RMSE"
    print("\n✅ v2-C memangkas RMSE >10% vs naive. Edge pre-open (v2-A) hilang di "
          "AAPL — konsisten dengan efisiensi pasar AS.")

    out = pd.DataFrame({"Date": dates[didx][ft:], "Actual_AAPL": y_true,
                        **{f"Challenger_v2_{k}": preds[k] for k in "ABC"}})
    out.to_csv("results/predictions_aapl.csv", index=False)
    print("Tersimpan: results/predictions_aapl.csv")


if __name__ == "__main__":
    main()
