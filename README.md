# 📈 Prediksi IHSG dengan Model Hybrid SARIMAX-LSTM

Implementasi Model **Hybrid SARIMAX-LSTM** dengan Multiple Exogenous Variables untuk Prediksi **Indeks Harga Saham Gabungan (IHSG)**.

> **Proposal Skripsi** — Pande Kadek Nathan Prabhaswara Sudiara Putra (NIM: 235150207111051)
> Teknik Informatika, Fakultas Ilmu Komputer, Universitas Brawijaya

---

## 📋 Deskripsi Proyek

Proyek ini mengimplementasikan dan membandingkan tiga skenario model prediksi harga penutupan harian IHSG:

| No | Model | Deskripsi |
|:---:|---|---|
| 1 | **Hybrid SARIMAX-LSTM Multivariat** | SARIMAX dengan 3 variabel eksogen + LSTM untuk residual non-linear |
| 2 | **Hybrid SARIMAX-LSTM Univariat** | SARIMAX tanpa variabel eksogen + LSTM untuk residual non-linear |
| 3 | **LSTM Tunggal** | LSTM standalone yang langsung memprediksi harga IHSG |

### Variabel Eksogen

| Variabel | Ticker Yahoo Finance | Keterangan |
|---|:---:|---|
| Kurs USD/IDR | `IDR=X` | Nilai tukar Rupiah terhadap Dolar AS |
| Harga Emas Dunia | `GC=F` | Harga emas dunia (XAU/USD) |
| Indeks S&P 500 | `^GSPC` | Indeks saham utama Amerika Serikat |

### Metodologi

Model hybrid bekerja dengan memecah pola data menjadi dua komponen:

$$\hat{y}_t = \hat{L}_t + \hat{N}_t$$

- $\hat{L}_t$ → Komponen **linear** yang ditangkap oleh **SARIMAX**
- $\hat{N}_t$ → Komponen **non-linear** (residual) yang ditangkap oleh **LSTM**

Prediksi pada data uji menggunakan pendekatan **Rolling One-Step-Ahead**, di mana SARIMAX memprediksi satu hari ke depan, kemudian menerima nilai aktual hari tersebut sebelum memprediksi hari berikutnya. Hal ini memastikan perbandingan yang adil antar model.

---

## 🚀 Cara Menjalankan

### Prasyarat

- **Python 3.12**
- **Conda** (Miniconda atau Anaconda)
- Koneksi internet (untuk mengunduh data dari Yahoo Finance)

### 1. Clone Repository

```bash
git clone https://github.com/Vallykrie/stock-prediction.git
cd stock-prediction
```

### 2. Buat Virtual Environment (Conda)

```bash
conda create -n prediction python=3.12 -y
conda activate prediction
```

### 3. Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn yfinance statsmodels scikit-learn tensorflow tqdm ipywidgets jupyter
```

<details>
<summary><strong>📦 Daftar Library yang Digunakan</strong></summary>

| Library | Kegunaan |
|---|---|
| `numpy` | Komputasi numerik dan operasi array |
| `pandas` | Manipulasi dan analisis data tabular |
| `matplotlib` | Visualisasi dan pembuatan grafik |
| `seaborn` | Visualisasi statistik (heatmap korelasi) |
| `yfinance` | Pengambilan data historis dari Yahoo Finance |
| `statsmodels` | Pemodelan SARIMAX dan uji statistik (ADF) |
| `scikit-learn` | Normalisasi data (MinMaxScaler) dan metrik evaluasi |
| `tensorflow` | Pemodelan LSTM (Deep Learning) |
| `tqdm` | Progress bar untuk proses iteratif |
| `ipywidgets` | Widget interaktif untuk Jupyter Notebook |
| `jupyter` | Jupyter Notebook untuk menjalankan notebook |

</details>

### 4. Jalankan Jupyter Notebook

```bash
jupyter notebook stock_prediction.ipynb
```

### 5. Eksekusi Notebook

Jalankan seluruh sel secara berurutan dari atas ke bawah (**Kernel → Restart & Run All**).

> ⚠️ **Catatan:** Proses grid search SARIMAX dan training LSTM membutuhkan waktu beberapa menit tergantung spesifikasi komputer Anda.

---

## 📊 Hasil Evaluasi

Metrik evaluasi yang digunakan: **RMSE**, **MAE**, dan **MAPE**.

| Model | RMSE | MAE | MAPE (%) |
|---|:---:|:---:|:---:|
| Hybrid SARIMAX-LSTM Multivariat | 94.97 | 64.57 | 0.8753 |
| **Hybrid SARIMAX-LSTM Univariat** | **92.62** | **61.61** | **0.8343** |
| LSTM Tunggal | 339.08 | 292.70 | 3.8004 |

> ✅ Model terbaik: **Hybrid SARIMAX-LSTM Univariat** dengan MAPE **0.83%** (kategori: Sangat Baik)

### Temuan Utama

1. **Model Hybrid mengalahkan LSTM Tunggal secara signifikan.** Pendekatan rolling one-step-ahead pada SARIMAX, dikombinasikan dengan LSTM untuk memodelkan residual non-linear, terbukti jauh lebih akurat dibandingkan LSTM standalone.

2. **Model Univariat sedikit lebih unggul dari Multivariat.** Hal ini mengindikasikan bahwa harga penutupan IHSG harian sudah merangkum informasi makroekonomi (*Efficient Market Hypothesis*), sehingga penambahan variabel eksogen mengintroduksi sedikit *noise* tambahan.

3. **Semua model Hybrid memiliki MAPE < 1%**, menunjukkan kemampuan peramalan yang sangat baik untuk prediksi jangka pendek (one-step-ahead).

---

## 📁 Struktur Proyek

```
stock-prediction/
│
├── stock_prediction.ipynb       # Notebook utama (seluruh implementasi)
├── proposal.md                  # Proposal skripsi (format teks)
├── Proposal Skripsi_...pdf      # Proposal skripsi (format PDF)
├── README.md                    # Dokumentasi proyek (file ini)
│
├── cleaned_dataset.csv          # Dataset gabungan yang sudah dibersihkan
├── predictions_results.csv      # Hasil prediksi ketiga model
├── evaluation_metrics.csv       # Metrik evaluasi (RMSE, MAE, MAPE)
│
├── eda_time_series.png          # Visualisasi data historis
├── correlation_matrix.png       # Matriks korelasi antar variabel
├── acf_pacf_ihsg.png            # Plot ACF & PACF (IHSG)
├── train_test_split.png         # Visualisasi pembagian data
├── lstm_training_hybrid_multi.png  # Training history LSTM (Hybrid Multi)
├── lstm_training_hybrid_uni.png    # Training history LSTM (Hybrid Uni)
├── lstm_training_standalone.png    # Training history LSTM (Standalone)
├── prediction_comparison_all.png   # Perbandingan prediksi semua model
├── prediction_individual.png       # Prediksi individual per model
├── prediction_errors.png           # Distribusi error prediksi
├── error_distribution.png          # Histogram error
├── scatter_actual_vs_predicted.png # Scatter plot aktual vs prediksi
├── model_comparison_bar.png        # Bar chart perbandingan metrik
├── final_summary.png               # Ringkasan akhir (comprehensive)
└── future_predictions_final.png    # Prediksi masa depan (out-of-sample)
```

---

## ⚙️ Parameter Model

| Parameter | Nilai |
|---|---|
| Periode Data | Januari 2010 – Januari 2026 |
| Rasio Train/Test | 80% / 20% |
| SARIMAX Order | Ditentukan melalui Grid Search (AIC) |
| SARIMAX Prediction | Rolling one-step-ahead (`append`, `refit=False`) |
| LSTM Look-back | 30 time steps |
| LSTM Units | 50 |
| LSTM Dropout | 0.2 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Epochs | 100 (Early Stopping, patience=10) |
| Batch Size | 32 |
| Scaler | MinMaxScaler [0, 1] |

---

## 📚 Referensi

1. **Achmadi dkk. (2023)** — Hybrid SARIMAX-LSTM untuk prediksi harga cryptocurrency
2. **Yusuf (2021)** — Penerapan LSTM untuk prediksi IHSG
3. **Hochreiter & Schmidhuber (1997)** — Long Short-Term Memory

---

## 📄 Lisensi

Proyek ini dikembangkan sebagai bagian dari tugas akhir (skripsi) di Program Studi Teknik Informatika, Universitas Brawijaya.
