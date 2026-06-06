Implementasi Model Hybrid SARIMAX-LSTM dengan
Multiple Exogenous Variables untuk Prediksi Indeks Harga
Saham Gabungan (IHSG)
PROPOSAL SKRIPSI
Disusun oleh:
Pande Kadek Nathan Prabhaswara Sudiara Putra
NIM: 235150207111051
TEKNIK INFORMATIKA
DEPARTEMEN TEKNIK INFORMATIKA
FAKULTAS ILMU KOMPUTER
UNIVERSITAS BRAWIJAYA
MALANG
2025

DAFTAR ISI
DAFTAR ISI ii
DAFTAR TABEL iv
DAFTAR GAMBAR v
DAFTAR LAMPIRAN vi
BAB 1 PENDAHULUAN 1
1.1 Latar Belakang 1
1.2 Rumusan Masalah 2
1.3 Tujuan 2
1.4 Manfaat 2
1.5 Batasan Masalah 2
1.6 Sistematika Pembahasan 3
BAB 2 LANDASAN KEPUSTAKAAN 4
2.1 Tinjauan Pustaka 4
2.2 Dasar Teori 6
2.2.1 Pasar Modal dan IHSG 6
2.2.2 Analisis Deret Waktu (Time Series) 6
2.2.3 Model SARIMAX 6
2.2.4 Model Long Short-Term Memory (LSTM) 7
2.2.5 Model Hybrid SARIMAX-LSTM 7
2.2.6 Evaluasi Model 8
2.3 Kerangka Konseptual Penelitian 8
BAB 3 METODOLOGI 10
3.1 Pendekatan Penelitian 10
3.2 Strategi dan Rancangan Penelitian 10
3.3 Sumber dan Teknik Pengumpulan Data 10
3.4 Variabel Penelitian 11
3.5 Metode Analisis Data 11
3.5.1 Pra-pemrosesan Data 11
3.5.2 Perancangan Model SARIMAX 12
3.5.3 Perancangan Model LSTM untuk Residual 12
3.5.4 Penggabungan Model (Hybrid) 12
3.6 Evaluasi Model 13
3.7 Perangkat Pendukung 13
2

DAFTAR TABEL
Tabel 2.1 Tinjauan Pustaka 4
Tabel 3.1 Variabel Penelitian 11
3

DAFTAR GAMBAR
Gambar 2.1 Model Hybrid ARIMAX-LSTM 9
4

DAFTAR LAMPIRAN
LAMPIRAN A PERSYARATAN FISIK DAN TATA LETAK 16
A.1 Kertas 16
A.2 Margin 16
A.3 Jenis dan ukuran huruf 16
A.4 Spasi 16
A.5 Kepala Bab dan Subbab 16
A.6 Nomor halaman 17
LAMPIRAN B PENGGUNAAN BAHASA 18
5

BAB 1 PENDAHULUAN
1.1 Latar Belakang
Pasar modal memegang peranan vital dalam perekonomian suatu negara
sebagai sarana pertemuan antara pihak yang memiliki kelebihan dana (investor)
dan pihak yang membutuhkan dana. Di Indonesia, indikator utama kinerja pasar
saham adalah Indeks Harga Saham Gabungan (IHSG), yang mencerminkan
rata-rata pergerakan harga seluruh saham yang tercatat di Bursa Efek Indonesia
(BEI). Pergerakan IHSG sangat dinamis dan memiliki volatilitas tinggi karena
dipengaruhi oleh berbagai faktor kompleks, baik internal maupun eksternal.
Fluktuasi yang sulit diprediksi ini menjadi tantangan besar bagi investor dalam
pengambilan keputusan investasi yang tepat untuk meminimalkan risiko kerugian.
Upaya memprediksi pergerakan harga saham maupun indeks saham telah
banyak dilakukan menggunakan berbagai metode. Metode statistik linear seperti
Autoregressive Integrated Moving Average (ARIMA) atau variasinya Seasonal
ARIMA with Exogenous variables (SARIMAX) sering digunakan karena
kemampuannya menangkap pola linear dan musiman. Namun, metode ini
memiliki keterbatasan dalam menangkap pola non-linear yang sering muncul
pada data keuangan yang fluktuatif. Di sisi lain, pendekatan Machine Learning
khususnya Deep Learning seperti Long Short-Term Memory (LSTM) telah terbukti
handal dalam menangani data deret waktu (time series) non-linear dan memori
jangka panjang. Yusuf (2021) telah menerapkan LSTM untuk memprediksi IHSG
dan menghasilkan akurasi yang cukup baik. Namun, penggunaan model tunggal
(hanya SARIMAX atau hanya LSTM) seringkali tidak mampu menangkap
keseluruhan karakteristik data yang mengandung komponen linear dan
non-linear sekaligus.
Untuk mengatasi kelemahan model tunggal, pendekatan hybrid mulai
dikembangkan. Achmadi dkk. (2023) dalam penelitiannya mengusulkan metode
Hybrid SARIMAX-LSTM untuk memprediksi harga Cryptocurrency (Bitcoin).
Penelitian tersebut membuktikan bahwa penggabungan SARIMAX (untuk
menangkap pola linear) dan LSTM (untuk menangkap sisa error non-linear)
menghasilkan akurasi yang lebih tinggi dibandingkan model tunggal. Meskipun
demikian, penelitian Achmadi dkk. (2023) hanya menggunakan satu variabel
eksogen, yaitu volume perdagangan. Padahal, pergerakan IHSG dipengaruhi oleh
banyak faktor makroekonomi (multiple exogenous variables) seperti nilai tukar
mata uang (Kurs USD/IDR), harga komoditas global (Emas), dan indeks pasar
global (S&P 500). Penggunaan satu variabel eksogen dinilai belum cukup untuk
merepresentasikan kompleksitas faktor yang memengaruhi IHSG.
Berdasarkan permasalahan tersebut, terdapat celah penelitian (research
gap) untuk menerapkan dan mengembangkan model Hybrid SARIMAX-LSTM
pada objek penelitian yang berbeda, yaitu IHSG, dengan melibatkan lebih banyak
1

variabel eksternal. Penelitian ini bertujuan untuk meningkatkan akurasi prediksi
IHSG dengan mengimplementasikan model Hybrid SARIMAX-LSTM yang
diperkaya dengan multiple exogenous variables. Pendekatan ini diharapkan
mampu menangkap pola linear, non-linear, serta pengaruh faktor eksternal secara
komprehensif.
1.2 Rumusan Masalah
Berdasarkan latar belakang yang telah dipaparkan, maka rumusan masalah
dalam penelitian ini adalah:
1. Bagaimana merancang dan mengimplementasikan model Hybrid
SARIMAX-LSTM dengan multiple exogenous variables (Kurs USD/IDR, Harga
Emas, dan Indeks S&P 500) untuk memprediksi IHSG?
2. Bagaimana perbandingan tingkat akurasi model Hybrid SARIMAX-LSTM
multivariat dibandingkan dengan model Hybrid SARIMAX-LSTM univariat dan
model tunggal LSTM dalam memprediksi IHSG?
1.3 Tujuan
Tujuan yang ingin dicapai dari penelitian ini adalah:
1. Merancang dan mengimplementasikan model prediksi IHSG menggunakan
metode Hybrid SARIMAX-LSTM yang memanfaatkan variabel eksogen Kurs
USD/IDR, Harga Emas, dan Indeks S&P 500.
2. Mengukur dan menganalisis kinerja model yang diusulkan dibandingkan
dengan model Hybrid univariat dan model tunggal LSTM menggunakan metrik
evaluasi Root Mean Square Error (RMSE), Mean Absolute Error (MAE), dan
Mean Absolute Percentage Error (MAPE).
1.4 Manfaat
Penelitian ini diharapkan dapat memberikan manfaat sebagai berikut:
1. Manfaat Teoritis: Memberikan kontribusi pada pengembangan ilmu komputer
khususnya di bidang kecerdasan buatan dan prediksi deret waktu finansial,
dengan membuktikan efektivitas model hybrid dan pengaruh penambahan
variabel eksogen jamak.
2. Manfaat Praktis: Memberikan alternatif alat bantu prediksi yang lebih akurat
bagi investor atau analis pasar modal dalam memproyeksikan pergerakan
IHSG, sehingga dapat membantu dalam pengambilan keputusan investasi.
1.5 Batasan Masalah
Agar penelitian ini lebih terarah dan fokus, maka ditetapkan batasan masalah
sebagai berikut:
1. Data yang digunakan adalah data harian IHSG (closing price) yang diperoleh
dari Yahoo Finance.
2

2. Periode data yang digunakan adalah 5 tahun terakhir (Januari 2019 –
Desember 2024) untuk mencakup kondisi sebelum, saat, dan pasca pandemi
COVID-19.
3. Variabel eksogen yang digunakan dibatasi pada: Nilai Tukar Rupiah terhadap
Dolar AS (USD/IDR), Harga Emas Dunia (XAU/USD), dan Indeks S&P 500.
4. Implementasi model menggunakan bahasa pemrograman Python dengan
library pendukung seperti statsmodels untuk SARIMAX dan TensorFlow/Keras
untuk LSTM.
5. Evaluasi kinerja model diukur menggunakan Root Mean Square Error (RMSE),
Mean Absolute Error (MAE), dan Mean Absolute Percentage Error (MAPE).
1.6 Sistematika Pembahasan
Sistematika penulisan dalam skripsi ini disusun sebagai berikut:
BAB 1 PENDAHULUAN Bab ini menjelaskan latar belakang pemilihan topik,
rumusan masalah yang akan diselesaikan, tujuan penelitian, manfaat yang
diharapkan, batasan masalah, serta sistematika penulisan laporan.
BAB 2 LANDASAN KEPUSTAKAAN Bab ini memuat teori-teori penunjang yang
relevan dengan penelitian, meliputi konsep Pasar Modal dan IHSG, teori Time
Series, metode SARIMAX, metode LSTM, konsep Model Hybrid, serta tinjauan
penelitian terdahulu yang berkaitan.
BAB 3 METODOLOGI Bab ini menjelaskan tahapan-tahapan penelitian yang
dilakukan secara sistematis, mulai dari pengumpulan data, pra-pemrosesan data,
perancangan model hybrid, skenario pengujian, hingga metode evaluasi yang
digunakan.
3

|     |     | BAB 2  |     | LANDASAN KEPUSTAKAAN |     |     |     |
| --- | --- | ------ | --- | -------------------- | --- | --- | --- |
2.1  Tinjauan Pustaka
Penelitian  mengenai prediksi  harga saham  maupun aset keuangan lainnya
telah  banyak  dilakukan  dengan  menggunakan  berbagai  metode,  mulai  dari
metode  statistik  linear  hingga  metode  kecerdasan  buatan  berbasis  Deep
Learning.
Tabel  2.1 Tinjauan Pustaka
| No  | Nama     |     | Judul       |     | Persamaan   |     | Perbedaan   |
| --- | -------- | --- | ----------- | --- | ----------- | --- | ----------- |
|     | Penulis  |     | Penelitian  |     | Penelitian  |     | Penelitian  |
(Tahun)
| 1   | Achmadi  |     | Cryptocurrency  |     | Menggunakan        |                        | Objek penelitian  |
| --- | -------- | --- | --------------- | --- | ------------------ | ---------------------- | ----------------- |
|     | dkk.     |     | Price           |     | pendekatan         | adalah Cryptocurrency  |                   |
|     | (2023)   |     | Movement        |     | metode Hybrid      |                        | (Bitcoin).        |
|     |          |     | Prediction      |     | SARIMAX-LSTM       | Menggunakan data       |                   |
|     |          |     | Using the       |     | untuk memprediksi  | mingguan (weekly).     |                   |
|     |          |     | Hybrid          |     | data deret waktu   | Hanya menggunakan      |                   |
|     |          |     | SARIMAX-LSTM    |     | keuangan.          | satu variabel eksogen  |                   |
|     |          |     | Method          |     | Melibatkan         |                        | yaitu Volume.     |
variabel eksogen
dalam pemodelan.
| 2   | Yusuf  |     | Prediksi Indeks  |     | Sama-sama  |     | Menggunakan  |
| --- | ------ | --- | ---------------- | --- | ---------- | --- | ------------ |
(2021)  Harga Saham  meneliti prediksi  metode tunggal LSTM.
|     |     |     | Gabungan     |     | pada objek Indeks  |     | Model bersifat    |
| --- | --- | --- | ------------ | --- | ------------------ | --- | ----------------- |
|     |     |     | (IHSG)       |     | Harga Saham        |     | univariat (hanya  |
|     |     |     | Menggunakan  |     | Gabungan (IHSG).   |     | menggunakan data  |
|     |     |     | Long         |     | Menggunakan        |     | historis harga    |
|     |     |     | Short-Term   |     | metode berbasis    |     | penutupan IHSG    |
|     |     |     | Memory       |     | Deep Learning      |     | tanpa variabel    |
|     |     |     |              |     | (LSTM).            |     | eksternal).       |
3  Rusyida  Prediksi Harga  Topik penelitian  Objek penelitian
|     | &        |     | Saham Garuda  |     | mengenai prediksi    | spesifik pada saham  |                  |
| --- | -------- | --- | ------------- | --- | -------------------- | -------------------- | ---------------- |
|     | Pratama  |     | Indonesia di  |     | harga saham          | perusahaan (GIAA).   |                  |
|     | (2020)   |     | Tengah        |     | menggunakan          |                      | Menggunakan      |
|     |          |     | Pandemi       |     | analisis data deret  |                      | metode tunggal   |
|     |          |     | Covid-19      |     | waktu historis.      | ARIMA yang memiliki  |                  |
|     |          |     | Menggunakan   |     | Menggunakan          | keterbatasan dalam   |                  |
|     |          |     | Metode ARIMA  |     | metode statistik     |                      | menangkap pola   |
|     |          |     |               |     | linear.              |                      | non-linear yang  |
kompleks.
4

4 Simatup Perancangan Menggunakan Fokus pada
ang dkk. Aplikasi metode Deep pembangunan aplikasi
(2022) Berbasis Web Learning berbasis berbasis website
untuk Prediksi LSTM untuk untuk memprediksi 15
Harga Saham memprediksi harga emiten saham LQ45.
dengan saham di pasar Menggunakan
Metode LSTM modal Indonesia. metode tunggal LSTM
dengan variasi
optimizer (Adam dan
Nadam).
Achmadi dkk. (2023) melakukan penelitian untuk memprediksi pergerakan
harga cryptocurrency (Bitcoin) menggunakan metode Hybrid SARIMAX-LSTM.
Penelitian ini menggunakan volume perdagangan sebagai variabel eksogen. Hasil
penelitian menunjukkan bahwa model hybrid mampu menangkap pola linear dan
non-linear secara bersamaan, menghasilkan kinerja yang lebih unggul
dibandingkan model tunggal SARIMAX atau LSTM, dengan nilai Root Mean Square
Error (RMSE) yang lebih rendah.
Yusuf (2021) menerapkan metode Long Short-Term Memory (LSTM) untuk
memprediksi Indeks Harga Saham Gabungan (IHSG). Penelitian ini menggunakan
data historis harian dan menemukan bahwa LSTM dengan parameter epoch 50
menghasilkan akurasi yang baik dengan nilai RMSE sebesar 6.2335. Namun,
penelitian ini hanya menggunakan data univariat (hanya data historis IHSG) tanpa
melibatkan variabel eksternal.
Di sisi lain, Rusyida dan Pratama (2020) menggunakan metode statistik ARIMA
untuk memprediksi harga saham PT. Garuda Indonesia Tbk di tengah pandemi
COVID-19. Penelitian ini menunjukkan bahwa ARIMA mampu memodelkan data
deret waktu jangka pendek, namun memiliki keterbatasan dalam menangkap
volatilitas ekstrem tanpa bantuan variabel pendukung atau metode non-linear.
Simatupang dkk. (2022) mengembangkan aplikasi berbasis web untuk prediksi
harga saham indeks LQ45 menggunakan LSTM. Penelitian ini membandingkan
pengoptimal (optimizer) Adam dan Nadam, serta menemukan bahwa jumlah
epoch berpengaruh signifikan terhadap akurasi model. Hal ini sejalan dengan
temuan Zahara dkk. (2019) yang melakukan komparasi berbagai algoritma
optimasi pada LSTM dan menyimpulkan bahwa penggunaan optimizer yang
tepat, seperti Nadam atau Adam, sangat krusial dalam meminimalkan error
prediksi pada data deret waktu finansial.
Berdasarkan tinjauan pustaka di atas, mayoritas penelitian sebelumnya
berfokus pada model tunggal (ARIMA atau LSTM saja) atau model hybrid yang
diterapkan pada aset kripto dengan variabel eksogen terbatas. Penelitian ini akan
mengisi celah tersebut dengan menerapkan model Hybrid SARIMAX-LSTM pada
prediksi IHSG dan memperkaya model dengan multiple exogenous variables (Kurs
USD/IDR, Harga Emas, dan Indeks S&P 500) untuk meningkatkan akurasi prediksi.
5

2.2 Dasar Teori
2.2.1 Pasar Modal dan IHSG
Pasar modal merupakan tempat bertemunya pihak penjual dan pembeli
instrumen keuangan, serta dapat dijadikan indikator kemajuan ekonomi suatu
negara. Di Indonesia, pasar modal dijalankan oleh Bursa Efek Indonesia (BEI).
Indeks Harga Saham Gabungan (IHSG) adalah indeks yang mengukur kinerja
harga seluruh saham yang tercatat di BEI. Pergerakan IHSG sangat dipengaruhi
oleh berbagai faktor, baik internal perusahaan maupun faktor makroekonomi
eksternal seperti inflasi dan tingkat suku bunga.
2.2.2 Analisis Deret Waktu (Time Series)
Data deret waktu adalah rangkaian data yang dikumpulkan berdasarkan
urutan waktu tertentu. Analisis deret waktu bertujuan untuk mengidentifikasi
pola historis guna memprediksi nilai di masa depan. Salah satu syarat penting
dalam analisis statistik deret waktu seperti ARIMA adalah stasioneritas data, di
mana rata-rata dan varians data konstan sepanjang waktu. Pengujian
stasioneritas sering dilakukan menggunakan uji Augmented Dickey-Fuller (ADF).
2.2.3 Model SARIMAX
Model Seasonal Autoregressive Integrated Moving Average with Exogenous
Variables (SARIMAX) merupakan pengembangan dari model ARIMA yang mampu
menangani pola musiman (seasonal) dan memasukkan variabel eksternal
(exogenous). Model ini mengasumsikan bahwa variabel dependen
dipengaruhi oleh nilai masa lalunya sendiri serta variabel independen lain pada
waktu .
Persamaan umum model SARIMAX dapat dituliskan sebagai berikut:
Keterangan:
: Nilai variabel dependen pada waktu
: Nilai variabel eksogen ke- pada waktu
: Operator backshift
: Parameter autoregressive dan moving average non-musiman
: Parameter autoregressive dan moving average musiman
6

2.2.4 Model Long Short-Term Memory (LSTM)
LSTM adalah pengembangan dari Recurrent Neural Network (RNN) yang
dirancang untuk mengatasi masalah vanishing gradient dan mampu mempelajari
ketergantungan jangka panjang (long-term dependencies) pada data deret waktu.
Arsitektur LSTM terdiri dari sel memori dan tiga gerbang utama (gates), yaitu
input gate, forget gate, dan output gate.
Forget Gate: Memutuskan informasi mana yang akan dibuang dari sel memori.
Input Gate: Menentukan informasi baru yang akan disimpan ke dalam sel
memori.
Cell State Update: Memperbarui keadaan sel memori.
Output Gate: Menentukan keluaran (hidden state) dari sel LSTM.
2.2.5 Model Hybrid SARIMAX-LSTM
Model hybrid ini bekerja dengan prinsip memodelkan komponen linear dan
non-linear secara terpisah. Model SARIMAX digunakan untuk menangkap pola
linear dari data deret waktu. Sisa hasil prediksi (residuals) dari model SARIMAX,
yang diasumsikan mengandung pola non-linear, kemudian dijadikan input untuk
dilatih menggunakan model LSTM.Prediksi akhir diperoleh dengan
menjumlahkan hasil prediksi linear dari SARIMAX dan hasil prediksi
non-linear dari LSTM :
7

Pendekatan ini terbukti menghasilkan prediksi yang lebih akurat dibandingkan
penggunaan model tunggal, seperti yang dibuktikan dalam penelitian Achmadi
dkk. (2023) pada data cryptocurrency.
2.2.6 Evaluasi Model
Untuk mengukur kinerja model prediksi, digunakan beberapa metrik evaluasi
kesalahan (error metrics):
1. Root Mean Square Error (RMSE) RMSE mengukur rata-rata kuadrat selisih
antara nilai prediksi dan nilai aktual. RMSE memberikan bobot lebih pada
kesalahan yang besar.
2. Mean Absolute Error (MAE) MAE mengukur rata-rata selisih mutlak antara
nilai prediksi dan nilai aktual tanpa memperhatikan arah kesalahan.
3. Mean Absolute Percentage Error (MAPE) MAPE menyatakan persentase
kesalahan rata-rata. Nilai MAPE di bawah 10% dikategorikan sebagai
kemampuan peramalan yang sangat baik.
Keterangan:
: Jumlah data uji
: Nilai aktual pada data ke-
: Nilai hasil prediksi pada data ke-
2.3 Kerangka Konseptual Penelitian
Kerangka konseptual penelitian ini menggambarkan alur proses dari input
hingga output sistem prediksi. Penelitian dimulai dengan pengumpulan data
historis IHSG dan variabel eksogen (Kurs USD/IDR, Harga Emas, S&P 500). Data
kemudian melalui tahap preprocessing (pembersihan dan normalisasi).
8

Gambar 2.1 Model Hybrid ARIMAX-LSTM
Tahap pemodelan dimulai dengan melatih model SARIMAX pada data latih
untuk menangkap pola linear. Residual dari model SARIMAX kemudian diekstraksi
dan digunakan sebagai data latih untuk model LSTM guna menangkap pola
non-linear. Prediksi akhir merupakan gabungan hasil kedua model tersebut.
Evaluasi dilakukan dengan membandingkan metrik RMSE, MAE, dan MAPE antara
model Hybrid SARIMAX-LSTM Multivariat, Hybrid Univariat, dan LSTM Tunggal.
9

BAB 3 METODOLOGI
3.1 Pendekatan Penelitian
Penelitian ini menggunakan pendekatan kuantitatif dengan jenis penelitian
eksperimental. Pendekatan kuantitatif dipilih karena penelitian berfokus pada
analisis data numerik berupa data deret waktu (time series) harga saham dan
variabel ekonomi makro. Sifat eksperimental dalam penelitian ini terletak pada
perancangan dan pengujian model Hybrid SARIMAX-LSTM dengan berbagai
skenario variabel eksogen untuk melihat pengaruhnya terhadap akurasi prediksi
Indeks Harga Saham Gabungan (IHSG).
3.2 Strategi dan Rancangan Penelitian
Strategi penelitian yang digunakan adalah studi kasus pada data pergerakan
IHSG di Bursa Efek Indonesia. Rancangan penelitian disusun secara sistematis
mulai dari pengumpulan data hingga evaluasi model. Alur tahapan penelitian
diadaptasi dari metode hybrid yang dikembangkan oleh Achmadi dkk. (2023),
yang terdiri dari pemodelan linear menggunakan SARIMAX dan pemodelan
non-linear residual menggunakan LSTM.
Secara garis besar, tahapan penelitian adalah sebagai berikut:
1. Studi Literatur: Mempelajari teori terkait SARIMAX, LSTM, dan faktor-faktor
yang memengaruhi IHSG.
2. Pengumpulan Data: Mengambil data historis IHSG dan variabel eksogen dari
sumber terpercaya.
3. Pra-pemrosesan Data (Preprocessing): Meliputi pembersihan data,
pengecekan stasioneritas, dan normalisasi.
4. Pemodelan SARIMAX: Melatih model SARIMAX pada data IHSG dengan
melibatkan variabel eksogen untuk menangkap pola linear.
5. Ekstraksi Residual: Menghitung selisih antara prediksi SARIMAX dan data
aktual (nilai error).
6. Pemodelan LSTM: Melatih model LSTM menggunakan data residual dari
SARIMAX untuk menangkap pola non-linear.
7. Penggabungan Model (Hybrid): Menjumlahkan hasil prediksi SARIMAX dan
hasil prediksi residual LSTM untuk mendapatkan prediksi final.
8. Evaluasi Model: Mengukur akurasi menggunakan metrik error dan
membandingkannya dengan model pembanding.
3.3 Sumber dan Teknik Pengumpulan Data
Data yang digunakan dalam penelitian ini adalah data sekunder yang
bersifat publik. Teknik pengumpulan data dilakukan dengan metode
dokumentasi, yaitu mengunduh data historis melalui layanan penyedia data
keuangan Yahoo Finance menggunakan pustaka yfinance pada bahasa
pemrograman Python.
10

Data yang dikumpulkan memiliki periode harian mulai dari 1 Januari 2019
hingga 31 Desember 2024. Pemilihan periode ini bertujuan untuk mencakup
dinamika pasar sebelum, selama, dan sesudah pandemi COVID-19. Data dibagi
menjadi dua bagian, yaitu data latih (training data) sebesar 80% untuk proses
pembelajaran model, dan data uji (testing data) sebesar 20% untuk validasi
kinerja model.
3.4 Variabel Penelitian
Variabel dalam penelitian ini terdiri dari variabel dependen (target
prediksi) dan variabel independen (input/eksogen). Rincian variabel dapat dilihat
pada Tabel 3.1.
Tabel 3.1 Variabel Penelitian
No Jenis Variabel Nama Variabel Simbol Keterangan
1 Dependen Harga Target utama prediksi
Penutupan yang mencerminkan
IHSG kinerja pasar saham
Indonesia.
2 Independen Kurs USD/IDR Nilai tukar Rupiah
(Eksogen) terhadap Dolar AS,
mempengaruhi aliran
dana asing.
3 Independen Harga Emas Harga penutupan
(Eksogen) Dunia komoditas emas
(XAU/USD) sebagai aset
safe haven.
4 Independen Indeks S&P 500 Indeks pasar saham
(Eksogen) Amerika Serikat yang
merepresentasikan
sentimen pasar global.
3.5 Metode Analisis Data
3.5.1 Pra-pemrosesan Data
Sebelum data dimasukkan ke dalam model, dilakukan beberapa tahapan
pra-pemrosesan:
1. Pembersihan Data: Menangani missing values (hari libur bursa) menggunakan
metode interpolasi atau forward fill agar deret waktu kontinu.
11

2. Uji Stasioneritas: Menggunakan uji Augmented Dickey-Fuller (ADF) untuk
memastikan data stasioner, yang merupakan syarat utama pemodelan
SARIMAX. Jika data tidak stasioner, dilakukan proses differencing .
3. Normalisasi Data: Khusus untuk input model LSTM, data dinormalisasi
menggunakan metode MinMax Scaler ke dalam rentang [0, 1] untuk
mempercepat konvergensi saat proses pelatihan.
3.5.2 Perancangan Model SARIMAX
Model SARIMAX digunakan sebagai filter linear pertama. Penentuan
parameter terbaik dilakukan menggunakan metode grid
search berdasarkan nilai Akaike Information Criterion (AIC) terendah. Variabel
eksogen dimasukkan ke dalam model untuk membantu
memprediksi . Persamaan prediksi linear yang dihasilkan adalah:
3.5.3 Perancangan Model LSTM untuk Residual
Setelah model SARIMAX terbentuk, dilakukan perhitungan nilai sisa atau
residual dengan rumus:
Nilai ini diasumsikan memuat pola non-linear yang tidak tertangkap
oleh SARIMAX. Model LSTM kemudian dirancang untuk memprediksi nilai
residual di masa depan . Arsitektur LSTM yang digunakan meliputi input
layer, hidden layer dengan jumlah neuron tertentu (misalnya 50 unit), dan output
layer. Optimasi bobot menggunakan algoritma Adam atau Nadam sesuai dengan
hasil eksperimen terbaik pada penelitian terdahulu.
3.5.4 Penggabungan Model (Hybrid)
Prediksi akhir diperoleh dengan menjumlahkan hasil prediksi
komponen linear dari SARIMAX dan hasil prediksi komponen non-linear dari
LSTM, sebagaimana dijelaskan pada persamaan berikut:
12

3.6 Evaluasi Model
Untuk mengukur tingkat keberhasilan model dalam melakukan prediksi,
digunakan metrik evaluasi kesalahan yang membandingkan nilai prediksi dengan
nilai aktual pada data uji. Metrik yang digunakan adalah:
1. Root Mean Square Error (RMSE): Digunakan untuk melihat besaran kesalahan
dengan penalti lebih besar pada kesalahan yang signifikan.
2. Mean Absolute Error (MAE): Digunakan untuk mengetahui rata-rata selisih
mutlak antara nilai prediksi dan nilai aktual, memberikan gambaran besaran
kesalahan yang lebih intuitif.
3. Mean Absolute Percentage Error (MAPE): Digunakan untuk melihat
persentase kesalahan rata-rata agar lebih mudah diinterpretasikan secara
relatif.
3.7 Perangkat Pendukung
Penelitian ini dilaksanakan menggunakan perangkat keras laptop dengan
spesifikasi prosesor setara AMD Ryzen 7 7840HS atau lebih tinggi dan NVIDIA
GeForce RTX 4060 atau lebih tinggi untuk menangani komputasi model Deep
Learning. Perangkat lunak yang digunakan meliputi Sistem Operasi Linux dan
bahasa pemrograman Python. Pustaka (library) utama yang digunakan adalah
Pandas untuk manipulasi data, Statsmodels untuk pemodelan SARIMAX,
TensorFlow/Keras untuk pembangunan model LSTM, serta Matplotlib untuk
visualisasi data.
13

DAFTAR REFERENSI
Achmadi, G. R., Saikhu, A., & Amaliah, B., 2023. Cryptocurrency Price Movement
Prediction Using the Hybrid SARIMAX-LSTM Method. Dalam 2023
International Conference on Advanced Mechatronics, Intelligent
Manufacture and Industrial Automation (ICAMIMIA) (hlm. 223-228).
Surabaya: IEEE.
Hochreiter, S., & Schmidhuber, J., 1997. Long Short-Term Memory. Neural
Computation, 9(8), pp. 1735-1780.
Rusyida, W. Y., & Pratama, V. Y., 2020. Prediksi Harga Saham Garuda Indonesia di
Tengah Pandemi Covid-19 Menggunakan Metode ARIMA. SQUARE: Journal
of Mathematics and Mathematics Education, 2(1), pp. 73-81.
Simatupang, C. G. K., Swastika, W., & Suganda, T. R., 2022. Perancangan Aplikasi
Berbasis Web untuk Prediksi Harga Saham dengan Metode LSTM.
SAINSBERTEK: Jurnal Ilmiah Sains & Teknologi, 3(1), pp. 1-8.
Yusuf, A., 2021. Prediksi Indeks Harga Saham Gabungan (IHSG) Menggunakan
Long Short-Term Memory. Jurnal Epsilon, 15(2), pp. 124-132.
Zahara, S., Sugianto, & Ilmiddafiq, M. B., 2019. Prediksi Indeks Harga Konsumen
Menggunakan Metode Long Short Term Memory (LSTM) Berbasis Cloud
Computing. Jurnal RESTI (Rekayasa Sistem dan Teknologi Informasi), 3(3),
pp. 357-363.
14

LAMPIRAN A PERSYARATAN FISIK DAN TATA LETAK
A.1 Kertas
Kertas yang digunakan adalah HVS 70 mg berukuran A4. Apabila terdapat
gambar-gambar yang menggunakan kertas berukuran lebih besar dari A4,
hendaknya dilipat sesuai dengan aturan yang berlaku. Pengetikan hanya
dilakukan pada satu muka kertas, tidak bolak balik.
A.2 Margin
Batas pengetikan naskah adalah sebagai berikut :
● Margin kiri: 4 cm
● Margin atas: 3 cm
● Margin kanan: 3 cm
● Margin bawah: 3 cm
A.3 Jenis dan Ukuran Huruf
Jenis huruf yang dipakai dalam skripsi adalah Calibri dengan ketentuan
sebagai berikut:
● Judul bab pada level 1 berukuran 16 pt
● Judul subbab pada level 2 berukuran 14 pt
● Judul subbab pada level 3 berukuran 14 pt
● Judul subbab pada level 4 berukuran 12 pt
● Badan teks berukuran 12 pt
Penggunaan jenis dan ukuran ini harus konsisten. Untuk memudahkan
memelihara konsistensi sekaligus penyusunan struktur skripsi, fasilitas seperti
styles dan multilevel list dalam program pengolah kata dapat digunakan. Sebuah
template untuk skripsi ini telah disediakan untuk membantu mahasiswa. Styles
dan multilevel list dalam template tersebut sudah dirancang untuk jenis dan
ukuran huruf yang disyaratkan.
A.4 Spasi
Jarak standar antar baris dalam badan teks adalah satu spasi. Jarak antar
paragraf, antara judul bab dan judul subbab, antara judul subbab dan badan teks,
dan seterusnya, dapat dilihat pada masing-masing style yang digunakan dan
tersedia dalam template untuk skripsi ini.
A.5 Kepala Bab dan Subbab
Kepala bab terdiri dari kata “BAB” yang diikuti dengan nomor bab dan judul
dari bab tersebut, misalnya “BAB 1 PENDAHULUAN” . Kepala subbab diawali
dengan nomor sesuai tingkat hirarkinya dan diikuti dengan judul subbab,
misalnya “1.2 Rumusan masalah”. Penomoran subbab disarankan tidak lebih dari
15

4 level (maksimal subbab X.X.X.X). Kepala bab dan subbab tidak boleh
mengandung widow atau orphan sehingga nampak menggantung atau terputus
di bagian awal atau akhir sebuah halaman. Widow adalah sebuah paragraf
dengan hanya satu baris pertama pada akhir halaman sedangkan sisanya berada
pada halaman berikutnya. Orphan adalah baris terakhir dari satu paragraf yang
tertulis pada awal suatu halaman sedangkan baris lainnya dari paragraf tersebut
berada pada halaman sebelumnya.
A.6 Nomor Halaman
Bagian awal skripsi menggunakan nomor halaman berupa angka
Romawi kecil (i, ii, iii, iv, dan seterusnya) yang dimulai dari sampul dalam.
Sedangkan bagian utama dan bagian akhir skripsi menggunakan nomor halaman
berupa angka Arab (1, 2, 3, dan seterusnya) yang dimulai dari bab 1. Semua
nomor halaman diletakkan di tengah bawah.
16

LAMPIRAN B PENGGUNAAN BAHASA
Bahasa yang dipakai dalam skripsi adalah bahasa Bahasa Indonesia yang baku.
Setiap kalimat harus memiliki subjek dan predikat, dan umumnya dilengkapi juga
dengan objek, pelengkap, atau keterangan. Setiap paragraf biasanya terdiri dari
beberapa kalimat. Penuturan isi dalam kalimat, paragraf, maupun antar paragraf
harus menggunakan bahasa yang tepat dan menggambarkan alur logika yang
runtut.
Penulisan bahasa asing yang sudah diserap dalam Bahasa Indonesia
disesuaikan dengan kaidah Bahasa Indonesia. Sedapat mungkin dihindari
penggunaan bahasa asing jika istilah dalam bahasa Indonesia sudah ada. Jika
terpaksa menggunakan istilah dalam bahasa asing, maka penulisannya harus
sesuai ejaan aslinya dan dicetak miring (italic), kecuali jika istilah tersebut adalah
nama.
Sebagai referensi untuk penulisan Bahasa Indonesia yang baku, dokumen
berikut dapat digunakan:
● Kamus Bahasa Indonesia, Tim Penyusun, Pusat Bahasa Departemen
Pendidikan Nasional, Jakarta 2008
● Peraturan Menteri Pendidikan Nasional Republik Indonesia nomor 46 tahun
2009 tentang Pedoman Umum Ejaan Bahasa Indonesia yang Disempurnakan
● Kamus Besar Bahasa Indonesia dalam jaringan (KBBI daring):
http://bahasa.kemdiknas.go.id/kbbi/index.php
17
