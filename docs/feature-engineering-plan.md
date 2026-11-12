# Feature Engineering dan Validasi

## Prinsip

Dataset sudah berisi 64 rasio finansial; jangan membuat ulang rasio tanpa memeriksa formula dan denominator yang membentuknya. Mapping ada di `feature-dictionary.md`. Mulai dari fitur yang diprofilkan, lalu seleksi berdasarkan validasi fold training dan makna finansial.

## Analisis kandidat

- Logistic regression dengan regularisasi L1/L2 sebagai scorecard interpretable.
- Gradient boosting sebagai challenger nonlinear.
- Bandingkan seluruh rasio dengan subset yang missing-nya lebih terkendali.
- Kelompokkan fitur menurut profitabilitas, likuiditas, leverage/solvabilitas, efisiensi, dan ukuran.
- Periksa korelasi, rasio ekstrem, duplikasi, dan stabilitas koefisien/importance antar fold.

## Missing dan outlier

Missingness berbeda menurut kelas: `Attr21` hilang pada 24% event bangkrut namun hanya 4 dari 5.500 non-event; `Attr27` hilang pada 30% event vs 5% non-event. Complete-case hanya mempertahankan 102 dari 410 event, sehingga bukan kandidat utama.

Bandingkan:

1. imputasi median yang dipelajari hanya pada training fold;
2. imputasi median plus missingness indicator;
3. sensitivitas tanpa `Attr37` (43,1% missing) dan tanpa atribut ekstrem;
4. complete-case hanya sebagai batas sensitivitas, dengan perubahan komposisi event dilaporkan.

Jangan gunakan SMOTE sebagai default. Jika diuji, lakukan hanya di dalam setiap training fold dan bandingkan dengan class weights. Evaluasi selalu pada prevalensi validasi yang tidak diubah.

## Protokol evaluasi

Tidak ada perusahaan ID atau year field yang dapat dipercaya dalam ARFF, sehingga jangan klaim out-of-time validation atau unseen-company validation. Deduplicate exact feature vectors sebelum split. Gunakan stratified nested cross-validation untuk pemilihan model dan satu holdout terstratifikasi yang ditetapkan sebelum tuning. Laporkan PR-AUC, recall dan precision pada top-k review capacity, ROC-AUC, Brier score, calibration curve, confusion matrix, serta interval ketidakpastian untuk event langka.

## Keputusan analis

Buat simulasi kapasitas review (misalnya top 1%, 5%, dan 10%) dan trade-off biaya false negative/false positive. Jelaskan asumsi biaya; jangan menampilkan threshold sebagai aturan universal. Karena data bukan sampel acak portofolio bank dan event rate-nya 6,94%, model score bukan estimasi PD yang siap dipakai tanpa validasi populasi dan koreksi prior.

## Tahap implementasi

- [x] Konversi ARFF ke CSV dengan `?` sebagai missing dan target `default_next_year`.
- [x] Hitung missingness per kelas dan simpan auditnya.
- [x] Deduplicate 60 exact duplicates sebelum split; panel modeling berisi 5.850 baris, 408 event, 5.442 non-event.
- [ ] Bangun pipeline preprocessing inside-fold dengan imputasi dan indikator missing.
- [ ] Fit baseline lalu challenger; kunci model sebelum membuka holdout.
