# Keputusan Pemilihan Dataset

## Dataset utama: Polish Companies Bankruptcy, file 5year

UCI menyediakan lima subset berbeda untuk beberapa jarak prediksi. File `5year.arff` mewakili prediksi kebangkrutan satu tahun ke depan dan berisi 5.910 observasi, 410 perusahaan bangkrut, serta 5.500 non-bangkrut. Ukurannya menengah dan target tersedia langsung. Ada 64 rasio keuangan yang cukup untuk scorecard, model logistik, dan boosted trees. Lisensinya CC BY 4.0.

Audit lokal menemukan missingness dan outlier yang perlu ditangani, tetapi tidak membutuhkan rekonstruksi target dari status perusahaan.

## Kandidat awal yang ditolak: American Bankruptcy Dataset

Dataset Amerika terlihat menarik karena memiliki 78.682 baris perusahaan-tahun dan split temporal dalam paper. Audit file unduhan menunjukkan:

- `status_label` konstan pada semua tahun perusahaan, sehingga label baris tidak dapat dipakai mentah sebagai target one-year-ahead;
- `X14` (paper: current liabilities) dan `X17` (paper: total liabilities) identik pada seluruh 78.682 baris, sehingga interpretasi leverage/solvency tidak dapat dipercaya;
- kolom yang menurut definisi sumber seharusnya non-negatif juga memiliki banyak nilai negatif, termasuk long-term debt, receivables, dan market value;
- file memiliki 8.971 perusahaan unik, berbeda dari angka 8.262 pada paper.

Jumlah perusahaan berstatus failed dan distribusi baris terminal memang cocok dengan tabel event paper setelah digeser setahun, tetapi kecocokan itu tidak memperbaiki masalah pemetaan fitur. Karena fitur risiko kunci bermasalah, dataset ini tidak akan digunakan untuk model portofolio.

## Trade-off dataset terpilih

Polish dataset lebih kecil (5.910 baris) dan tidak memiliki ID perusahaan atau tanggal eksplisit pada ARFF yang tersedia. Karena itu kita menggunakan validasi terstratifikasi berulang dan holdout terstratifikasi, bukan mengklaim validasi temporal. Untuk memperkuat proyek, fokus pada missing-data governance, kalibrasi yang dibatasi konteks sampel, model interpretable vs challenger, dan keputusan review kapasitas.
